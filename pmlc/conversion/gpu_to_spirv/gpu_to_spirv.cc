// Copyright 2020 Intel Corporation

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"

#include "pmlc/conversion/gpu_to_spirv/pass_detail.h"
#include "pmlc/conversion/gpu_to_spirv/passes.h"
#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::gpu_to_spirv {
namespace stdx = dialect::stdx;

namespace {
/// Pass to lower to SPIRV that includes GPU, SCF, Std and Stdx dialects
template <class T>
struct StdxSubgroupBroadcastOpConversion final
    : public OpConversionPattern<stdx::SubgroupBroadcastOp> {
  using OpConversionPattern<stdx::SubgroupBroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stdx::SubgroupBroadcastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto stdxType = op.getResult().getType();
    auto spirvType = typeConverter->convertType(stdxType);
    rewriter.replaceOpWithNewOp<T>(op, spirvType, spirv::Scope::Subgroup,
                                   operands[0], operands[1]);

    return success();
  }
};

struct StdxSubgroupBlockReadINTELOpConversion
    : public OpConversionPattern<stdx::SubgroupBlockReadINTELOp> {
  using OpConversionPattern<
      stdx::SubgroupBlockReadINTELOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stdx::SubgroupBlockReadINTELOp blockReadOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    stdx::SubgroupBlockReadINTELOpAdaptor blockReadOperands(operands);
    auto loc = blockReadOp.getLoc();
    auto memrefType = blockReadOp.memref().getType().cast<MemRefType>();
    auto memrefElementType = memrefType.getElementType();
    auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();

    // Check if bitcast is needed, implementation should transform to block
    // ops only for certain types supported by the HW, that is why other types
    // are not checked here, stdx transform pass should handle those cases
    auto needBitcast = memrefElementType.isF16() || memrefElementType.isF32();

    if (needBitcast) {
      // Support inly fp16 and fp32 for now
      auto fType = memrefElementType.isF16() ? rewriter.getF16Type()
                                             : rewriter.getF32Type();
      auto iType = memrefElementType.isF16() ? rewriter.getIntegerType(16)
                                             : rewriter.getIntegerType(32);
      auto memrefIType = MemRefType::get(memrefType.getShape(), iType, {},
                                         memrefType.getMemorySpace());
      auto fToI_ptr = rewriter.create<spirv::BitcastOp>(
          loc, typeConverter.convertType(memrefIType),
          blockReadOperands.memref());

      auto loadPtr =
          spirv::getElementPtr(typeConverter, memrefIType, fToI_ptr.getResult(),
                               blockReadOperands.indices(), loc, rewriter);
      auto ptrType =
          loadPtr.component_ptr().getType().cast<spirv::PointerType>();

      auto blockOutMemType =
          blockReadOp.getResult().getType().dyn_cast<VectorType>();

      VectorType vecIType, vecFType;
      if (blockOutMemType) {
        vecIType = VectorType::get(blockOutMemType.getShape(), iType);
        vecFType = VectorType::get(blockOutMemType.getShape(), fType);
      }

      auto blockRead = rewriter.create<spirv::SubgroupBlockReadINTELOp>(
          loc,
          blockOutMemType ? typeConverter.convertType(vecIType)
                          : ptrType.getPointeeType(),
          loadPtr.component_ptr());

      auto UToF_val = rewriter.create<spirv::BitcastOp>(
          loc,
          blockOutMemType ? typeConverter.convertType(vecFType)
                          : blockReadOp.getResult().getType(),
          blockRead.getResult());

      rewriter.replaceOp(blockReadOp, {UToF_val});
    } else {
      auto loadPtr = spirv::getElementPtr(
          typeConverter, memrefType, blockReadOperands.memref(),
          blockReadOperands.indices(), blockReadOp.getLoc(), rewriter);
      auto ptrType =
          loadPtr.component_ptr().getType().cast<spirv::PointerType>();
      rewriter.replaceOpWithNewOp<spirv::SubgroupBlockReadINTELOp>(
          blockReadOp, ptrType.getPointeeType(), loadPtr.component_ptr());
    }
    return success();
  }
};

struct StdxSubgroupBlockWriteINTELOpConversion
    : public OpConversionPattern<stdx::SubgroupBlockWriteINTELOp> {
  using OpConversionPattern<
      stdx::SubgroupBlockWriteINTELOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stdx::SubgroupBlockWriteINTELOp blockWriteOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    stdx::SubgroupBlockWriteINTELOpAdaptor blockWriteOperands(operands);
    auto loc = blockWriteOp.getLoc();
    auto memrefType = blockWriteOp.memref().getType().cast<MemRefType>();
    auto memrefElementType = memrefType.getElementType();
    auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();

    // Check if bitcast is needed, implementation should transform to block
    // ops only for certain types supported by the HW, that is why other types
    // are not checked here, stdx transform pass should handle those cases
    auto needBitcast = memrefElementType.isF16() || memrefElementType.isF32();

    if (needBitcast) {
      // Support inly fp16 and fp32 for now
      auto iType = memrefElementType.isF16() ? rewriter.getIntegerType(16)
                                             : rewriter.getIntegerType(32);

      // Bitcast mem pointer
      auto memrefIType = MemRefType::get(memrefType.getShape(), iType, {},
                                         memrefType.getMemorySpace());
      auto fToI_ptr = rewriter.create<spirv::BitcastOp>(
          loc, typeConverter.convertType(memrefIType),
          blockWriteOperands.memref());

      // Bitcast value
      auto blockOutMemType =
          blockWriteOp.value().getType().dyn_cast<VectorType>();

      VectorType vecIType;
      if (blockOutMemType)
        vecIType = VectorType::get(blockOutMemType.getShape(), iType);

      auto fToI_val = rewriter.create<spirv::BitcastOp>(
          loc, blockOutMemType ? typeConverter.convertType(vecIType) : iType,
          blockWriteOperands.value());

      auto storePtr =
          spirv::getElementPtr(typeConverter, memrefIType, fToI_ptr.getResult(),
                               blockWriteOperands.indices(), loc, rewriter);

      rewriter.replaceOpWithNewOp<spirv::SubgroupBlockWriteINTELOp>(
          blockWriteOp, storePtr, fToI_val.getResult());
    } else {
      auto storePtr = spirv::getElementPtr(
          typeConverter, memrefType, blockWriteOperands.memref(),
          blockWriteOperands.indices(), blockWriteOp.getLoc(), rewriter);
      rewriter.replaceOpWithNewOp<spirv::SubgroupBlockWriteINTELOp>(
          blockWriteOp, storePtr, blockWriteOperands.value());
    }
    return success();
  }
};

// TODO: this is only temporary, move it to proper place leter
struct StdxTransferWriteOpConversion final
    : public OpConversionPattern<vector::TransferWriteOp> {
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferWriteOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    IVLOG(3, " " << debugString(*op));
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(op, op.source(), op.vector());
    return success();
  }
};

// Convert all allocations within SPIRV code to function local allocations.
// TODO: Allocations outside of threads but inside blocks should probably be
// shared memory, but currently we never generate such allocs.
struct AllocOpPattern final : public OpConversionPattern<AllocOp> {
public:
  using OpConversionPattern<AllocOp>::OpConversionPattern;

  AllocOpPattern(MLIRContext *context, SPIRVTypeConverter &typeConverter)
      : OpConversionPattern<AllocOp>(typeConverter, context, /*benefit=*/1000) {
  }

  LogicalResult
  matchAndRewrite(AllocOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
    MemRefType allocType = operation.getType();
    Type spirvType = typeConverter.convertType(allocType);
    rewriter.replaceOpWithNewOp<spirv::VariableOp>(
        operation, spirvType, spirv::StorageClass::Function, nullptr);
    return success();
  }
};

template <typename StdxOpTy, typename SpirvOpTy>
struct UnaryOpConversion : public OpConversionPattern<StdxOpTy> {
  using OpConversionPattern<StdxOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StdxOpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 1);
    auto dstType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<SpirvOpTy>(op, dstType, operands.front());
    return success();
  }
};

template <typename StdxOpTy, typename SpirvOpTy>
struct BinaryOpConversion : public OpConversionPattern<StdxOpTy> {
  using OpConversionPattern<StdxOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StdxOpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    auto dstType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<SpirvOpTy>(op, dstType, operands.front(),
                                           operands.back());
    return success();
  }
};

// ============================================================================
// GLSL SPIRV -> Standard SPIRV
// ============================================================================
template <typename GLSLOpTy, typename NegOpTy, typename LessThanOpTy>
struct GLSLAbsOpPattern final : public OpConversionPattern<GLSLOpTy> {
  using OpConversionPattern<GLSLOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GLSLOpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Value arg = operands[0];
    Type targetType = arg.getType();
    Value cst0 = rewriter.create<spirv::ConstantOp>(
        op.getLoc(), targetType, rewriter.getIntegerAttr(targetType, 0));
    Value negArg = rewriter.create<NegOpTy>(op.getLoc(), arg);
    Value isNeg = rewriter.create<LessThanOpTy>(op.getLoc(), arg, cst0);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, isNeg, negArg, arg);
    return success();
  }
};

using GLSLFAbsOpPattern = GLSLAbsOpPattern<spirv::GLSLFAbsOp, spirv::FNegateOp,
                                           spirv::FOrdLessThanOp>;
using GLSLSAbsOpPattern =
    GLSLAbsOpPattern<spirv::GLSLSAbsOp, spirv::SNegateOp, spirv::SLessThanOp>;

struct GPUToSPIRVCustomPass
    : public GPUToSPIRVCustomBase<GPUToSPIRVCustomPass> {
  GPUToSPIRVCustomPass() = default;
  explicit GPUToSPIRVCustomPass(bool nonUniformBroadcast) {
    this->nonUniformBroadcast = nonUniformBroadcast;
  }
  void runOnOperation() final {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    SmallVector<Operation *, 1> kernelModules;
    OpBuilder builder(context);
    module.walk([&builder, &kernelModules](gpu::GPUModuleOp moduleOp) {
      // For each kernel module (should be only 1 for now, but that is not a
      // requirement here), clone the module for conversion because the
      // gpu.launch function still needs the kernel module.
      builder.setInsertionPoint(moduleOp.getOperation());
      kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
    });

    auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
    std::unique_ptr<ConversionTarget> target =
        spirv::SPIRVConversionTarget::get(targetAttr);

    SPIRVTypeConverter typeConverter(targetAttr);
    ScfToSPIRVContext scfContext;
    OwningRewritePatternList patterns;
    populateGPUToSPIRVPatterns(context, typeConverter, patterns);
    populateSCFToSPIRVPatterns(context, typeConverter, scfContext, patterns);
    populateVectorToSPIRVPatterns(context, typeConverter, patterns);
    populateStandardToSPIRVPatterns(context, typeConverter, patterns);

    if (nonUniformBroadcast) {
      IVLOG(3, "GPUToSPIRVCustomPass: Using group non-uniform broadcast op");
      patterns.insert<
          StdxSubgroupBroadcastOpConversion<spirv::GroupNonUniformBroadcastOp>>(
          typeConverter, context);
    } else {
      IVLOG(3, "GPUToSPIRVCustomPass: Using group broadcast op");
      patterns
          .insert<StdxSubgroupBroadcastOpConversion<spirv::GroupBroadcastOp>>(
              typeConverter, context);
    }
    populateStdxToSPIRVPatterns(context, typeConverter, patterns);
    patterns.insert<AllocOpPattern>(context, typeConverter);
    if (spirv::getMemoryModel(targetAttr) == spirv::MemoryModel::GLSL450)
      populateStdxToSPIRVGLSLPatterns(context, typeConverter, patterns);
    if (spirv::getMemoryModel(targetAttr) != spirv::MemoryModel::GLSL450) {
      populateCustomGLSLToStdSpirvPatterns(context, typeConverter, patterns);
      target->addIllegalOp<spirv::GLSLFAbsOp, spirv::GLSLSAbsOp,
                           spirv::GLSLExpOp>();
    }
    if (spirv::getMemoryModel(targetAttr) == spirv::MemoryModel::OpenCL)
      populateCustomStdToOCLSpirvPatterns(context, typeConverter, patterns);

    if (failed(
            applyFullConversion(kernelModules, *target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void populateStdxToSPIRVPatterns(MLIRContext *context,
                                 SPIRVTypeConverter &typeConverter,
                                 OwningRewritePatternList &patterns) {
  patterns.insert<StdxSubgroupBlockReadINTELOpConversion,
                  StdxSubgroupBlockWriteINTELOpConversion,
                  StdxTransferWriteOpConversion>(typeConverter, context);
}

void populateStdxToSPIRVGLSLPatterns(MLIRContext *context,
                                     SPIRVTypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  patterns.insert<UnaryOpConversion<stdx::RoundOp, spirv::GLSLRoundOp>,
                  UnaryOpConversion<stdx::FloorOp, spirv::GLSLFloorOp>,
                  UnaryOpConversion<stdx::TanOp, spirv::GLSLTanOp>,
                  UnaryOpConversion<stdx::SinHOp, spirv::GLSLSinhOp>,
                  UnaryOpConversion<stdx::CosHOp, spirv::GLSLCoshOp>,
                  UnaryOpConversion<stdx::ASinOp, spirv::GLSLAsinOp>,
                  UnaryOpConversion<stdx::ACosOp, spirv::GLSLAcosOp>,
                  UnaryOpConversion<stdx::ATanOp, spirv::GLSLAtanOp>,
                  BinaryOpConversion<stdx::PowOp, spirv::GLSLPowOp>>(
      typeConverter, context);
}

void populateCustomGLSLToStdSpirvPatterns(MLIRContext *context,
                                          SPIRVTypeConverter &typeConverter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<GLSLFAbsOpPattern, GLSLSAbsOpPattern>(typeConverter, context);
}

void populateCustomStdToOCLSpirvPatterns(MLIRContext *context,
                                         SPIRVTypeConverter &typeConverter,
                                         OwningRewritePatternList &patterns) {
  patterns.insert<UnaryOpConversion<math::ExpOp, spirv::OCLExpOp>>(
      typeConverter, context);
}

std::unique_ptr<Pass> createGPUToSPIRVCustomPass() {
  return std::make_unique<GPUToSPIRVCustomPass>();
}

std::unique_ptr<Pass> createGPUToSPIRVCustomPass(bool nonUniformBroadcast) {
  return std::make_unique<GPUToSPIRVCustomPass>(nonUniformBroadcast);
}

} // namespace pmlc::conversion::gpu_to_spirv
