// Copyright 2020 Intel Corporation
#include "pmlc/target/x86/pipeline.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/scf_to_omp/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/layer/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/stencil.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/target/x86/heatmap.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/transforms/passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

#include "omp.h" // NOLINT

#include "pmlc/target/x86/utils.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace layer = dialect::layer;
namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;
namespace tile = dialect::tile;
namespace xsmm = dialect::xsmm;

namespace {

struct LowerPXAToAffinePass
    : public ConvertPXAToAffineBase<LowerPXAToAffinePass> {
  void runOnOperation() final {
    auto &ctx = getContext();
    conversion::pxa_to_affine::PXAToAffineConversionTarget target(ctx);
    target.addLegalDialect<xsmm::XSMMDialect>();

    OwningRewritePatternList patterns;
    populatePXAGemmToXSMMConversionPatterns(patterns, &ctx);
    populatePXAPrngToAffineConversionPatterns(patterns, &ctx);
    conversion::pxa_to_affine::populatePXAToAffineConversionPatterns(patterns,
                                                                     &ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      getOperation().dump();
      emitError(UnknownLoc::get(&ctx), "Error lowering pxa -> affine\n");
      signalPassFailure();
    }
  }
};

struct ConvertStandardToLLVMPass
    : public ConvertStandardToLLVMBase<ConvertStandardToLLVMPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = module.getContext();

    LowerToLLVMOptions options = {
        /*useBarePtrCallConv=*/false,
        /*emitCWrappers=*/true,
        /*indexBitwidth=*/kDeriveIndexBitwidthFromDataLayout,
        /*useAlignedAlloc=*/false,
    };
    LLVMTypeConverter typeConverter(context, options);

    OwningRewritePatternList patterns;
    populateExpandTanhPattern(patterns, context);
    populateXSMMToLLVMConversionPatterns(typeConverter, patterns);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(
        typeConverter, patterns);
    populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(*context);
    target.addDynamicallyLegalOp<omp::ParallelOp>([&](omp::ParallelOp op) {
      return typeConverter.isLegal(&op.getRegion());
    });
    target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                      omp::BarrierOp, omp::TaskwaitOp>();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// OpenMP has issues passing values through to OpenMP blocks.  As a workaround,
// we have a simple pass to smuggle values that cross the boundary via an
// alloca()'d struct.
struct OpenMPWorkaroundPass final
    : public OpenMPWorkaroundBase<OpenMPWorkaroundPass> {
  void runOnOperation() final {
    LLVM::LLVMFuncOp funcOp = getOperation();
    OpBuilder builder{&getContext()};
    funcOp.walk([&](omp::ParallelOp parOp) {
      llvm::SetVector<Value> values;

      visitUsedValuesDefinedAbove({parOp.region()}, [&](OpOperand *opOperand) {
        Value value = opOperand->get();

        // If it's not an LLVM pointer type, we don't need or want to smuggle
        // this value in via a struct.
        Type llvmType = value.getType();
        if (llvmType.isa<LLVM::LLVMPointerType>()) {
          return;
        }

        // Otherwise, we need to smuggle the value through an alloca'd
        // struct.
        values.insert(value);
      });

      if (values.empty()) {
        return; // Nothing to do.
      }

      // Build the structure.
      builder.setInsertionPoint(parOp);
      SmallVector<Type, 8> types;
      for (Value value : values) {
        types.push_back(value.getType());
      }
      auto structTy = LLVM::LLVMStructType::getLiteral(&getContext(), types);
      auto structPtrTy = LLVM::LLVMPointerType::get(structTy);
      auto numElements = builder.create<LLVM::ConstantOp>(
          parOp.getLoc(), builder.getI64Type(), builder.getIndexAttr(1));
      auto structPtr = builder.create<LLVM::AllocaOp>(
          parOp.getLoc(), structPtrTy, numElements, 0);
      Value srcStructVal =
          builder.create<LLVM::UndefOp>(parOp.getLoc(), structTy);
      for (auto srcIdx : llvm::enumerate(values)) {
        srcStructVal = builder.create<LLVM::InsertValueOp>(
            parOp.getLoc(), srcStructVal, srcIdx.value(),
            builder.getI64ArrayAttr(srcIdx.index()));
      }
      builder.create<LLVM::StoreOp>(parOp.getLoc(), srcStructVal, structPtr);

      // Unpack the structure, rewriting the affected values.
      builder.setInsertionPointToStart(&parOp.region().front());
      auto dstStructVal =
          builder.create<LLVM::LoadOp>(parOp.getLoc(), structPtr);
      for (auto srcIdx : llvm::enumerate(values)) {
        auto smuggledValue = builder.create<LLVM::ExtractValueOp>(
            parOp.getLoc(), srcIdx.value().getType(), dstStructVal,
            builder.getI64ArrayAttr(srcIdx.index()));
        replaceAllUsesInRegionWith(srcIdx.value(), smuggledValue,
                                   parOp.region());
      }
    });
  }
};

} // namespace

// NOTE: the stencil pass uses row-major ordering, the heatmap is
// specified in column-major ordering.
static pxa::StencilCost heatmapCostTransposed(ArrayRef<int64_t> tile,
                                              ArrayRef<Type> types) {
  // Only f32 is supported currently.
  if (llvm::any_of(types, [](Type type) {
        if (auto shapedType = type.dyn_cast<ShapedType>()) {
          type = shapedType.getElementType();
        }
        return !type.isF32();
      })) {
    return pxa::StencilCost{/*throughput=*/0.0, /*startupCost=*/0};
  }
  return heatmapCost(ArrayRef<int64_t>{tile[1], tile[0], tile[2]});
}

struct XSMMStencilPass : public XSMMStencilBase<XSMMStencilPass> {
  XSMMStencilPass() = default;
  XSMMStencilPass(unsigned numThreads, bool isBatched) {
    this->numThreads = numThreads;
    this->isBatched = isBatched;
  }

  void runOnFunction() final {
    if (!numThreads.getValue()) {
      numThreads = std::thread::hardware_concurrency();
    }
    IVLOG(3, "XSMMStencilPass> numThreads: " << numThreads.getValue());
    IVLOG(3, "XSMMStencilPass> isBatched: " << isBatched.getValue());
    getFunction().walk([this](AffineParallelOp op) {
      pxa::applyStencilGEMM(op, numThreads.getValue(), isBatched.getValue(),
                            heatmapCostTransposed);
    });
  }
};

std::unique_ptr<Pass> createXSMMStencilPass(unsigned numThreads,
                                            bool isBatched) {
  return std::make_unique<XSMMStencilPass>(numThreads, isBatched);
}

std::unique_ptr<Pass> createXSMMStencilPass() {
  return std::make_unique<XSMMStencilPass>();
}

std::unique_ptr<Pass> createLowerPXAToAffinePass() {
  return std::make_unique<LowerPXAToAffinePass>();
}

std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<ConvertStandardToLLVMPass>();
}

std::unique_ptr<Pass> createOpenMPWorkaroundPass() {
  return std::make_unique<OpenMPWorkaroundPass>();
}

void pipelineBuilder(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(layer::createInlineLayersPass());
  pm.addNestedPass<FuncOp>(tile::createComputeBoundsPass());
  pm.addPass(tile::createSplitMainPass());
  pm.addPass(transforms::createHoistingPass());
  pm.addNestedPass<FuncOp>(tile::createPadConstraintsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addNestedPass<FuncOp>(tile::createScfForReplacePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<FuncOp>(layer::createInlineLayersPass());

  pm.addNestedPass<FuncOp>(
      createXSMMStencilPass(/*numThreads=*/1, /*isBatched=*/true));
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(pxa::createTileAccumulatePass());
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass(/*promote=*/false));
  pm.addPass(createCanonicalizerPass());

  // Use OMP thread count
  unsigned maxThreads = omp_get_max_threads();
  unsigned physCores = getPhysicalCoreNumber();
  if (0 != physCores) {
    maxThreads = std::min(physCores, maxThreads);
  }

  pm.addNestedPass<FuncOp>(pxa::createCPUThreadPass(maxThreads));

  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(pxa::createFusionPass());
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(pxa::createMemRefDataFlowOptPass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(pxa::createLocalizePass());
  pm.addNestedPass<FuncOp>(pxa::createResizeTmpsPass());
  pm.addPass(pxa::createDeallocPlacementPass());
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerPXAToAffinePass());

  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::scf_to_omp::createLowerSCFToOpenMPPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerToCFGPass());
  if (pmlc::util::getEnvVar("PLAIDML_BOUNDS_CHECK") == "1") {
    pm.addPass(stdx::createBoundsCheckPass());
  }

  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createTraceLinkingPass());
  pm.addNestedPass<LLVM::LLVMFuncOp>(createOpenMPWorkaroundPass());
}

} // namespace pmlc::target::x86
