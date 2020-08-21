// Copyright 2020, Intel Corporation

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/conversion/stdx_to_llvm/pass_detail.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/dialect/stdx/ir/ops.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::stdx_to_llvm {

namespace stdx = dialect::stdx;
namespace edsc = mlir::edsc;

namespace {

template <typename T>
struct LibMCallLowering : public ConvertOpToLLVMPattern<T> {
  using ConvertOpToLLVMPattern<T>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto f32Type = LLVM::LLVMType::getFloatTy(rewriter.getContext());
    SmallVector<LLVM::LLVMType, 2> argTypes(getArity(), f32Type);
    auto funcType = LLVM::LLVMType::getFunctionTy(f32Type, argTypes, false);
    auto sym = getOrInsertFuncOp(getFuncName(), funcType, op);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, ArrayRef<Type>{f32Type}, rewriter.getSymbolRefAttr(sym), operands);
    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFuncOp(StringRef funcName,
                                     LLVM::LLVMType funcType,
                                     Operation *op) const {
    using LLVM::LLVMFuncOp;

    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcName);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);

    mlir::OpBuilder builder(op->getParentOfType<LLVMFuncOp>());
    return builder.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  }

protected:
  virtual std::string getFuncName() const = 0;
  virtual size_t getArity() const { return 1; }
};

struct ACosLowering : public LibMCallLowering<stdx::ACosOp> {
  using LibMCallLowering<stdx::ACosOp>::LibMCallLowering;
  std::string getFuncName() const override { return "acosf"; }
};

struct ASinLowering : public LibMCallLowering<stdx::ASinOp> {
  using LibMCallLowering<stdx::ASinOp>::LibMCallLowering;
  std::string getFuncName() const override { return "asinf"; }
};

struct ATanLowering : public LibMCallLowering<stdx::ATanOp> {
  using LibMCallLowering<stdx::ATanOp>::LibMCallLowering;
  std::string getFuncName() const override { return "atanf"; }
};

struct CosHLowering : public LibMCallLowering<stdx::CosHOp> {
  using LibMCallLowering<stdx::CosHOp>::LibMCallLowering;
  std::string getFuncName() const override { return "coshf"; }
};

struct ErfLowering : public LibMCallLowering<stdx::ErfOp> {
  using LibMCallLowering<stdx::ErfOp>::LibMCallLowering;
  std::string getFuncName() const override { return "erff"; }
};

struct FloorLowering : public LibMCallLowering<stdx::FloorOp> {
  using LibMCallLowering<stdx::FloorOp>::LibMCallLowering;
  std::string getFuncName() const override { return "floorf"; }
};

struct PowLowering : public LibMCallLowering<stdx::PowOp> {
  using LibMCallLowering<stdx::PowOp>::LibMCallLowering;
  std::string getFuncName() const override { return "powf"; }
  size_t getArity() const override { return 2; }
};

struct RoundLowering : public LibMCallLowering<stdx::RoundOp> {
  using LibMCallLowering<stdx::RoundOp>::LibMCallLowering;
  std::string getFuncName() const override { return "roundf"; }
};

struct SinHLowering : public LibMCallLowering<stdx::SinHOp> {
  using LibMCallLowering<stdx::SinHOp>::LibMCallLowering;
  std::string getFuncName() const override { return "sinhf"; }
};

struct TanLowering : public LibMCallLowering<stdx::TanOp> {
  using LibMCallLowering<stdx::TanOp>::LibMCallLowering;
  std::string getFuncName() const override { return "tanf"; }
};

class BaseViewConversionHelper {
public:
  explicit BaseViewConversionHelper(Type type)
      : desc(MemRefDescriptor::undef(rewriter(), loc(), type)) {}

  explicit BaseViewConversionHelper(Value v) : desc(v) {}

  /// Wrappers around MemRefDescriptor that use EDSC builder and location.
  Value allocatedPtr() { return desc.allocatedPtr(rewriter(), loc()); }
  void setAllocatedPtr(Value v) { desc.setAllocatedPtr(rewriter(), loc(), v); }
  Value alignedPtr() { return desc.alignedPtr(rewriter(), loc()); }
  void setAlignedPtr(Value v) { desc.setAlignedPtr(rewriter(), loc(), v); }
  Value offset() { return desc.offset(rewriter(), loc()); }
  void setOffset(Value v) { desc.setOffset(rewriter(), loc(), v); }
  Value size(unsigned i) { return desc.size(rewriter(), loc(), i); }
  void setSize(unsigned i, Value v) { desc.setSize(rewriter(), loc(), i, v); }
  void setConstantSize(unsigned i, int64_t v) {
    desc.setConstantSize(rewriter(), loc(), i, v);
  }
  Value stride(unsigned i) { return desc.stride(rewriter(), loc(), i); }
  void setStride(unsigned i, Value v) {
    desc.setStride(rewriter(), loc(), i, v);
  }
  void setConstantStride(unsigned i, int64_t v) {
    desc.setConstantStride(rewriter(), loc(), i, v);
  }

  operator Value() { return desc; }

private:
  OpBuilder &rewriter() { return edsc::ScopedContext::getBuilderRef(); }
  Location loc() { return edsc::ScopedContext::getLocation(); }

  MemRefDescriptor desc;
};

struct ReshapeLowering : public ConvertOpToLLVMPattern<stdx::ReshapeOp> {
  using ConvertOpToLLVMPattern<stdx::ReshapeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto reshapeOp = cast<stdx::ReshapeOp>(op);
    MemRefType dstType = reshapeOp.getResult().getType().cast<MemRefType>();

    if (!dstType.hasStaticShape())
      return failure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto res = getStridesAndOffset(dstType, strides, offset);
    if (failed(res) || llvm::any_of(strides, [](int64_t val) {
          return ShapedType::isDynamicStrideOrOffset(val);
        }))
      return failure();

    edsc::ScopedContext context(rewriter, op->getLoc());
    stdx::ReshapeOpAdaptor adaptor(operands);
    BaseViewConversionHelper baseDesc(adaptor.tensor());
    BaseViewConversionHelper desc(typeConverter.convertType(dstType));
    desc.setAllocatedPtr(baseDesc.allocatedPtr());
    desc.setAlignedPtr(baseDesc.alignedPtr());
    desc.setOffset(baseDesc.offset());
    for (auto en : llvm::enumerate(dstType.getShape()))
      desc.setConstantSize(en.index(), en.value());
    for (auto en : llvm::enumerate(strides))
      desc.setConstantStride(en.index(), en.value());
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

/// A pass converting MLIR operations into the LLVM IR dialect.
struct LowerToLLVMPass : public LowerToLLVMBase<LowerToLLVMPass> {
  // Run the dialect converter on the module.
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto context = module.getContext();
    LLVMTypeConverter typeConverter(context);

    OwningRewritePatternList patterns;
    populateLoopToStdConversionPatterns(patterns, context);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    populateStdXToLLVMConversionPatterns(typeConverter, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateStdXToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<ACosLowering,    //
                  ASinLowering,    //
                  ATanLowering,    //
                  CosHLowering,    //
                  ErfLowering,     //
                  FloorLowering,   //
                  PowLowering,     //
                  ReshapeLowering, //
                  RoundLowering,   //
                  SinHLowering,    //
                  TanLowering>(converter);
}

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}

} // namespace pmlc::conversion::stdx_to_llvm
