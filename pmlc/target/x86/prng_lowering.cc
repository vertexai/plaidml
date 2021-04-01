// Copyright 2020, Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "pmlc/dialect/pxa/ir/ops.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;

namespace {

struct PrngOpConversion : public OpConversionPattern<pxa::PrngOp> {
  using OpConversionPattern<pxa::PrngOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pxa::PrngOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    pxa::PrngOp::Adaptor transformed(operands);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();

    auto resultUnrankedType =
        UnrankedMemRefType::get(rewriter.getF32Type(), /*memorySpace=*/0);
    auto stateUnrankedType =
        UnrankedMemRefType::get(rewriter.getIntegerType(32), /*memorySpace=*/0);
    auto symbol = getOrInsertFunc(rewriter, module, rewriter.getF32Type(), loc,
                                  resultUnrankedType, stateUnrankedType);

    auto resultCast = rewriter.create<MemRefCastOp>(loc, transformed.tensor(),
                                                    resultUnrankedType);
    auto stateCast = rewriter.create<MemRefCastOp>(loc, transformed.state(),
                                                   stateUnrankedType);
    auto newStateCast = rewriter.create<MemRefCastOp>(
        loc, transformed.new_state(), stateUnrankedType);

    rewriter.create<CallOp>(
        loc, symbol, ArrayRef<Type>{},
        ArrayRef<Value>{stateCast, resultCast, newStateCast});

    op.result_tensor().replaceAllUsesWith(transformed.tensor());
    op.result_state().replaceAllUsesWith(transformed.new_state());

    rewriter.eraseOp(op);

    return success();
  }

private:
  FlatSymbolRefAttr
  getOrInsertFunc(ConversionPatternRewriter &rewriter, ModuleOp module,
                  Type elementType, Location loc,
                  UnrankedMemRefType resultUnrankedType,
                  UnrankedMemRefType stateUnrankedType) const {
    const char *symbol = "plaidml_rt_prng";
    auto context = module.getContext();
    if (module.lookupSymbol(symbol)) {
      return SymbolRefAttr::get(context, symbol);
    }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto funcType = rewriter.getFunctionType(ArrayRef<Type>{stateUnrankedType,
                                                            resultUnrankedType,
                                                            stateUnrankedType},
                                             ArrayRef<Type>{});
    rewriter.create<FuncOp>(loc, symbol, funcType, ArrayRef<NamedAttribute>{})
        .setPrivate();
    return SymbolRefAttr::get(context, symbol);
  }
};

} // namespace

void populatePXAPrngToAffineConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<PrngOpConversion>(ctx);
}

} // namespace pmlc::target::x86
