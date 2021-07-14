// Copyright 2021, Intel Corporation

#include <limits>
#include <utility>

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/conversion/linalg_to_tile/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

#include "pmlc/util/ident.h"

namespace pmlc::conversion::linalg_to_tile {

namespace layer = dialect::layer;
namespace stdx = dialect::stdx;
namespace tile = dialect::tile;

using namespace mlir; // NOLINT

using util::AggregationKind;
using util::CombinationKind;

namespace {

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    return success();
  }
};

struct LowerLinalgToTilePass
    : public LowerLinalgToTileBase<LowerLinalgToTilePass> {
  void runOnOperation() final {
    // Set up target (i.e. what is legal)
    ConversionTarget target(getContext());
    LinalgToTileTypeConverter converter;
    target.addLegalDialect<mlir::StandardOpsDialect, //
                           mlir::math::MathDialect,  //
                           layer::LayerDialect,      //
                           tile::TileDialect>();
    target.addLegalOp<scf::ForOp,   //
                      scf::YieldOp, //
                      scf::IfOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });

    // Setup rewrite patterns
    RewritePatternSet patterns(&getContext());
    patterns.insert<FuncOpConversion>(&getContext());

    populateLinalgToTileSpecialPatterns(patterns);

    // Run the conversion
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLowerLinalgToTilePass() {
  return std::make_unique<LowerLinalgToTilePass>();
}

} // namespace pmlc::conversion::linalg_to_tile
