// Copyright 2020 Intel Corporation

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "pmlc/target/intel_gen/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"

using namespace mlir;         // NOLINT
using namespace mlir::gpu;    // NOLINT
using namespace mlir::vector; // NOLINT

namespace pmlc::target::intel_gen {

namespace {
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallVector;

class AffineParallelLowering : public OpRewritePattern<AffineParallelOp> {
public:
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  explicit AffineParallelLowering(mlir::MLIRContext *context)
      : OpRewritePattern<AffineParallelOp>(context, /*benefit=*/1000) {}

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 8> steps;
    SmallVector<Value, 8> upperBoundTuple;
    SmallVector<Value, 8> lowerBoundTuple;
    SmallVector<ParallelLoopDimMapping, 8> mappings;
    // Finding lower and upper bound by expanding the map expression.
    // Checking if expandAffineMap is not giving NULL.
    Optional<SmallVector<Value, 8>> upperBound = expandAffineMap(
        rewriter, loc, op.upperBoundsMap(), op.getUpperBoundsOperands());
    Optional<SmallVector<Value, 8>> lowerBound = expandAffineMap(
        rewriter, loc, op.lowerBoundsMap(), op.getLowerBoundsOperands());
    if (!lowerBound || !upperBound)
      return failure();
    upperBoundTuple = *upperBound;
    lowerBoundTuple = *lowerBound;
    steps.reserve(op.steps().size());
    for (Attribute step : op.steps())
      steps.push_back(rewriter.create<ConstantIndexOp>(
          loc, step.cast<IntegerAttr>().getInt()));
    // Pick mapping level
    Processor proc = Processor::Sequential;
    if (hasUnitTag(op, gpuBlockTag())) {
      proc = Processor::BlockX;
    } else if (hasUnitTag(op, gpuThreadTag())) {
      proc = Processor::ThreadX;
    }
    if (proc != Processor::Sequential && steps.size() > 3) {
      op.emitError("Failed to lower to GPU due to lack of index packing");
      return failure();
    }
    // Creating empty scf.parallel op body with appropriate bounds.
    auto parallelOp = rewriter.create<scf::ParallelOp>(loc, lowerBoundTuple,
                                                       upperBoundTuple, steps);
    for (unsigned i = 0; i < steps.size(); i++) {
      mappings.push_back(getParallelLoopDimMappingAttr(
          proc, rewriter.getDimIdentityMap(), rewriter.getDimIdentityMap()));
      if (proc != Processor::Sequential) {
        proc = static_cast<Processor>(static_cast<int>(proc) + 1);
      }
    }
    if (failed(setMappingAttr(parallelOp, mappings)))
      return failure();
    copyTags(parallelOp, op);

    rewriter.eraseBlock(parallelOp.getBody());
    rewriter.inlineRegionBefore(op.region(), parallelOp.region(),
                                parallelOp.region().end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct IntelGenLowerAffinePass
    : public IntelGenLowerAffineBase<IntelGenLowerAffinePass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<AffineParallelLowering>(&getContext());
    populateAffineToStdConversionPatterns(patterns, &getContext());
    populateAffineToVectorConversionPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target
        .addLegalDialect<scf::SCFDialect, StandardOpsDialect, VectorDialect>();
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createIntelGenLowerAffinePass() {
  return std::make_unique<IntelGenLowerAffinePass>();
}
} // namespace pmlc::target::intel_gen
