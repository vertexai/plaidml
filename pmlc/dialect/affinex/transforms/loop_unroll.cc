// Copyright 2020 Intel Corporation

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/LoopUtils.h"
#include "pmlc/dialect/affinex/transforms/pass_detail.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::affinex {

struct AffinexLoopUnroll : public AffinexLoopUnrollBase<AffinexLoopUnroll> {
  explicit AffinexLoopUnroll(uint64_t operationLimit) {
    this->operationLimit = operationLimit;
  }

  void runOnFunction() override {
    DenseMap<Operation *, uint64_t> opCount;
    SmallVector<AffineForOp, 4> loopsToUnroll;

    getFunction().walk([&](Operation *op) {
      auto count = opCount[op];
      if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
        Optional<uint64_t> tripCount = getConstantTripCount(forOp);
        if (tripCount.hasValue()) {
          count *= tripCount.getValue();
          if (count <= operationLimit.getValue()) {
            loopsToUnroll.push_back(forOp);
          }
        } else {
          count = operationLimit.getValue() + 1;
        }
      } else if (isa<AffineYieldOp>(op)) {
        // don't count these op(s)
        count = 0;
      } else if (isa<AffineParallelOp>(op)) {
        // stop unrolling above these op(s)
        count = operationLimit.getValue() + 1;
      } else {
        // otherwise, count the op
        count++;
      }
      opCount[op->getParentOp()] += count;
    });

    for (auto forOp : loopsToUnroll) {
      // TODO: check LogicalResult
      (void)loopUnrollFull(forOp);
    }
  }
};

std::unique_ptr<Pass> createAffinexLoopUnroll(uint64_t operationLimit) {
  return std::make_unique<AffinexLoopUnroll>(operationLimit);
}
} // namespace pmlc::dialect::affinex
