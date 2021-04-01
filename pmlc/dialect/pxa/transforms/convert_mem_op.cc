// Copyright 2021 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

namespace {

bool isLoopArgumentOrConstant(Value value) {
  if (auto arg = value.dyn_cast<BlockArgument>()) {
    return isa<AffineParallelOp>(arg.getOwner()->getParentOp());
  }
  if (auto defOp = value.getDefiningOp()) {
    return isa<ConstantOp>(defOp);
  }
  return false;
}

template <typename MemOp>
struct MemOpPattern final : public OpRewritePattern<MemOp> {
  using OpRewritePattern<MemOp>::OpRewritePattern;

  void replaceMemOp(MemOp op, PatternRewriter &rewriter) const {
    auto idxs = op.getIndices();
    auto affineMap =
        AffineMap::getMultiDimIdentityMap(idxs.size(), op.getContext());
    auto memRef = op.getMemRef();
    if (isa<LoadOp>(op.getOperation())) {
      // Convert to PxaLoadOp
      rewriter.replaceOpWithNewOp<PxaLoadOp>(op, memRef, affineMap, idxs);
      return;
    }
    if (auto store = dyn_cast<StoreOp>(op.getOperation())) {
      // Convert to PxaReduceOp
      auto result = rewriter.create<PxaReduceOp>(
          op.getLoc(), AtomicRMWKind::assign, store.getValueToStore(), memRef,
          affineMap, idxs);
      if (auto loop =
              dyn_cast<AffineParallelOp>(op.getOperation()->getParentOp())) {
        // Fix affine.yield
        Operation &yield = loop.getBody()->back();
        if (auto yieldOp = dyn_cast<AffineYieldOp>(&yield)) {
          auto operands = yieldOp.operands();
          for (unsigned i = 0; i < operands.size(); ++i) {
            if (operands[i] == memRef) {
              yieldOp.setOperand(i, result);
            }
          }
        }
      }
      op.erase();
      return;
    }
    op->emitOpError("Op is not Load or Store.");
  }

  LogicalResult matchAndRewrite(MemOp op, PatternRewriter &rewriter) const {
    bool affineIdxs = true;
    auto idxs = op.getIndices();
    // If all indices are constant or from loop arguments, the operation can be
    // converted to PXA
    for (auto idx : idxs) {
      if (!isLoopArgumentOrConstant(idx)) {
        affineIdxs = false;
        break;
      }
    }
    if (affineIdxs) {
      replaceMemOp(op, rewriter);
      return success();
    }
    return failure();
  }
};

struct ConvertMemOpPass : public ConvertMemOpBase<ConvertMemOpPass> {
public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<MemOpPattern<LoadOp>, MemOpPattern<StoreOp>>(context);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertMemOpPass() {
  return std::make_unique<ConvertMemOpPass>();
}

} // namespace pmlc::dialect::pxa
