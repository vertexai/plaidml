// Copyright 2019, Intel Corporation

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/Support/MathExtras.h"

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/padding.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/math/util.h"
#include "pmlc/util/strides.h"

namespace pmlc::dialect::tile {

using namespace mlir; // NOLINT

namespace {

struct PadRangesPass : public PadRangesBase<PadRangesPass> {
  void runOnFunction() final;
};

void PadRangesPass::runOnFunction() {
  assert(maxPowerOfTwo >= minPowerOfTwo);
  assert(util::math::IsPo2(maxPowerOfTwo));
  assert(util::math::IsPo2(minPowerOfTwo));
  assert(maxIncrease >= 0.0);
  assert(maxIncrease < 1.0);
  auto func = getFunction();
  func.walk([&](ContractionOp op) {
    // Skip some cases where the padding pass can't operate.
    if (op.getNumSymbols()) {
      IVLOG(3, "padding cannot be run on symbolic contractions");
      return;
    }
    if (!op.lowerBounds() || !op.upperBounds()) {
      IVLOG(3, "contraction bounds must be computed");
      return;
    }
    // Extract which dimensions of the contraction are which output dimension
    unsigned outRank = op.getResultType().getRank();
    SmallVector<unsigned, 4> outIdx(outRank);
    for (unsigned i = 0; i < outRank; i++) {
      auto outDim = op.sink().getResult(i).dyn_cast<AffineDimExpr>();
      if (!outDim) {
        IVLOG(3, "Invalid output access");
        return;
      }
      outIdx[i] = outDim.getPosition();
    }
    // Verify bounds are simple
    unsigned indexCount = op.lowerBounds()->getNumResults();
    SmallVector<int64_t, 6> ranges;
    for (unsigned i = 0; i < indexCount; i++) {
      auto lowerConst =
          op.lowerBounds()->getResult(i).dyn_cast<AffineConstantExpr>();
      auto upperConst =
          op.upperBounds()->getResult(i).dyn_cast<AffineConstantExpr>();
      if (!lowerConst || lowerConst.getValue() != 0) {
        IVLOG(3, "Invalid lower bound");
        return;
      }
      if (!upperConst) {
        IVLOG(3, "Invalid lower bound");
        return;
      }
      ranges.push_back(upperConst.getValue() + 1);
    }
    // Get strides for output
    auto outStrides = util::computeStrideArray(op.getResultType(), op.sink());
    if (!outStrides) {
      IVLOG(3, "Unable to compute output strides");
      return;
    }
    IVLOG(2, "Pad Ranges: outStrides = " << *outStrides);
    // Decide wthat to round up
    bool didChange = false;
    SmallVector<int64_t, 6> newRanges = ranges;
    for (unsigned i = 0; i < indexCount; i++) {
      if (outStrides->strides[i] == 0) {
        continue;
      }
      for (int64_t po2 = maxPowerOfTwo; po2 >= minPowerOfTwo; po2 /= 2) {
        int64_t roundedRange = llvm::divideCeil(ranges[i], po2) * po2;
        if (roundedRange == ranges[i]) {
          break;
        }
        int64_t increase = roundedRange - ranges[i];
        float percIncrease =
            static_cast<float>(increase) / static_cast<float>(ranges[i]);
        IVLOG(3, "Pad Ranges: Orig: " << ranges[i]
                                      << ", rounded: " << roundedRange
                                      << ", percIncrease = " << percIncrease);
        if (percIncrease < maxIncrease) {
          newRanges[i] = roundedRange;
          didChange = true;
          break;
        }
      }
    }
    // Exit early if nothing is changing
    if (!didChange) {
      return;
    }
    // Start building a new contraction to 'shrink'
    OpBuilder builder(op.getContext());
    builder.setInsertionPointAfter(op.getOperation());
    // Make shrinking contraction
    Type elementType = op.getResultType().getElementType();
    auto ident = tile::createIdentity(builder, op.getLoc(), elementType,
                                      AggregationKind::assign);
    auto idMap = AffineMap::getMultiDimIdentityMap(outRank, op.getContext());
    auto newOp = builder.create<ContractionOp>(
        op.getLoc(), op.getResultType(), ident, ArrayRef<Value>{op},
        AggregationKind::assign, CombinationKind::none, idMap,
        ArrayRef<AffineMap>{idMap},
        IntegerSet::getEmptySet(outRank, 0, op.getContext()), "shrink");
    newOp.setLowerBounds(SmallVector<int64_t, 4>(outRank, 0));
    SmallVector<int64_t, 4> newOutSize;
    SmallVector<int64_t, 4> shrinkUpperBounds;
    for (unsigned i = 0; i < outRank; i++) {
      newOutSize.push_back(newRanges[outIdx[i]]);
      shrinkUpperBounds.push_back(ranges[outIdx[i]] - 1);
    }
    newOp.setUpperBounds(shrinkUpperBounds);
    // Switch all uses to the new contracton
    op.getResult().replaceAllUsesExcept(
        newOp.getResult(), SmallPtrSet<Operation *, 1>{newOp.getOperation()});
    // Resize + rerange the output of the original op
    op.getResult().setType(RankedTensorType::get(newOutSize, elementType));
    // Convert ranges to bounds
    SmallVector<int64_t, 4> newBounds;
    for (unsigned i = 0; i < indexCount; i++) {
      newBounds.push_back(newRanges[i] - 1);
    }
    op.setUpperBounds(newBounds);
  });
}

} // namespace

std::unique_ptr<Pass> createPadRangesPass() {
  return std::make_unique<PadRangesPass>();
}

} // namespace pmlc::dialect::tile
