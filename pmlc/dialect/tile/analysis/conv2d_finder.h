// Copyright 2021, Intel Corporation

#pragma once

#include <string>
#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "pmlc/dialect/tile/ir/ops.h"

namespace pmlc::dialect::tile {

using namespace mlir; // NOLINT

class Conv2dFinder {
public:
  explicit Conv2dFinder(ContractionOp op) { isOpConv2d = isaConv2D(op); }

  // returns whether the contract op is a Conv2d
  bool isContractOpConv2d() { return isOpConv2d; }

  // returns reason if NOT a Conv2d, else empty
  std::string getReason() { return reason; }

  // returns paddings if a Conv2d, else empty
  SmallVector<int64_t, 2> getPaddings() { return paddings; }

  // returns strides if a Conv2d, else empty
  SmallVector<int64_t, 2> getStrides() { return strides; }

  // returns dilations if a Conv2d, else empty
  SmallVector<int64_t, 2> getDilations() { return dilations; }

private:
  bool isOpConv2d{false};
  std::string reason;
  static const unsigned numTotalDims{4};
  static const unsigned numSpatialDims{2};
  SmallVector<int64_t, numSpatialDims> paddings;
  SmallVector<int64_t, numSpatialDims> strides;
  SmallVector<int64_t, numSpatialDims> dilations;
  typedef llvm::SmallVector<int64_t, 8> flatExpr;

  bool isaConv2D(ContractionOp op) {
    if (op.combo() != CombinationKind::mul) {
      reason = "Invalid CombinationKind";
      return false;
    }

    if (op.agg() != AggregationKind::add) {
      reason = "Invalid AggregationKind";
      return false;
    }

    // assume activation is 0, kernel is 1
    assert(op.getNumTensors() == 2);
    AffineMap act = op.getSourceMap(0);
    AffineMap ker = op.getSourceMap(1);
    AffineMap out = op.sink();

    // assume N is dimension 0 of activation, output
    unsigned nPos = 0;
    if (act.getResult(nPos) != out.getResult(nPos)) {
      reason = "Unable to find batch dimension";
      return false;
    }

    // look for Ci
    // skip dimension 0 (nPos) of the activation
    unsigned actCiPos = 1;
    unsigned kerCiPos = 0;
    if (!findCommonResult(act, ker, actCiPos, kerCiPos)) {
      reason = "Unable to find channel-in dimension";
      return false;
    }

    // look for Co
    // skip dimension 0 (nPos) of the output
    unsigned outCoPos = 1;
    unsigned kerCoPos = 0;
    if (!findCommonResult(out, ker, outCoPos, kerCoPos)) {
      reason = "Unable to find channel-out dimension";
      return false;
    }

    // activation, kernel, output must be 4D tensors
    if (act.getNumResults() != numTotalDims ||
        ker.getNumResults() != numTotalDims ||
        out.getNumResults() != numTotalDims) {
      reason = "Invalid tensor rank";
      return false;
    }

    // calculate spatial dimensions
    llvm::SmallVector<unsigned, numSpatialDims> actSpatial, kerSpatial,
        outSpatial;
    for (unsigned i = 0; i < numTotalDims; ++i) {
      // activation - exclude batch and channel in dimensions
      if (i != nPos && i != actCiPos) {
        actSpatial.push_back(i);
      }
      // kernel - exclude channel in and out dimensions
      if (i != kerCiPos && i != kerCoPos) {
        kerSpatial.push_back(i);
      }
      // output - exclude batch and channel out dimensions
      if (i != nPos && i != outCoPos) {
        outSpatial.push_back(i);
      }
    }

    if (actSpatial.size() != numSpatialDims ||
        kerSpatial.size() != numSpatialDims ||
        outSpatial.size() != numSpatialDims) {
      reason = "Invalid spatial rank";
      return false;
    }

    // flatten the Affine expressions in each AffineMap
    std::vector<flatExpr> actFlatExprs, kerFlatExprs, outFlatExprs;
    if (failed(getFlattenedAffineExprs(act, &actFlatExprs, nullptr)) ||
        failed(getFlattenedAffineExprs(ker, &kerFlatExprs, nullptr)) ||
        failed(getFlattenedAffineExprs(out, &outFlatExprs, nullptr))) {
      reason = "Unable to get flattened Affine expressions";
      return false;
    }

    // Canonical NHWC example:
    // affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
    // affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4 * 3 - 2, d2 *
    // 2 + d5 * 3 - 2, d6)> affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5,
    // d6, d3)>
    //
    // N = batch
    // Y = output Y
    // X = output X
    // Co = Channel out
    // Ky = Kernel Y
    // Kx = Kernel X
    // Ci = Channel in
    //
    // P = padding
    // S = strides
    // D = dilations
    //
    //                 N Y X C K K C
    //                       o y x i
    // Activation:
    // (N)  flatExpr = 1 0 0 0 0 0 0 0
    // (H)  flatExpr = 0 S 0 0 D 0 0 P
    // (W)  flatExpr = 0 0 S 0 0 D 0 P
    // (Ci) flatExpr = 0 0 0 0 0 0 1 0
    // Kernel:
    // (H)  flatExpr = 0 0 0 0 1 0 0 0
    // (W)  flatExpr = 0 0 0 0 0 1 0 0
    // (Ci) flatExpr = 0 0 0 0 0 0 1 0
    // (Co) flatExpr = 0 0 0 1 0 0 0 0
    // Output:
    // (N)  flatExpr = 1 0 0 0 0 0 0 0
    // (H)  flatExpr = 0 1 0 0 0 0 0 0
    // (W)  flatExpr = 0 0 1 0 0 0 0 0
    // (Co) flatExpr = 0 0 0 1 0 0 0 0
    for (unsigned i = 0; i < numSpatialDims; ++i) {
      flatExpr actFlat = actFlatExprs[actSpatial[i]];
      flatExpr kerFlat = kerFlatExprs[kerSpatial[i]];
      flatExpr outFlat = outFlatExprs[outSpatial[i]];

      // Activation:
      // (H)  flatExpr = 0 S 0 0 D 0 0 P*
      // (W)  flatExpr = 0 0 S 0 0 D 0 P*
      int64_t padding = actFlat.back();

      // Activation:
      // (H)  flatExpr = 0 S* 0 0 D 0 0 P
      // (W)  flatExpr = 0 0 S* 0 0 D 0 P
      // Output:
      // (H)  flatExpr = 0 1* 0 0 0 0 0 0
      // (W)  flatExpr = 0 0 1* 0 0 0 0 0
      int64_t stride = multiply(actFlat, outFlat);

      // Activation:
      // (H)  flatExpr = 0 S 0 0 D* 0 0 P
      // (W)  flatExpr = 0 0 S 0 0 D* 0 P
      // Kernel:
      // (H)  flatExpr = 0 0 0 0 1* 0 0 0
      // (W)  flatExpr = 0 0 0 0 0 1* 0 0
      int64_t dilation = multiply(actFlat, kerFlat);

      // Expecting non-zero stride, dilation else something is wrong
      // Effectively, this means...
      // 1) Spatial dimensions are listed in order in each tensor
      //    e.g. can't have NHWC activation and NWHC output
      // 2) The activation spatial dimensions are comprised of both
      //    kernel and output spatial dimensions even in the case
      //    of a 1x1 kernel
      if (!stride || !dilation) {
        reason = "Invalid spatial dimensions";

        // clear any previously set paddings, strides, dilations
        paddings.clear();
        strides.clear();
        dilations.clear();

        return false;
      }

      paddings.push_back(padding);
      strides.push_back(stride);
      dilations.push_back(dilation);
    }
    return true;
  }

  int64_t multiply(flatExpr flatExprA, flatExpr flatExprB) {
    size_t size = flatExprA.size();
    assert(flatExprB.size() == size);
    int64_t ret = 0;
    for (size_t i = 0; i < size; ++i) {
      ret += (flatExprA[i] * flatExprB[i]);
    }
    return ret;
  }

  bool findCommonResult(AffineMap mapA, AffineMap mapB, unsigned &indexA,
                        unsigned &indexB) {
    unsigned init = indexB;
    for (; indexA <= mapA.getNumResults(); ++indexA) {
      for (indexB = init; indexB < mapB.getNumResults(); ++indexB) {
        if (mapA.getResult(indexA) == mapB.getResult(indexB)) {
          return true;
        }
      }
    }
    return false;
  }
};

} // namespace pmlc::dialect::tile
