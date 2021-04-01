// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/cache.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/tile.h"
#include "pmlc/dialect/pxa/transforms/tile_accumulate.h"
#include "pmlc/dialect/pxa/transforms/vectorize.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

using llvm::SmallVector;

struct SubgroupPlan {
  int64_t subgroupSize;
  SmallVector<int64_t, 6> innerTile;
  SmallVector<int64_t, 6> subgroupTile;
};

std::ostream &operator<<(std::ostream &os, const SubgroupPlan &plan) {
  size_t sz = plan.innerTile.size();
  os << plan.subgroupSize << ":[";
  for (size_t i = 0; i < sz; i++) {
    os << plan.innerTile[i];
    if (i != sz - 1) {
      os << " ";
    }
  }
  os << "]:[";
  for (size_t i = 0; i < sz; i++) {
    os << plan.subgroupTile[i];
    if (i != sz - 1) {
      os << " ";
    }
  }
  os << "]";
  return os;
}

struct SubgroupParams {
  SmallVector<int64_t, 4> subgroupSizes;
  int64_t maxRegsPerThread;
  double cacheWidth;
  double cacheLatency;
  double memoryLatency;
};

struct SubgroupCostModel {
  SubgroupCostModel(const SubgroupParams &params, AffineParallelOp op)
      : params(params), op(op) {
    bestCost = std::numeric_limits<double>::infinity();
    // Tile accumulations only works on parallel loops with a single result
    if (op.getNumResults() != 1) {
      return;
    }
    // Skip low dimensional cases.  It's not profitable to subgroup such cases.
    // However, additionally running the subgroup transform anyway should be
    // safe, but actually causes backend test resnetDense to fail
    // TODO: Debug correctness failure of resnetDense when this code is removed.
    if (op.getIVs().size() < 3) {
      return;
    }
    // Verify we have constant ranges and cache
    auto maybeRanges = op.getConstantRanges();
    if (!maybeRanges) {
      return;
    }
    ranges = *maybeRanges;
    IVLOG(3, "Ranges = " << *maybeRanges);
    // Preflight all loads/stores + cache
    bool safe = true;
    op.walk([&](PxaReduceOp red) {
      if (red.agg() != AtomicRMWKind::addf) {
        // This isn't really unsafe, but basically this test removes
        // non-contraction like ops from consideration.  Eltwise ops are not
        // good to subgroup due to low computational density (we should
        // explicity check for that), and pooling fail due to vectorization
        // issues with CMP (we should fix that).  However, this works for now.
        safe = false;
        return;
      }
      if (!preflightIO(red)) {
        safe = false;
      }
    });
    op.walk([&](PxaLoadOp load) {
      if (!preflightIO(load)) {
        safe = false;
      }
    });
    if (!safe) {
      return;
    }
    plan.innerTile.resize(ranges.size());
    plan.subgroupTile.resize(ranges.size());
    for (int64_t subgroupSize : params.subgroupSizes) {
      plan.subgroupSize = subgroupSize;
      computeCostRecursive(0);
    }
  }

  template <typename OpType>
  bool preflightIO(OpType ioOp) {
    IVLOG(3, "Preflight: " << debugString(*ioOp.getOperation()));
    if (ioOp.getOperation()->getBlock() != op.getBody()) {
      IVLOG(3, "Not part of block");
      return false;
    }

    auto maybeDimStrides =
        computeStrideInfo(ioOp.getAffineMap(), ioOp.getMapOperands());
    if (!maybeDimStrides) {
      IVLOG(3, "Cannot compute dimensionalized strides");
      return false;
    }
    ioDimStrides.push_back(*maybeDimStrides);
    IVLOG(3, "  dimensional strides = " << *maybeDimStrides)

    auto memreftype = ioOp.getMemRefType();

    int64_t offset;
    SmallVector<int64_t, 4> tensorStrides;
    if (failed(getStridesAndOffset(memreftype, tensorStrides, offset))) {
      IVLOG(3, "Cannot compute tensor strides");
      return false;
    }
    ioTensorStrides.push_back(tensorStrides);
    IVLOG(3, "  dimensional strides = " << tensorStrides);

    auto elementtype = memreftype.getElementType();
    assert(elementtype.isIntOrFloat());
    ioElementSizesInBytes.push_back(elementtype.getIntOrFloatBitWidth() / 8);
    return true;
  }

  void computeCostRecursive(unsigned idx) {
    if (idx == ranges.size()) {
      auto cost = computeCost();
      if (cost < bestCost) {
        bestCost = cost;
        bestPlan = plan;
      }
      return;
    }
    // Try using a subgroup or not for each index
    for (unsigned doSubgroup = 0; doSubgroup < 2; doSubgroup++) {
      // Get the range of this index
      int64_t range = ranges[idx];
      if (range % plan.subgroupSize != 0 && doSubgroup == 1) {
        continue; // Skip the subgroup if not even divison by subgroup size
      }
      if (doSubgroup) {
        // If we are doing subgrouping, set + reduce range
        plan.subgroupTile[idx] = plan.subgroupSize;
        range /= plan.subgroupSize;
      } else {
        plan.subgroupTile[idx] = 1;
      }
      // Now try all even divisors for remaining size
      for (int64_t ts = 1; ts <= range; ts++) {
        if (range % ts != 0) {
          continue;
        }
        plan.innerTile[idx] = ts * plan.subgroupTile[idx];
        computeCostRecursive(idx + 1);
      }
    }
    return;
  }

  struct MemInfo {
    unsigned subgroupCount;
    int64_t registers;
    int64_t accesses;
    double cacheMiss;
    SmallVector<int64_t, 4> tensorDimensions;
  };

  MemInfo computeMemoryInfo(SmallVector<StrideInfo, 4> dimStrides,
                            SmallVector<int64_t, 4> tensorStrides,
                            unsigned elementSizeInBytes) {
    MemInfo out = {0, 1, 1, 0.0};
    assert(dimStrides.size() == tensorStrides.size());
    for (size_t i = 0; i < dimStrides.size(); i++) {
      int64_t pos = 0;
      int64_t neg = 0;

      for (auto kvp : dimStrides[i].strides) {
        auto index = kvp.first.getArgNumber();
        auto dimStride = kvp.second;

        if (plan.subgroupTile[index] == plan.subgroupSize &&
            tensorStrides[i] == 1) {
          out.subgroupCount++;
        }

        if (dimStride > 0) {
          pos += dimStride * (plan.innerTile[index] - 1);
        } else {
          neg += dimStride * (plan.innerTile[index] - 1);
        }

        out.accesses *= plan.innerTile[index];
      }

      auto dimSize = pos - neg + 1;
      out.registers *= dimSize;
      out.tensorDimensions.push_back(dimSize);
    }

    out.cacheMiss = computeCacheMiss(params.cacheWidth / elementSizeInBytes,
                                     out.tensorDimensions, tensorStrides);

    return out;
  }

  double computeCost() {
    int64_t totRegisters = 0;
    int64_t totAccesses = 0;
    double totCacheMiss = 0.0;

    for (size_t i = 0; i < ioDimStrides.size(); i++) {
      auto mi = computeMemoryInfo(ioDimStrides[i], ioTensorStrides[i],
                                  ioElementSizesInBytes[i]);
      if (mi.subgroupCount > 1) {
        return std::numeric_limits<double>::infinity();
      }
      if (i == 0 && mi.subgroupCount == 0) {
        return std::numeric_limits<double>::infinity();
      }

      if (mi.subgroupCount) {
        mi.registers /= plan.subgroupSize;
        mi.accesses /= plan.subgroupSize;
        mi.cacheMiss /= plan.subgroupSize;
      }

      totRegisters += mi.registers;
      totAccesses += mi.accesses;
      totCacheMiss += mi.cacheMiss;
    }

    if (totRegisters > params.maxRegsPerThread) {
      IVLOG(3, "Invalid subgroup plan: " << plan << ", totRegisters = "
                                         << totRegisters);
      return std::numeric_limits<double>::infinity();
    }
    IVLOG(3, "Valid subgroup plan: " << plan
                                     << ", totRegisters = " << totRegisters);

    int64_t totOps = 1;
    for (auto it : plan.innerTile) {
      totOps *= it;
    }

    double totMemIO = (totAccesses - totCacheMiss) * params.cacheLatency +
                      totCacheMiss * params.memoryLatency;

    return totMemIO / totOps;
  }

  // The parameters to the cost model
  SubgroupParams params;
  // The operation being optimized
  AffineParallelOp op;
  // Current plan
  SubgroupPlan plan;
  // Best cost found so far
  double bestCost;
  // Best plan found for far
  SubgroupPlan bestPlan;
  // Cache of the index ranges
  SmallVector<int64_t, 8> ranges;
  // Strides for all io ops
  SmallVector<SmallVector<StrideInfo, 4>, 4> ioDimStrides;
  SmallVector<SmallVector<int64_t, 4>, 4> ioTensorStrides;
  SmallVector<unsigned, 4> ioElementSizesInBytes;
};

void SubgroupApply(AffineParallelOp op, SubgroupPlan plan) {
  // Perform the primary innermost tiling
  auto inner = performTiling(op, plan.innerTile);
  // Perform the deep inner tiling
  auto subgroup = performTiling(inner, plan.subgroupTile);
  // Tile over accumulations
  auto accum = tileAccumulations(op, false);
  // Cache innermost loads at accum level
  subgroup.walk([&](PxaLoadOp load) {
    // TODO: check LogicalResult
    (void)cacheLoad(accum, load);
  });
  // Cache innermost reduces at op level
  subgroup.walk([&](PxaReduceOp reduce) {
    // TODO: check LogicalResult
    (void)cacheReduce(op, reduce);
  });
  // Vectorize everything we can
  op.walk([&](AffineParallelOp par) {
    // TODO: check LogicalResult
    (void)vectorizeOverOutputs(par, plan.subgroupSize);
  });
  // Try to 'vector cache' any remaining innermost loads
  subgroup.walk([&](PxaLoadOp load) {
    // TODO: check LogicalResult
    (void)cacheLoadAsVector(inner, load, plan.subgroupSize);
  });
  // Convert local allocations to vector types
  op.walk([&](AllocOp alloc) {
    // TODO: check LogicalResult
    (void)vectorizeBuffer(alloc);
  });
  // Attach subgroup size
  setIntegerTag(op, subgroupSizeTag(), plan.subgroupSize);
}

struct SubgroupsPass : public SubgroupsBase<SubgroupsPass> {
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](AffineParallelOp op) { doSubgroups(op); });
  }

  void doSubgroups(AffineParallelOp op) {
    SubgroupParams params = {
        {8, 16}, // Subgroup sizes to consider
        40,      // Maximum register per thread
        64.0,    // Cache width
        125.0,   // Cache latency
        420.0,   // Memory latency
    };
    SubgroupCostModel cm(params, op);
    if (cm.bestCost == std::numeric_limits<double>::infinity()) {
      // If subgrouping fails, we tile accumulations instead to handle the other
      // cases
      tileAccumulations(op, false);
      setIntegerTag(op, subgroupSizeTag(), 1);
      return;
    }
    IVLOG(3, "best plan = " << cm.bestPlan);
    SubgroupApply(op, cm.bestPlan);
  }
};

} // namespace

std::unique_ptr<Pass> createSubgroupsPass() {
  return std::make_unique<SubgroupsPass>();
}

} // namespace pmlc::dialect::pxa
