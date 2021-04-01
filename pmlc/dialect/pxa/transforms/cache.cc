// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/cache.h"

#include <memory>
#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/ident.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

static Value createInitLoop(OpBuilder &builder, Location loc, Value memref,
                            Value initVal) {
  auto memrefType = memref.getType().cast<MemRefType>();
  ArrayRef<int64_t> size = memrefType.getShape();
  assert(memrefType.getElementType() == initVal.getType());
  auto loop = builder.create<AffineParallelOp>(loc, memrefType,
                                               AtomicRMWKind::assign, size);
  auto initBuilder = loop.getBodyBuilder();
  auto idMap =
      AffineMap::getMultiDimIdentityMap(size.size(), builder.getContext());
  auto stored = initBuilder.create<PxaReduceOp>(
      loc, AtomicRMWKind::assign, initVal, memref, idMap, loop.getIVs());
  initBuilder.create<AffineYieldOp>(loc, ArrayRef<Value>{stored});
  return loop.getResult(0);
}

static AffineParallelOp createCopyLoop(OpBuilder &builder,               //
                                       Location loc,                     //
                                       ArrayRef<int64_t> size,           //
                                       Value srcMemRef, Value dstMemRef, //
                                       ArrayRef<StrideInfo> srcOffset,   //
                                       ArrayRef<StrideInfo> dstOffset,   //
                                       AtomicRMWKind agg) {
  assert(size.size() == srcOffset.size());
  assert(size.size() == dstOffset.size());
  size_t dims = size.size();
  auto ctx = builder.getContext();
  auto loop = builder.create<AffineParallelOp>(loc, dstMemRef.getType(),
                                               AtomicRMWKind::assign, size);
  SmallVector<StrideInfo, 4> srcAccess;
  SmallVector<StrideInfo, 4> dstAccess;
  for (size_t i = 0; i < dims; i++) {
    auto offset = StrideInfo(loop.getIVs()[i]);
    srcAccess.push_back(srcOffset[i] + offset);
    dstAccess.push_back(dstOffset[i] + offset);
  }
  auto loadMap = convertToValueMap(ctx, srcAccess);
  auto reduceMap = convertToValueMap(ctx, dstAccess);
  auto txBuilder = loop.getBodyBuilder();
  auto loaded = txBuilder.create<PxaLoadOp>(
      loc, srcMemRef, loadMap.getAffineMap(), loadMap.getOperands());
  auto stored = txBuilder.create<PxaReduceOp>(loc, agg, loaded, dstMemRef,
                                              reduceMap.getAffineMap(),
                                              reduceMap.getOperands());
  txBuilder.create<AffineYieldOp>(loc, ArrayRef<Value>{stored});
  return loop;
}

static Value allocateLocalCache(OpBuilder &builder, Value memref, Location loc,
                                const RelativeAccessPattern &rap,
                                bool wholeBlock = false) {
  auto originalType = memref.getType().cast<MemRefType>();
  auto elementType = originalType.getElementType();
  auto memRefType = MemRefType::get(
      wholeBlock ? rap.wholeInnerCount : rap.innerCount, elementType);
  auto alloc = builder.create<AllocOp>(loc, memRefType);
  alloc.getOperation()->setAttr("cache", builder.getUnitAttr());
  return alloc;
}

LogicalResult cacheLoad(AffineParallelOp par, PxaLoadOp load) {
  // Get the striding information for the load op, fail if unsuccessful
  auto maybeRap = computeRelativeAccess(load, par.getBody());
  if (!maybeRap) {
    IVLOG(3, "cacheLoad: Failed due to a non-strided access");
    return failure();
  }
  const auto &rap = *maybeRap;
  // Fail for non-1 strides, TODO: Handle compression case
  auto innerStride = rap.innerStride();
  for (int64_t stride : innerStride) {
    if (stride != 1) {
      IVLOG(3, "cacheLoad failed: not all strides are one");
      return failure();
    }
  }
  // Prep for generation
  auto loc = load.getLoc();
  auto builder = OpBuilder::atBlockBegin(par.getBody());
  // Allocate a temporary buffer
  auto type =
      MemRefType::get(rap.innerCount, load.getMemRefType().getElementType());
  auto localBuf = builder.create<AllocOp>(loc, type);
  // Implement the copy loop
  SmallVector<StrideInfo, 4> zeroOffset(rap.innerCount.size());
  auto copyLoop =
      createCopyLoop(builder, loc, rap.innerCount, load.getMemRef(), localBuf,
                     rap.outer, zeroOffset, AtomicRMWKind::assign);

  // Make a new load and remove the old one
  auto innerMap = convertToValueMap(par.getContext(), rap.inner);
  OpBuilder newLoadBuilder(load);
  auto newLoad = newLoadBuilder.create<PxaLoadOp>(loc, copyLoop.getResult(0),
                                                  innerMap.getAffineMap(),
                                                  innerMap.getOperands());
  load.replaceAllUsesWith(newLoad.result());
  load.erase();

  return success();
}

LogicalResult cacheLoadAsVector(AffineParallelOp par, PxaLoadOp load,
                                int64_t reqVecSize) {
  // Get the striding information for the load op, fail if unsuccessful
  auto maybeRap = computeRelativeAccess(load, par.getBody());
  if (!maybeRap) {
    IVLOG(3, "cacheLoadAsVector: Failed due to a non-strided access");
    return failure();
  }
  const auto &rap = *maybeRap;
  // Require all sizes to be 1 except the final one, which is the vector size
  // and must be stride 1.
  if (rap.innerCount.size() < 1) {
    IVLOG(3, "cacheLoadAsVector: Failed due to 0-dim memref");
    return failure();
  }
  for (unsigned i = 0; i < rap.inner.size() - 1; i++) {
    if (rap.innerCount[i] != 1) {
      IVLOG(3, "cacheLoadAsVector: Failed due to invalid non-final count: "
                   << rap.innerCount[i]);
      return failure();
    }
  }
  unsigned last = rap.inner.size() - 1;
  auto innerStride = rap.innerStride();
  if (innerStride[last] != 1) {
    IVLOG(3, "cacheLoadAsVector: Failed due to invalid stride: "
                 << innerStride[last]);
    return failure();
  }
  int64_t vectorSize = rap.innerCount[last];
  if (vectorSize == 1) {
    IVLOG(3, "cacheLoadAsVector: Failed due to size 1 vector width");
    return failure();
  }
  if (reqVecSize && vectorSize != reqVecSize) {
    IVLOG(3, "cacheLoadAsVector: Failed due to mismatch of required size");
    return failure();
  }
  auto eltType = load.getMemRefType().getElementType();
  auto vecType = VectorType::get({vectorSize}, eltType);
  // Prep for generation
  auto loc = load.getLoc();
  auto builder = OpBuilder::atBlockBegin(par.getBody());
  // Load as a vector
  auto loadMap = convertToValueMap(load.getContext(), rap.outer);
  IVLOG(2, "Making vector load");
  auto loadVec = builder.create<PxaVectorLoadOp>(loc, vecType, load.getMemRef(),
                                                 loadMap.getAffineMap(),
                                                 loadMap.getOperands());
  // Make a new load and remove the old one
  OpBuilder newLoadBuilder(load);
  // Do an affine apply to get the index
  auto innerMap = convertToValueMap(par.getContext(), rap.inner);
  // Extract the right element of the vector
  Value idx = newLoadBuilder.create<AffineApplyOp>(
      loc, innerMap.getAffineMap().getSubMap({last}), innerMap.getOperands());

  if (idx.getType().isa<IndexType>()) {
    auto indexCast = newLoadBuilder.create<IndexCastOp>(load.getLoc(), idx,
                                                        builder.getI32Type());
    idx = indexCast.getResult();
  }
  auto newLoad = newLoadBuilder.create<vector::ExtractElementOp>(
      loc, eltType, loadVec.getResult(), idx);
  load.replaceAllUsesWith(newLoad.result());
  load.erase();
  return success();
}

LogicalResult cacheReduce(AffineParallelOp par, PxaReduceOp reduce) {
  // Get the striding information for the load op, fail if unsuccessful
  auto maybeRap = computeRelativeAccess(reduce, par.getBody());
  if (!maybeRap) {
    IVLOG(3, "cacheReduce failed: due to a non-strided access");
    return failure();
  }
  const auto &rap = *maybeRap;

  // Fail for non-1 strides, TODO: Handle compression case
  auto innerStride = rap.innerStride();
  for (int64_t stride : innerStride) {
    if (stride != 1) {
      IVLOG(3, "cacheReduce failed: not all strides are one.");
      return failure();
    }
  }

  // Compute the first use of the buffer written to by the reduce and verify
  // that there are only yields between
  Value out = reduce;
  while (out.getParentBlock() != par.getBody()) {
    if (!out.hasOneUse()) {
      IVLOG(3, "cacheReduce failed: multiple uses");
      return failure();
    }
    auto &use = *out.use_begin();
    auto yieldOp = dyn_cast<AffineYieldOp>(use.getOwner());
    if (!yieldOp) {
      IVLOG(3, "cacheReduce failed: missing yield op");
      return failure();
    }
    out = yieldOp->getParentOp()->getResult(use.getOperandNumber());
  }

  // Verify final output has a single use, TODO: THis probably isn't a hard
  // requirement, but it's easier
  if (!out.hasOneUse()) {
    IVLOG(3, "cacheReduce failed: multiple uses");
    return failure();
  }
  auto &finalUse = *out.use_begin();

  // Prep for generation
  auto loc = reduce.getLoc();
  auto builder = OpBuilder::atBlockBegin(par.getBody());

  // Allocate a temporary buffer
  auto eltType = reduce.getMemRefType().getElementType();
  auto type = MemRefType::get(rap.innerCount, eltType);
  auto localBuf = builder.create<AllocOp>(loc, type);

  // If it's not an assign, clear it to the reduction identity
  Value initBuf = localBuf;
  if (reduce.agg() != AtomicRMWKind::assign) {
    auto ident = createIdentity(builder, loc, reduce.agg(), eltType);
    initBuf = createInitLoop(builder, loc, localBuf, ident);
  }

  // Make a new load and remove the old one
  auto innerMap = convertToValueMap(par.getContext(), rap.inner);
  OpBuilder newReduceBuilder(reduce);
  auto newReduce = newReduceBuilder.create<PxaReduceOp>(
      loc, reduce.agg(), reduce.val(), initBuf, innerMap.getAffineMap(),
      innerMap.getOperands());
  reduce.replaceAllUsesWith(newReduce.result());

  // Walk upwards a second time and fix typing
  out = newReduce;
  Type newType = out.getType();
  while (out.getParentBlock() != par.getBody()) {
    auto &use = *out.use_begin();
    auto yieldOp = cast<AffineYieldOp>(use.getOwner());
    out = yieldOp->getParentOp()->getResult(use.getOperandNumber());
    out.setType(newType);
  }

  // Implement the copy loop
  builder.setInsertionPoint(par.getBody(), std::prev(par.getBody()->end()));
  SmallVector<StrideInfo, 4> zeroOffset(rap.innerCount.size());
  auto copyLoop =
      createCopyLoop(builder, loc, rap.innerCount, out, reduce.getMemRef(),
                     zeroOffset, rap.outer, reduce.agg());
  finalUse.set(copyLoop.getResult(0));
  reduce.erase();

  return success();
}

static bool isInitialized(Value memref) {
  if (auto op = memref.getDefiningOp()) {
    if (isa<AllocOp>(op)) {
      return false;
    }
    return true;
  }
  auto arg = memref.cast<BlockArgument>();
  auto *parentOp = arg.getOwner()->getParentOp();
  if (auto funcOp = dyn_cast<FuncOp>(parentOp)) {
    auto numInputs = funcOp.getNumArguments() - funcOp.getNumResults();
    return arg.getArgNumber() < numInputs;
  }
  return true;
}

void CachePlan::addLoad(PxaLoadOp op) {
  auto memref = getIndirectDefOutsideScope(op.memref(), outerBand);
  if (!memref) {
    return;
  }

  auto maybeRap = computeRelativeAccess(op, middleBand.getBody());
  if (!maybeRap) {
    return;
  }

  Entry entry{*maybeRap};
  auto [it, isNew] = entries.insert(std::make_pair(memref, entry));
  if (!isNew) {
    // TODO: check LogicalResult
    (void)it->second.rap.unionMerge(*maybeRap);
  }
  it->second.loads.push_back(LoadInfo{op, *maybeRap});
}

void CachePlan::addReduce(PxaReduceOp op) {
  auto memref = getIndirectDefOutsideScope(op.memref(), outerBand);
  if (!memref) {
    return;
  }

  auto maybeRap = computeRelativeAccess(op, middleBand.getBody());
  if (!maybeRap) {
    return;
  }

  Entry entry{*maybeRap};
  auto [it, isNew] = entries.insert(std::make_pair(memref, entry));
  if (!isNew) {
    // TODO: check LogicalResult
    (void)it->second.rap.unionMerge(*maybeRap);
  }
  it->second.reduces.push_back(ReduceInfo{op, *maybeRap});
}

static bool hasIndices(ArrayRef<StrideInfo> strideInfos,
                       ArrayRef<BlockArgument> idxs) {
  return llvm::any_of(idxs, [&](BlockArgument idx) {
    return llvm::any_of(strideInfos, [&](const StrideInfo &si) {
      return si.strides.count(idx);
    });
  });
}

static AffineParallelOp getRelevantBand(const RelativeAccessPattern &rap,
                                        AffineParallelOp outerBand,
                                        AffineParallelOp middleBand) {
  return hasIndices(rap.outer, middleBand.getIVs()) ? middleBand : outerBand;
}

static void replaceLoad(PxaLoadOp load, Value source,
                        const RelativeAccessPattern &rap) {
  // Make a new load and remove the old one
  OpBuilder builder(load);
  auto innerMap = convertToValueMap(load.getContext(), rap.inner);
  auto newLoad = builder.create<PxaLoadOp>(
      load.getLoc(), source, innerMap.getAffineMap(), innerMap.getOperands());
  load.replaceAllUsesWith(newLoad.result());
  load.erase();
}

static Value replaceReduce(AffineParallelOp band, PxaReduceOp reduce,
                           Value cache, const RelativeAccessPattern &rap) {
  // Make a new reduce and remove the old one
  auto loc = reduce.getLoc();
  auto innerMap = convertToValueMap(band.getContext(), rap.inner);
  OpBuilder builder(reduce);
  auto newReduce = builder.create<PxaReduceOp>(loc, reduce.agg(), reduce.val(),
                                               cache, innerMap.getAffineMap(),
                                               innerMap.getOperands());
  reduce.replaceAllUsesWith(newReduce.result());
  reduce.erase();

  // Walk upwards to adjust types
  Value result = newReduce;
  Type newType = newReduce.getType();
  while (result.getParentBlock() != band.getBody()) {
    auto use = result.use_begin();
    result = getNextIndirectUse(*use);
    // if (!result) {
    //   band.emitOpError("No indirect uses found");
    // }
    if (auto ifOp = dyn_cast<AffineIfOp>(result.getDefiningOp())) {
      // TODO: should we fail in this case?
      auto yield =
          dyn_cast<AffineYieldOp>(ifOp.getElseBlock()->getTerminator());
      yield.setOperand(0, cache);
    }
    result.setType(newType);
  }
  return result;
}

void CachePlan::execute() {
  for (auto &[memref, entry] : entries) {
    // determine the level to cache at
    entry.band = getRelevantBand(entry.rap, outerBand, middleBand);
    auto loc = entry.band.getLoc();
    auto builder = OpBuilder::atBlockBegin(entry.band.getBody());
    SmallVector<StrideInfo, 4> zeroOffset(entry.rap.innerCount.size());

    entry.cache =
        allocateLocalCache(builder, memref, loc, entry.rap, wholeBlock);
    if (isInitialized(memref)) {
      // copy global -> local
      entry.copyInto = true;
      auto copyLoop = createCopyLoop(
          builder, loc,
          wholeBlock ? entry.rap.wholeInnerCount : entry.rap.innerCount, memref,
          entry.cache, entry.rap.outer, zeroOffset, AtomicRMWKind::assign);
      copyLoop.getOperation()->setAttr("cache_in", builder.getUnitAttr());
      entry.cache = copyLoop.getResult(0);
    }

    for (const auto &load : entry.loads) {
      replaceLoad(load.op, entry.cache, load.rap);
    }

    if (entry.reduces.size()) {
      Value finalValue;
      for (const auto &reduce : entry.reduces) {
        finalValue =
            replaceReduce(entry.band, reduce.op, entry.cache, reduce.rap);
      }

      // copy local -> global
      entry.copyFrom = true;
      OpBuilder::InsertionGuard guard(builder);
      auto yield = entry.band.getBody()->getTerminator();
      builder.setInsertionPoint(yield);
      auto &finalUse = *finalValue.use_begin();
      auto copyLoop =
          createCopyLoop(builder, loc, entry.rap.innerCount, finalValue, memref,
                         zeroOffset, entry.rap.outer, AtomicRMWKind::assign);
      copyLoop.getOperation()->setAttr("cache_out", builder.getUnitAttr());
      finalUse.set(copyLoop.getResult(0));
    }
  }
}

struct CachePass : public CacheBase<CachePass> {
  explicit CachePass(bool wholeBlock) { this->wholeBlock = wholeBlock; }

  void runOnFunction() final {
    auto func = getFunction();

    func.walk([&](AffineParallelOp inner) {
      if (!util::hasTag(inner, innerTag)) {
        return;
      }

      auto middle = dyn_cast<AffineParallelOp>(inner->getParentOp());
      if (!middle || !util::hasTag(middle, middleTag)) {
        middle.emitError("Middle loop does not have tag");
        signalPassFailure();
        return;
      }

      auto outer = dyn_cast<AffineParallelOp>(middle->getParentOp());
      if (!outer || !util::hasTag(outer, outerTag)) {
        outer.emitError("Outer loop does not have tag");
        signalPassFailure();
        return;
      }

      CachePlan plan(outer, middle, wholeBlock);
      inner.walk([&](PxaLoadOp load) { plan.addLoad(load); });
      inner.walk([&](PxaReduceOp reduce) { plan.addReduce(reduce); });
      plan.execute();
    });
  }
};

std::unique_ptr<mlir::Pass> createCachePass(bool wholeBlock) {
  return std::make_unique<CachePass>(wholeBlock);
}

} // namespace pmlc::dialect::pxa
