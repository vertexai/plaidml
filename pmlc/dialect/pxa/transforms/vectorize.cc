// Copyright 2020 Intel Corporation

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Interfaces/VectorInterfaces.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/vectorize.h"
#include "pmlc/util/logging.h"

#include "mlir/Support/DebugStringHelper.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {
using pmlc::dialect::pxa::PxaReduceOp;

static std::string getValueName(Operation *op, Value value) {
  AsmState state(op->getParentOfType<FuncOp>());
  std::string str;
  llvm::raw_string_ostream os(str);
  value.printAsOperand(os, state);
  return os.str();
}

class VectorizeCandidate {
private:
  AffineParallelOp loop;
  BlockArgument index;
  unsigned vectorWidth;
  bool vectorizeMathOp;
  DenseSet<Value> vectorizedValues;
  DenseSet<Operation *> vectorizedOps;
  DenseSet<Operation *> zeroStrideReductions;

  const char *stringifyAtomicRMWKindForVectorReductionOp(AtomicRMWKind val) {
    switch (val) {
    case AtomicRMWKind::addf:
      return "add";
    case AtomicRMWKind::addi:
      return "add";
    case AtomicRMWKind::assign:
      return "invalid";
    case AtomicRMWKind::maxf:
      return "max";
    case AtomicRMWKind::maxs:
      return "max";
    case AtomicRMWKind::maxu:
      return "max";
    case AtomicRMWKind::minf:
      return "min";
    case AtomicRMWKind::mins:
      return "min";
    case AtomicRMWKind::minu:
      return "min";
    case AtomicRMWKind::mulf:
      return "mul";
    case AtomicRMWKind::muli:
      return "mul";
    }
    llvm_unreachable("Invalid aggregation type");
  }

  LogicalResult tryVectorizePxaLoadOp(PxaLoadOp op) {
    auto strideInfo = computeStrideInfo(op);
    if (!strideInfo) {
      return op.emitRemark("Vectorize op: Failed, non-affine strides");
    }

    auto it = strideInfo->strides.find(index);
    if (it == strideInfo->strides.end()) {
      // Stride 0, safe to leave unvectorized
      return success();
    }

    // Stride is non-zero, must vectorize
    if (it->second != 1) {
      return op.emitRemark("Vectorize op: Failed, stride != 1");
    }
    vectorizedOps.insert(op);
    vectorizedValues.insert(op.getResult());
    return success();
  }

  LogicalResult tryVectorizePxaReduceOp(PxaReduceOp op) {
    auto strideInfo = computeStrideInfo(op);
    if (!strideInfo) {
      return op.emitRemark("Vectorize op: Failed, non-affine strides");
    }

    auto it = strideInfo->strides.find(index);
    if (it == strideInfo->strides.end()) {
      // vector::ReductionOp doesn't support pxa's assign reduction.
      // Also, make sure we handle only the supported types -
      // see the vector::ReductionOp verification code
      // (https://github.com/llvm/llvm-project/blob/master/mlir/lib/Dialect/Vector/VectorOps.cpp#L134).
      Type elementType = op.getMemRefType().getElementType();
      if (op.agg() == AtomicRMWKind::assign ||
          (!elementType.isF32() && !elementType.isF64() &&
           !elementType.isSignlessInteger(32) &&
           !elementType.isSignlessInteger(64))) {
        return op.emitRemark(
            "Vectorize op: Failed, unsupported reduction or type "
            "for vector::ReductionOp");
      }
      // If stride is 0, "remember it" as such.
      zeroStrideReductions.insert(op.getOperation());
    } else if (it->second != 1) {
      return op.emitRemark("Vectorize op: Failed, stride != 1");
    }

    vectorizedOps.insert(op);
    return success();
  }

  template <typename OpTy>
  void createLoopForMathUnaryOp(OpTy op) {
    if (op->getNumOperands() != 1) {
      op->emitRemark("Vectorize op: Failed, op has more than 1 operand");
      return;
    }
    OpBuilder builder(op);
    auto loc = op.getLoc();

    // For each non-vector operand, broadcast as needed
    mlir::Value operand = op.getOperand();
    if (!operand.getType().isa<VectorType>()) {
      auto vectorType = VectorType::get({vectorWidth}, operand.getType());
      auto broadcast =
          builder.create<vector::BroadcastOp>(loc, vectorType, operand);
      operand.replaceAllUsesWith(broadcast);
    }
    // Update the result type
    auto result = op.getResult();
    auto vectorType = VectorType::get({vectorWidth}, result.getType());
    result.setType(vectorType);

    auto memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType());
    auto vecMem = builder.create<memref::AllocOp>(loc, memrefType);
    SmallVector<Value, 8> vectorIvs{builder.create<ConstantIndexOp>(loc, 0)};
    builder.create<vector::TransferWriteOp>(loc, operand, vecMem, vectorIvs);

    auto parallelOp = builder.create<AffineParallelOp>(
        loc, ArrayRef<Type>{memrefType},
        ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        AffineMap::getConstantMap(0, op.getContext()), ValueRange(),
        AffineMap::getConstantMap(vectorWidth, op.getContext()), ValueRange(),
        ArrayRef<int64_t>{1});
    auto parallelBody = parallelOp.getBody();
    builder.setInsertionPointToStart(parallelBody);

    auto elementIvs = parallelBody->getArguments();
    auto element = builder.create<AffineLoadOp>(loc, vecMem, elementIvs);
    auto mathOpResult = builder.create<OpTy>(loc, element).getResult();
    auto idMap = builder.getMultiDimIdentityMap(memrefType.getRank());
    auto reduceOp = builder.create<pxa::PxaReduceOp>(
        loc, AtomicRMWKind::assign, mathOpResult, vecMem, idMap, elementIvs);
    builder.create<AffineYieldOp>(loc, reduceOp.getResult());

    builder.setInsertionPointAfter(parallelOp);
    auto vectorOp = builder.create<vector::TransferReadOp>(
        loc, vectorType, parallelOp.getResult(0), vectorIvs);
    op.replaceAllUsesWith(vectorOp.getOperation());
    op.erase();
  }

  template <typename OpTy>
  void createLoopForMathBinaryOp(OpTy op) {
    if (op->getNumOperands() != 2) {
      op->emitRemark("Vectorize op: Failed, op does not have 2 operands");
      return;
    }
    OpBuilder builder(op);
    auto loc = op.getLoc();

    // For each non-vector operand, broadcast as needed
    for (mlir::Value operand : op.getOperands()) {
      if (!operand.getType().isa<VectorType>()) {
        auto vectorType = VectorType::get({vectorWidth}, operand.getType());
        auto broadcast =
            builder.create<vector::BroadcastOp>(loc, vectorType, operand);
        operand.replaceAllUsesWith(broadcast);
      }
    }
    // Update the result type
    auto result = op.getResult();
    auto vectorType = VectorType::get({vectorWidth}, result.getType());
    result.setType(vectorType);

    auto memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType());
    SmallVector<Value, 8> vectorIvs{builder.create<ConstantIndexOp>(loc, 0)};

    mlir::Value operand1 = op.getOperand(0);
    auto vecMem1 = builder.create<memref::AllocOp>(loc, memrefType);
    builder.create<vector::TransferWriteOp>(loc, operand1, vecMem1, vectorIvs);
    mlir::Value operand2 = op.getOperand(1);
    auto vecMem2 = builder.create<memref::AllocOp>(loc, memrefType);
    builder.create<vector::TransferWriteOp>(loc, operand2, vecMem2, vectorIvs);

    auto parallelOp = builder.create<AffineParallelOp>(
        loc, ArrayRef<Type>{memrefType},
        ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        AffineMap::getConstantMap(0, op.getContext()), ValueRange(),
        AffineMap::getConstantMap(vectorWidth, op.getContext()), ValueRange(),
        ArrayRef<int64_t>{1});
    auto parallelBody = parallelOp.getBody();
    builder.setInsertionPointToStart(parallelBody);

    auto elementIvs = parallelBody->getArguments();
    auto element1 = builder.create<AffineLoadOp>(loc, vecMem1, elementIvs);
    auto element2 = builder.create<AffineLoadOp>(loc, vecMem2, elementIvs);
    auto mathOpResult =
        builder.create<OpTy>(loc, element1, element2).getResult();
    auto idMap = builder.getMultiDimIdentityMap(memrefType.getRank());
    auto reduceOp = builder.create<pxa::PxaReduceOp>(
        loc, AtomicRMWKind::assign, mathOpResult, vecMem1, idMap, elementIvs);
    builder.create<AffineYieldOp>(loc, reduceOp.getResult());

    builder.setInsertionPointAfter(parallelOp);
    auto vectorOp = builder.create<vector::TransferReadOp>(
        loc, vectorType, parallelOp.getResult(0), vectorIvs);
    op.replaceAllUsesWith(vectorOp.getOperation());
    op.erase();
  }

  LogicalResult tryVectorizeScalarOp(Operation *op) {
    if (op->getNumRegions() != 0) {
      return op->emitRemark("Vectorize op: Failed, interior loops");
    }
    // TODO: consider more generic way to add ops supported here
    if (!isa<FPExtOp, FPTruncOp, IndexCastOp, VectorUnrollOpInterface, SelectOp,
             CmpFOp>(op)) {
      // Probably not a vectorizable op. Verify it doesn't use an
      // vectorized results.
      for (auto operand : op->getOperands()) {
        if (vectorizedValues.count(operand)) {
          return op->emitRemark(
              "Vectorize op: Failed, unknown op used vectorized result");
        }
      }
      // Otherwise, safe and ignorable.
      return success();
    }
    // Only vectorize if at least one operand is vectorized
    bool anyVec = llvm::any_of(op->getOperands(), [&](Value operand) {
      return vectorizedValues.count(operand);
    });
    if (!anyVec) {
      // No need to vectorize, all is good
      return success();
    }
    // We also don't handle ops with multiple results
    if (op->getNumResults() != 1) {
      return op->emitRemark("Vectorize op: Failed, multi-result scalar op");
    }
    vectorizedOps.insert(op);
    vectorizedValues.insert(op->getResult(0));
    return success();
  }

  LogicalResult tryVectorizeOperation(Operation *op) {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<PxaLoadOp>([&](auto op) { return tryVectorizePxaLoadOp(op); })
        .Case<PxaReduceOp>([&](auto op) { return tryVectorizePxaReduceOp(op); })
        .Default([&](Operation *op) { return tryVectorizeScalarOp(op); });
  }

public:
  VectorizeCandidate(AffineParallelOp loop, BlockArgument index,
                     unsigned vectorWidth, bool vectorizeMathOp)
      : loop(loop), index(index), vectorWidth(vectorWidth),
        vectorizeMathOp(vectorizeMathOp) {
    IVLOG(3, "Vectorize candidate: " << getValueName(loop, index));
  }

  void vectorizeScalarOp(Operation *op) {
    OpBuilder builder(op);
    for (auto &operand : op->getOpOperands()) {
      // For each non-vector operand, broadcast as needed
      if (!operand.get().getType().isa<VectorType>()) {
        auto vectorType =
            VectorType::get({vectorWidth}, operand.get().getType());
        auto broadcast = builder.create<vector::BroadcastOp>(
            op->getLoc(), vectorType, operand.get());
        operand.set(broadcast);
      }
    }
    // Update the result type
    auto result = op->getResult(0);
    auto vectorType = VectorType::get({vectorWidth}, result.getType());
    result.setType(vectorType);
  }

  void vectorizeLoadOp(PxaLoadOp op) {
    Value operand = op.getMemRef();
    auto elementType = op.getMemRefType().getElementType();
    auto vectorType = VectorType::get(vectorWidth, elementType);
    OpBuilder builder(op);
    auto vecOp =
        builder.create<PxaVectorLoadOp>(op.getLoc(), vectorType, operand,
                                        op.getAffineMap(), op.getMapOperands());
    op.replaceAllUsesWith(vecOp.getResult());
    op.erase();
  }

  void vectorizeReduceOp(PxaReduceOp op) {
    Value val = op.val();
    OpBuilder builder(op);
    if (!val.getType().isa<VectorType>()) {
      auto vectorType = VectorType::get({vectorWidth}, val.getType());
      auto broadcast =
          builder.create<vector::BroadcastOp>(op.getLoc(), vectorType, val);
      val = broadcast.getResult();
    }
    if (zeroStrideReductions.count(op.getOperation())) {
      // Add vector_reduction only if the stride is 0
      auto reductionOp = builder.create<vector::ReductionOp>(
          op.getLoc(), op.getMemRefType().getElementType(),
          builder.getStringAttr(
              stringifyAtomicRMWKindForVectorReductionOp(op.agg())),
          val, ValueRange{});
      auto reduceOp = builder.create<PxaReduceOp>(
          op.getLoc(), ArrayRef<Type>{op.getMemRefType()}, op.agg(),
          reductionOp.getResult(), op.memref(), op.map(), op.idxs());
      op.replaceAllUsesWith(reduceOp.getResult());
      op.erase();
    } else {
      auto vectorReduceOp = builder.create<PxaVectorReduceOp>(
          op.getLoc(), ArrayRef<Type>{op.getMemRefType()}, op.agg(), val,
          op.memref(), op.map(), op.idxs());
      op.replaceAllUsesWith(vectorReduceOp.getResult());
      op.erase();
    }
  }

  void vectorizeOperation(Operation *op) {
    if (!vectorizedOps.count(op)) {
      return;
    }
    if (vectorizeMathOp) {
      TypeSwitch<Operation *>(op)
          .Case<PxaLoadOp>([&](auto op) { vectorizeLoadOp(op); })
          .Case<PxaReduceOp>([&](auto op) { vectorizeReduceOp(op); })
          .Default([&](Operation *op) { vectorizeScalarOp(op); });
    } else {
      TypeSwitch<Operation *>(op)
          .Case<PxaLoadOp>([&](auto op) { vectorizeLoadOp(op); })
          .Case<PxaReduceOp>([&](auto op) { vectorizeReduceOp(op); })
          .Case<math::CosOp>(
              [&](auto op) { createLoopForMathUnaryOp<math::CosOp>(op); })
          .Case<math::ExpOp>(
              [&](auto op) { createLoopForMathUnaryOp<math::ExpOp>(op); })
          .Case<math::LogOp>(
              [&](auto op) { createLoopForMathUnaryOp<math::LogOp>(op); })
          .Case<math::SinOp>(
              [&](auto op) { createLoopForMathUnaryOp<math::SinOp>(op); })
          .Case<math::SqrtOp>(
              [&](auto op) { createLoopForMathUnaryOp<math::SqrtOp>(op); })
          .Case<math::TanhOp>(
              [&](auto op) { createLoopForMathUnaryOp<math::TanhOp>(op); })
          .Default([&](Operation *op) { vectorizeScalarOp(op); });
    }
  }

  LogicalResult isLegal() {
    auto ranges = loop.getConstantRanges();
    if (!ranges) {
      return loop.emitRemark("Vectorize: Failed, Requires constant ranges");
    }

    auto argNum = index.getArgNumber();
    if ((*ranges)[argNum] % vectorWidth != 0) {
      return loop.emitRemark(
          "Vectorize: Failed, dimension is not a multiple of the vector width");
    }

    auto steps = loop.getSteps();
    auto step = steps[argNum];
    if (step != 1) {
      return loop.emitRemark("Vectorize: Failed, dimension step is not 1");
    }

    Block *body = loop.getBody();
    bool vectorizable = llvm::all_of(*body, [&](Operation &op) {
      return succeeded(tryVectorizeOperation(&op));
    });
    if (!vectorizable) {
      return loop.emitRemark("Vectorize: Failed, !vectorizable");
    }
    if (vectorizedOps.empty()) {
      // TODO: should we actually fail in this case?  Currently we need to since
      // we have no cost model
      return loop.emitRemark("Vectorize: Failed, nothing to vectorize");
    }
    return success();
  }

  LogicalResult vectorize() {
    Block *body = loop.getBody();
    auto steps = loop.getSteps();
    for (auto &op : llvm::make_early_inc_range(body->getOperations())) {
      vectorizeOperation(&op);
    }
    auto argNum = index.getArgNumber();
    steps[argNum] *= vectorWidth;
    loop.setSteps(steps);
    return success();
  }
};

LogicalResult performVectorization(AffineParallelOp op, BlockArgument index,
                                   unsigned vectorWidth, bool vectorizeMathOp) {
  VectorizeCandidate candidate(op, index, vectorWidth, vectorizeMathOp);
  if (failed(candidate.isLegal())) {
    return failure();
  }
  // Preflight complete, do the transform
  return candidate.vectorize();
}

LogicalResult vectorizeOverOutputs(AffineParallelOp op, unsigned vectorWidth,
                                   bool vectorizeMathOp) {
  IVLOG(3, "Attempting to vectorize: " << debugString(*op));
  if (op.getNumResults() != 1) {
    return op.emitRemark("vectorizeOverOutputs: Failed, #result != 1");
  }
  auto reduce = dyn_cast<PxaReduceOp>(getPrevWriter(op.getResult(0)));
  if (!reduce) {
    return failure();
  }
  auto maybeSI = computeStrideInfo(reduce);
  if (!maybeSI) {
    return op.emitRemark(
        "vectorizeOverOutputs: Failed, could not compute StrideInfo");
  }
  IVLOG(3, "StrideInfo: " << debugString(*maybeSI));
  SmallVector<BlockArgument, 4> options;
  for (auto ba : op.getIVs()) {
    if (maybeSI->strides.count(ba) && maybeSI->strides[ba] == 1) {
      options.push_back(ba);
    }
  }
  if (options.size() != 1) {
    return op.emitRemark("vectorizeOverOutputs: Failed, options != 1");
  }
  return performVectorization(op, options[0], vectorWidth, vectorizeMathOp);
}

LogicalResult vectorizeOverIVs(AffineParallelOp band, unsigned vectorWidth,
                               bool vectorizeMathOp) {
  for (BlockArgument iv : band.getIVs()) {
    if (succeeded(performVectorization(band, iv, vectorWidth, vectorizeMathOp))) {
      break;
    }
  }
  return success();
}

LogicalResult vectorizeRecursive(AffineParallelOp band, unsigned vectorWidth,
                                 bool vectorizeMathOp) {
  band.walk([&](AffineParallelOp op) {
    for (BlockArgument iv : op.getIVs()) {
      if (succeeded(performVectorization(op, iv, vectorWidth, vectorizeMathOp))) {
        break;
      }
    }
  });
  return success();
}

using StrategyFn = std::function<LogicalResult(
    AffineParallelOp op, unsigned vectorWidth, bool vectorizeMathOp)>;

static llvm::StringMap<StrategyFn> strategies{
    {kVectorizeStrategy_Simple, vectorizeOverIVs},
    {kVectorizeStrategy_Outputs, vectorizeOverOutputs},
    {kVectorizeStrategy_Recursive, vectorizeRecursive},
};

struct VectorizePass : public VectorizeBase<VectorizePass> {
  VectorizePass() = default;

  explicit VectorizePass(StringRef strategy, unsigned vectorWidth,
                         bool vectorizeMathOp) {
    this->strategy = strategy.str();
    this->vectorWidth = vectorWidth;
    this->vectorizeMathOp = vectorizeMathOp;
  }

  void runOnFunction() final {
    auto func = getFunction();
    auto it = strategies.find(strategy);
    if (it == strategies.end()) {
      emitError(func.getLoc(), "Invalid strategy specified: ") << strategy;
      return signalPassFailure();
    }
    for (auto band : func.getOps<AffineParallelOp>()) {
      if (failed(it->second(band, vectorWidth, vectorizeMathOp))) {
        return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<Pass> createVectorizePass() {
  return std::make_unique<VectorizePass>();
}

std::unique_ptr<mlir::Pass>
createVectorizePass(StringRef strategy, unsigned vectorWidth, bool vectorizeMathOp) {
  return std::make_unique<VectorizePass>(strategy, vectorWidth, vectorizeMathOp);
}

// TODO: Maybe move this to a generic utility somewhere
template <typename OpTy, typename... Args>
static OpTy replaceOp(Operation *op, Args &&... args) {
  OpBuilder builder(op);
  auto newOp = builder.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  op->getResult(0).replaceAllUsesWith(newOp.getResult());
  op->erase();
  return newOp;
}

LogicalResult vectorizeBuffer(memref::AllocOp op) {
  // Verify that all uses are vector load/stores of the same width and with
  // valid minimum strides
  int64_t vectorWidth = 0;
  // Make generic lambda to verify/capture vector size
  auto validAccess = [&](auto vecOp) -> LogicalResult {
    auto shape = vecOp.getVectorType().getShape();
    if (shape.size() != 1) {
      return failure(); // Only support 1-d vectors
    }
    int64_t newSize = shape[0];
    if (vectorWidth && vectorWidth != newSize) {
      return failure(); // All vectors must have the same width
    }
    vectorWidth = newSize;
    assert(vectorWidth != 0 && "Vector shape should never be zero elements");
    auto maybeStride = computeStrideInfo(vecOp);
    if (!maybeStride) {
      return failure(); // Non strided access
    }
    auto range = maybeStride->range();
    if (range.stride % vectorWidth != 0) {
      return failure(); // Vector op not aligned
    }
    return success();
  };
  // Call the lambda on all uses + verify all are are valid vector ops
  for (auto &use : getIndirectAccessUses(op)) {
    if (auto vecOp = dyn_cast<PxaVectorLoadOp>(use.getOwner())) {
      if (failed(validAccess(vecOp))) {
        return failure();
      }
    } else if (auto vecOp = dyn_cast<PxaVectorReduceOp>(use.getOwner())) {
      if (failed(validAccess(vecOp))) {
        return failure();
      }
    } else {
      return failure(); // Non vector access detected
    }
  }
  // Exit early if no accesses
  if (!vectorWidth) {
    return failure();
  }
  // Compute new memref shape
  auto mtype = op.getType();
  auto mshape = mtype.getShape();
  if (mshape.size() < 1) {
    return failure(); // Can't perform transform on a scalar
  }
  SmallVector<int64_t, 4> newShape;
  for (size_t i = 0; i < mshape.size(); i++) {
    if (i == mshape.size() - 1) {
      // Last index, verify it's divisible vector size + reduce
      if (mshape[i] % vectorWidth != 0) {
        return failure();
      }
      newShape.push_back(mshape[i] / vectorWidth);
    } else {
      // Leave non-final indexes alone
      newShape.push_back(mshape[i]);
    }
  }
  // Make the new type
  auto newType = MemRefType::get(
      newShape, VectorType::get({vectorWidth}, mtype.getElementType()));
  // Replace the alloc
  auto newOp = replaceOp<memref::AllocOp>(op, newType);
  // Walk over the uses and update them all
  auto curUse = IndirectUsesIterator(newOp);
  auto updateMap = [&](AffineMap map) {
    SmallVector<AffineExpr, 4> affineExprs(map.getResults().begin(),
                                           map.getResults().end());
    unsigned lastIdx = affineExprs.size() - 1;
    affineExprs[lastIdx] = affineExprs[lastIdx].floorDiv(vectorWidth);
    return AffineMap::get(map.getNumDims(), map.getNumSymbols(), affineExprs,
                          map.getContext());
  };
  while (curUse != IndirectUsesIterator()) {
    curUse->get().setType(newType);
    if (auto vecOp = dyn_cast<PxaVectorLoadOp>(curUse->getOwner())) {
      curUse++;
      replaceOp<PxaLoadOp>(vecOp, vecOp.getMemRef(),
                           updateMap(vecOp.getAffineMap()),
                           vecOp.getMapOperands());
    } else if (auto vecOp = dyn_cast<PxaVectorReduceOp>(curUse->getOwner())) {
      auto newVecOp = replaceOp<PxaReduceOp>(
          vecOp, vecOp.agg(), vecOp.getValueToStore(), vecOp.getMemRef(),
          updateMap(vecOp.getAffineMap()), vecOp.getMapOperands());
      curUse = IndirectUsesIterator(newVecOp);
    } else {
      curUse++;
    }
  }
  return success();
}

} // namespace pmlc::dialect::pxa
