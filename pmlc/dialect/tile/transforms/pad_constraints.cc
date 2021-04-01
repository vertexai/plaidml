// Copyright 2019, Intel Corporation

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/padding.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

bool isValidForPadding(AggregationKind agg, CombinationKind comb) {
  if (agg == AggregationKind::assign) {
    return false;
  }
  if (comb == CombinationKind::none) {
    return true;
  }
  if (comb == CombinationKind::mul && agg == AggregationKind::add) {
    return true;
  }
  return false;
}

// Helper function
IntegerSet makeConstraintSet(size_t numDims, ArrayRef<AffineExpr> cons) {
  auto allFalse = llvm::SmallVector<bool, 10>(cons.size());
  return IntegerSet::get(numDims, 0, cons, allFalse);
}

struct PadConstraintsPass : public PadConstraintsBase<PadConstraintsPass> {
  void runOnFunction() final;
};

void PadConstraintsPass::runOnFunction() {
  auto func = getFunction();
  llvm::DenseMap<Value, llvm::DenseMap<AggregationKind, PaddingInfo>> toPad;

  func.walk([&](ContractionOp op) {
    // Skip some cases where the padding pass can't operate.
    if (op.getNumSymbols()) {
      op.emitRemark("padding cannot be run on symbolic contractions");
      return;
    }

    if (!op.lowerBounds() || !op.upperBounds()) {
      op.emitRemark("contraction bounds must be computed");
      return;
    }

    if (!isValidForPadding(op.agg(), op.combo())) {
      op.emitRemark("invalid agg/comb for use in padding");
      return;
    }

    for (unsigned i = 0; i < op.getNumTensors(); i++) {
      auto tensor = op.getTensor(i);
      auto rankedTensorType = tensor.getType().cast<RankedTensorType>();
      if (!rankedTensorType.hasStaticShape()) {
        op.emitRemark("padding cannot support dynamic memref sizes");
        return;
      }

      // Compute the largest region the source may access.
      auto bounds = computePaddingBounds(op.getSourceMap(i), *op.lowerBounds(),
                                         *op.upperBounds());

      // Check if there are any out-of-bounds reads.
      auto shape = rankedTensorType.getShape();
      assert(bounds.size() == shape.size());
      bool needsPadding = false;
      for (unsigned i = 0, e = bounds.size(); i < e; ++i) {
        if (bounds[i].min < 0 || bounds[i].max >= shape[i]) {
          needsPadding = true;
        }
      }

      // If there is no need to pad, don't bother adding an entry.
      if (!needsPadding)
        return;

      // Merge discovered padding into the recorded data.
      auto &info = toPad[tensor][op.agg()];
      info.lower.resize(bounds.size());
      info.upper.resize(bounds.size());
      for (unsigned i = 0, e = bounds.size(); i < e; ++i) {
        info.lower[i] = std::max(info.lower[i], -bounds[i].min);
        info.upper[i] = std::max(info.upper[i], bounds[i].max + 1 - shape[i]);
      }
    }
  });

  OpBuilder builder(&getContext());
  for (const auto &kvp : toPad) {
    // Get the value which we want to pad.
    Value def = kvp.first;
    const auto &map = kvp.second;

    // If there are two conflicting ways to pad, don't pad.
    if (map.size() != 1) {
      if (Operation *op = def.getDefiningOp()) {
        op->emitRemark("padding would require multiple initialization values");
      }
      continue;
    }

    // Check if it's a block argument or unpack source, and if so add an IdentOp
    // to copy the value.
    if (def.isa<BlockArgument>() ||
        dyn_cast_or_null<stdx::UnpackOp>(def.getDefiningOp())) {
      OpBuilder inner(&getContext());
      inner.setInsertionPointAfterValue(def);
      // Construct an initial identity operation.
      auto ident = inner.create<IdentOp>(def.getLoc(), def.getType(), def);
      // Replace all uses with ident (except for newly generated use).
      def.replaceAllUsesExcept(ident, llvm::SmallPtrSet<Operation *, 1>{ident});
      // Now use ident for all further work.
      def = ident;
    }

    // Get defining operation (should now always work).
    Operation *op = def.getDefiningOp();
    assert(op);

    // Attach attributes specifying the padding details.
    AggregationKind agg = map.begin()->first;
    const auto &info = map.begin()->second;
    op->setAttr("padType",
                builder.getI64IntegerAttr(static_cast<int64_t>(agg)));
    op->setAttr("padLower", builder.getI64ArrayAttr(info.lower));
    op->setAttr("padUpper", builder.getI64ArrayAttr(info.upper));
  }

  // Finally, remove unnecessary constraints
  func.walk([&](ContractionOp op) {
    // Early exit if no constraints
    if (!op.cons())
      return;
    auto numDims = op.sink().getNumDims();
    // Otherwise, we will check each constraint against each input
    llvm::SmallVector<AffineExpr, 8> savedConstraints;
    llvm::SmallVector<llvm::SmallVector<AffineExpr, 10>, 2> inputs;
    // Go over all the input tensors + makey polyhedra for any padded ones
    for (size_t i = 0; i < op.getNumTensors(); i++) {
      Value in = op.getTensor(i);
      // If it's not padded, forget it.
      if (!getPaddingInfo(in.getDefiningOp()))
        continue;
      // Get the tensor and the map associated with the tensor
      auto ttype = in.getType().cast<TensorType>();
      auto map = op.getSourceMap(i);
      // The should have the same rank.
      assert(ttype.getRank() == map.getNumResults());
      // Add all of the constraints that define the valid region of a tensor
      inputs.emplace_back();
      auto &cons = inputs.back();
      for (int64_t dim = 0; dim < ttype.getRank(); dim++) {
        // Get dimension specific expression
        auto expr = map.getResult(dim);
        // Lower bound (expr >= 0)
        cons.emplace_back(expr);
        // Upper bound (expr <= (size - 1) or in canonical form:
        // (size - 1) - expr >= 0
        cons.emplace_back(-expr + (ttype.getDimSize(dim) - 1));
      }
    }
    // If no padded inputs, early exit
    if (inputs.empty())
      return;
    // Check each constraint by adding its inverse to each input polyhedra.
    // If the resulting integer set is empty, then all of the false cases are
    // outside of the valid region (i.e. in the padded region).  Thus any loads
    // when the constraint is false will load the padding value, and thus have
    // no effect on the output.  This means that removing the constraint is
    // safe, since all places it would have previously prevented will turn into
    // no-ops post padding.
    for (auto expr : op.cons()->getConstraints()) {
      // Compute the 'inverse' of the constraint.
      // The opposite of expr >= 0 is expr < 0
      // Given expr is an integer this is the same as expr + 1 <= 0 or
      // -expr - 1 >= 0
      auto inv_expr = -expr - 1;
      // Initially presume we will not keep this constraint
      bool keep = true;
      // For each input
      for (auto &cons : inputs) {
        // Temporarily add the inv_expr to the constraints
        cons.push_back(inv_expr);
        // Make into an IntegerSet, and then into FlatAffineConstraints.
        // It seems wasteful to intern a temporary integer set, but any other
        // way of doing this is also annoying given the current class structures
        auto set = makeConstraintSet(numDims, cons);
        FlatAffineConstraints fac(set);
        // Can't prove it's empty, keep constraint
        if (fac.isEmpty())
          keep = false;
        // Remove
        cons.pop_back();
        // If we've decided not to keep the constraint, don't bother considering
        // more inputs.
        if (!keep) {
          break;
        }
      }
      if (keep)
        savedConstraints.push_back(expr);
    }
    if (savedConstraints.empty()) {
      op->removeAttr(ContractionOp::getConstraintsAttrName());
    } else {
      op.setConstraints(makeConstraintSet(numDims, savedConstraints));
    }
  });
}

} // namespace

llvm::Optional<PaddingInfo> getPaddingInfo(Operation *op) {
  if (!op)
    return llvm::None;
  auto padType = op->getAttrOfType<IntegerAttr>("padType");
  if (!padType)
    return llvm::None;
  auto agg = util::symbolizeAggregationKind(padType.getInt());
  if (!agg)
    return llvm::None;

  PaddingInfo ret{*agg};
  auto padLower = op->getAttrOfType<ArrayAttr>("padLower");
  if (!padLower)
    return llvm::None;
  for (auto attr : padLower.getAsRange<IntegerAttr>()) {
    ret.lower.push_back(attr.getInt());
  }

  auto padUpper = op->getAttrOfType<ArrayAttr>("padUpper");
  if (!padUpper)
    return llvm::None;
  for (auto attr : padUpper.getAsRange<IntegerAttr>()) {
    ret.upper.push_back(attr.getInt());
  }

  return ret;
}

std::unique_ptr<Pass> createPadConstraintsPass() {
  return std::make_unique<PadConstraintsPass>();
}

} // namespace pmlc::dialect::tile
