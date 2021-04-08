// Copyright 2020, Intel Corporation

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SetVector.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/layout.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

int64_t getArgPos(Operation &op, BlockArgument &arg) {
  auto argPos = 0;
  for (size_t i = 0; i < op.getOperands().size(); i++) {
    if (op.getOperand(i) == arg) {
      argPos = i;
    }
  }
  return argPos;
}

void performDataReordering(OpBuilder &builder, Operation *op, BlockArgument arg,
                           util::MLFramework framework) {
  // Check if op is a BoxOp, if so scan which argument is used,
  // and get its tensor type.
  // Later on create the new contraction op with the same sizes and additional
  // layout tag. These ops would be moved to init as weights reorders in the
  // later pass.
  auto layerOp = dyn_cast<layer::BoxOp>(op);
  if (!layerOp)
    return;

  auto argPos = getArgPos(*op, arg);
  auto tensorType =
      op->getOperand(argPos).getType().dyn_cast<mlir::RankedTensorType>();
  if (!tensorType)
    return;

  Type elementType = tensorType.getElementType();
  auto ident = tile::createIdentity(builder, op->getLoc(), elementType,
                                    AggregationKind::assign);
  auto outRank = tensorType.getRank();
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(outRank);

  for (unsigned i = 0; i < outRank; i++) {
    dimExprs.push_back(mlir::getAffineDimExpr(i, op->getContext()));
  }
  auto idMap = AffineMap::get(outRank, 0, dimExprs, op->getContext());

  std::stringstream ss;
  ss << "reordered_" << arg.getArgNumber();

  auto newOp = builder.create<ContractionOp>(
      op->getLoc(), op->getOperand(argPos).getType(), ident,
      ArrayRef<Value>{op->getOperand(argPos)}, AggregationKind::assign,
      CombinationKind::none, idMap, ArrayRef<AffineMap>{idMap},
      IntegerSet::getEmptySet(outRank, 0, op->getContext()), ss.str());
  newOp.setLowerBounds(SmallVector<int64_t, 4>(outRank, 0));

  newOp.setUpperBounds(tensorType.getShape());

  // Switch all uses to the new contraction
  op->getOperand(argPos).replaceAllUsesExcept(
      newOp.getResult(), SmallPtrSet<Operation *, 1>{newOp.getOperation()});

  // Add layout tag
  setLayoutTag(newOp, getLayoutType(framework, layerOp.op(), true));
}

struct ReorderInputsPass : public ReorderInputsBase<ReorderInputsPass> {
  void runOnFunction() final {
    auto func = getFunction();

    auto layoutSet = false;
    auto framework = util::MLFramework::none;

    // Get Framework and check if at least one BoxOp is present
    func.walk([&](layer::BoxOp op) {
      framework = util::getMLFramework(op.op());
      layoutSet = true;
      return;
    });

    // If no BoxOp present or no framework defined then terminate
    if (!layoutSet || framework == util::MLFramework::none)
      return;

    // Set proper layout tags for the BoxOps
    func.walk([&](layer::BoxOp op) {
      auto outputLayer = false;
      for (auto *user : op->getUsers()) {
        if (isa<mlir::ReturnOp>(user))
          outputLayer = true;
      }
      for (auto &innerOp : op.getBody()->getOperations()) {
        if (!outputLayer)
          setLayoutTag(&innerOp, getLayoutType(framework, op.op()));
      }
    });

    // Go over each function argument to set correct layouts
    // in the inner tile ops if they are used
    auto &block = func.getBody().front();
    Operation *op = &block.front();
    OpBuilder builder(op);

    for (auto &arg : func.getArguments()) {
      for (auto *op : arg.getUsers()) {
        if (func.getArgAttr(arg.getArgNumber(), "tile.const"))
          performDataReordering(builder, op, arg, framework);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createReorderInputsPass() {
  return std::make_unique<ReorderInputsPass>();
}

} // namespace pmlc::dialect::tile
