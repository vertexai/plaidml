// Copyright 2019, Intel Corporation

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::stdx; // NOLINT

namespace pmlc::dialect::tile {

namespace {

struct ScfForReplacePass : public ScfForReplaceBase<ScfForReplacePass> {
  void runOnFunction() final {
    auto funcOp = getFunction();
    if (funcOp.getName() == "init" || funcOp.getName() == "fini") {
      return;
    }
    funcOp.walk([&](ReturnOp op) {
      auto &block = op->getParentRegion()->front();
      auto funcOp = op->getParentOfType<FuncOp>();
      auto blockArg = funcOp.getType().getNumInputs() - op.getNumOperands();
      auto operands = op.getOperands();
      for (Value operand : operands){
        if (!dyn_cast<scf::ForOp>(operand.getDefiningOp())){
          blockArg++;
        }
      }
      for (Value operand : operands) {
        if (dyn_cast<scf::ForOp>(operand.getDefiningOp())){
          auto def = pxa::getIndirectDef(operand);
          def.replaceAllUsesWith(block.getArgument(blockArg++));
        }
      }
    });
    return;
  }
};

} // namespace

std::unique_ptr<Pass> createScfForReplacePass() {
  return std::make_unique<ScfForReplacePass>();
}

} // namespace pmlc::dialect::tile
