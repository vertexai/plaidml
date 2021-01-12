// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

struct DeallocPlacementPass
    : public DeallocPlacementBase<DeallocPlacementPass> {
  void runOnOperation() final {
    // Get the module
    ModuleOp op = getOperation();
    // Run all functions.  This could almost be a function pass, but init + fini
    // interact, which breaks the independence requirements
    op.walk([&](FuncOp fn) {
      if (fn.getName() == "init") {
        // If the function is named init, find fini
        auto finiFunc = op.lookupSymbol<FuncOp>("fini");
        if (!finiFunc) {
          fn.emitError() << "Init with no fini";
          signalPassFailure();
          return;
        }
        // Find fini's unpack op
        auto unpackOp = dyn_cast<stdx::UnpackOp>(finiFunc.begin()->begin());
        if (!unpackOp) {
          finiFunc.emitError() << "Fini must begin with unpack";
          signalPassFailure();
          return;
        }
        // Now, place deallocs on the init functions, moving escaping allocs to
        // be dealloced in fini
        OpBuilder deallocBuilder(finiFunc.begin()->getTerminator());
        runOnFunction(fn, [&](unsigned i) {
          deallocBuilder.create<DeallocOp>(fn.getLoc(), unpackOp.getResult(i));
        });
      } else {
        // Place allocs, if any escape, it's an error
        runOnFunction(
            fn, [&](unsigned i) {
              fn.emitError()
                  << "Allocations escape via a pack for non-init function";
              signalPassFailure();
            });
      }
    });
  }

  template <typename Callback>
  void runOnFunction(FuncOp fn, Callback onPack) {
    fn.walk([&](AllocOp alloc) {
      IVLOG(3, "alloc: " << debugString(*alloc));

      Operation *lastOp = alloc;
      OpOperand *lastUse = nullptr;
      Block *allocBlock = alloc.getOperation()->getBlock();
      for (auto &itUse : getIndirectUses(alloc)) {
        auto use = itUse.getOwner();
        IVLOG(3, "  use: " << debugString(*use));

        auto ancestor = allocBlock->findAncestorOpInBlock(*use);
        assert(ancestor && "use and alloc do not have a common ancestor");
        IVLOG(3, "  ancestor: " << debugString(*use));

        if (isa<ReturnOp>(ancestor)) {
          IVLOG(3, "  return");
          return;
        }

        if (!ancestor->isBeforeInBlock(lastOp)) {
          lastOp = ancestor;
          lastUse = &itUse;
        }
      }
      IVLOG(3, "  last ancestor: " << debugString(*lastOp));
      if (auto packOp = dyn_cast<stdx::PackOp>(lastOp)) {
        IVLOG(3, "  pack op");
        // Alloc 'escapes' via a pack, call our callback to handle
        onPack(lastUse->getOperandNumber());
        return;
      }

      Operation *nextOp = lastOp->getNextNode();
      if (!nextOp) {
        IVLOG(3, "  terminator");
        return;
      }

      IVLOG(3, "  next operation: " << debugString(*nextOp));
      OpBuilder builder(nextOp);
      builder.create<DeallocOp>(alloc.getLoc(), alloc);
    });
  }
};

std::unique_ptr<mlir::Pass> createDeallocPlacementPass() {
  return std::make_unique<DeallocPlacementPass>();
}

} // namespace pmlc::dialect::pxa
