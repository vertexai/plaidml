// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/analysis/uses.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/ADT/TypeSwitch.h"

#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

Value getPrevIndirectDef(OpResult def) {
  return TypeSwitch<Operation *, Value>(def.getOwner())
      .Case<AffineParallelOp>([&](auto op) {
        auto yield = cast<AffineYieldOp>(op.getBody()->getTerminator());
        return yield.getOperand(def.getResultNumber());
      })
      .Case<AffineIfOp>([&](auto op) {
        auto yield = cast<AffineYieldOp>(op.getThenBlock()->getTerminator());
        return yield.getOperand(def.getResultNumber());
      })
      .Case<scf::ForOp>([&](auto op) {
        auto yield = cast<scf::YieldOp>(op.getBody()->getTerminator());
        return yield.getOperand(def.getResultNumber());
      })
      .Case<layer::BoxOp>([&](auto op) {
        return op.getOperand(def.getResultNumber()); //
      })
      .Case<PrngOp>([&](auto op) {
        if (op.getResult(def.getResultNumber()) == op.result_tensor()) {
          return op.tensor();
        }
        if (op.getResult(def.getResultNumber()) == op.result_state()) {
          return op.new_state();
        }
        return Value();
      })
      .Case<PxaReduceOp>([&](auto op) { return op.memref(); })
      .Case<PxaVectorReduceOp>([&](auto op) { return op.memref(); })
      .Case<PxaGemmOp>([&](auto op) { return op.c(); })
      .Case<stdx::ReshapeOp>([&](auto op) { return op.tensor(); })
      .Default([](auto op) { return nullptr; });
}

Value getNextIndirectUse(mlir::OpOperand &use) {
  return TypeSwitch<Operation *, Value>(use.getOwner())
      .Case<AffineYieldOp>([&](auto op) {
        return op->getParentOp()->getResult(use.getOperandNumber());
      })
      .Case<layer::BoxOp>([&](auto op) {
        return op.getResult(use.getOperandNumber()); //
      })
      .Case<PxaReduceOp>([&](auto op) { return op.result(); })
      .Case<PxaVectorReduceOp>([&](auto op) { return op.result(); })
      .Case<PxaGemmOp>([&](auto op) {
        if (op.getOperand(use.getOperandNumber()) == op.c()) {
          return op.out();
        }
        return Value();
      })
      .Case<PrngOp>([&](auto op) {
        if (op.getOperand(use.getOperandNumber()) == op.tensor()) {
          return op.result_tensor();
        }
        if (op.getOperand(use.getOperandNumber()) == op.new_state()) {
          return op.result_state();
        }
        return Value();
      })
      .Case<stdx::ReshapeOp>([&](auto op) { return op.result(); })
      .Default([](auto op) { return nullptr; });
}

Operation *getPrevWriter(Value value) {
  while (auto opResult = value.dyn_cast<OpResult>()) {
    auto op = opResult.getOwner();
    if (isa<AffineParallelOp, AffineIfOp>(op)) {
      value = getPrevIndirectDef(opResult);
    } else {
      return op;
    }
  }
  return nullptr;
}

Value getIndirectDef(Value value) {
  while (auto opResult = value.dyn_cast<OpResult>()) {
    value = getPrevIndirectDef(opResult);
    if (!value) {
      return opResult;
    }
  }
  return value;
}

Value getIndirectDefOutsideScope(Value value, Operation *scope) {
  while (auto opResult = value.dyn_cast<OpResult>()) {
    auto op = opResult.getOwner();
    if (!scope->isAncestor(op)) {
      return opResult;
    }
    value = getPrevIndirectDef(opResult);
    if (!value) {
      return nullptr;
    }
  }
  return value;
}

IndirectValuesIterator &IndirectValuesIterator::operator++() {
  for (OpOperand &use : curValue.getUses()) {
    if (auto next = getNextIndirectUse(use)) {
      enqueueNext(next);
    }
  }
  if (workQueue.empty()) {
    curValue = nullptr;
  } else {
    curValue = workQueue.front();
    workQueue.pop();
  }
  return *this;
}

void IndirectValuesIterator::enqueueNext(Value value) {
  if (!visited.count(value)) {
    visited.insert(value);
    workQueue.push(value);
  }
}

IndirectValuesRange getIndirectValues(Value value) {
  return {IndirectValuesIterator(value), IndirectValuesIterator()};
}

IndirectUsesIterator::IndirectUsesIterator(Value value)
    : inner(value), curIt(value.use_begin()) {
  nextValue();
}

IndirectUsesIterator &IndirectUsesIterator::operator++() {
  ++curIt;
  nextValue();
  return *this;
}

void IndirectUsesIterator::nextValue() {
  while (curIt == Value::use_iterator()) {
    ++inner;
    if (inner == IndirectValuesIterator()) {
      return;
    }
    curIt = inner.getValue().use_begin();
  }
}

IndirectUsesRange getIndirectUses(Value value) {
  return {IndirectUsesIterator(value), IndirectUsesIterator()};
}

IndirectAccessUsesIterator &IndirectAccessUsesIterator::operator++() {
  ++inner;
  skipNonAccess();
  return *this;
}

void IndirectAccessUsesIterator::skipNonAccess() {
  while (inner != IndirectUsesIterator()) {
    if (isa<PxaLoadOp, PxaReduceOp, PxaVectorLoadOp, PxaVectorReduceOp>(
            inner->getOwner())) {
      break;
    }
    ++inner;
  }
}

IndirectAccessUsesRange getIndirectAccessUses(Value value) {
  return {IndirectAccessUsesIterator(value), IndirectAccessUsesIterator()};
}

} // namespace pmlc::dialect::pxa
