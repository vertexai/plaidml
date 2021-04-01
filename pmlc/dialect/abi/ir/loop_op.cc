// Copyright 2020, Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"

#include "pmlc/dialect/abi/ir/dialect.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::abi {

void LoopOp::build(OpBuilder &odsBuilder, OperationState &odsState) {
  odsState.addAttribute("networkFieldTypes", odsBuilder.getTypeArrayAttr({}));
  odsState.addRegion();
  odsState.addRegion();
  odsState.addRegion();
}

std::vector<Type> LoopOp::getNetworkFieldTypes() {
  std::vector<Type> result;
  auto arrayAttr = networkFieldTypes();
  if (arrayAttr) {
    for (auto attr : arrayAttr) {
      if (auto tyAttr = attr.dyn_cast<TypeAttr>()) {
        result.emplace_back(tyAttr.getValue());
      }
    }
  }
  return result;
}

unsigned LoopOp::getNumNetworkFields() {
  auto arrayAttr = networkFieldTypes();
  if (arrayAttr) {
    return arrayAttr.size();
  }
  return 0;
}

void LoopOp::setNetworkFieldTypes(TypeRange types) {
  SmallVector<Attribute, 8> attrs;
  for (auto ty : types) {
    attrs.emplace_back(TypeAttr::get(ty));
  }
  auto arrayAttr = ArrayAttr::get(getContext(), attrs);
  networkFieldTypesAttr(arrayAttr);
}

Block *LoopOp::getBodyEntryBlock() { return &bodyRegion().front(); }
Block *LoopOp::getFiniEntryBlock() { return &finiRegion().front(); }

YieldOp LoopOp::getInitTerminator() {
  return cast<YieldOp>(initRegion().back().getTerminator());
}

TerminatorOp LoopOp::getFiniTerminator() {
  return cast<TerminatorOp>(finiRegion().back().getTerminator());
}

} // namespace pmlc::dialect::abi
