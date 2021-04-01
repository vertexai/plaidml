// Copyright 2020 Intel Corporation

#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::stdx {

// ---- ReshapeOp ----

Value ReshapeOp::getViewSource() { return tensor(); }

// ---- SubgroupBlockReadINTELOp ----

void SubgroupBlockReadINTELOp::build(OpBuilder &builder, OperationState &result,
                                     Value memref, ValueRange indices) {
  auto memrefType = memref.getType().cast<MemRefType>();
  result.addOperands(memref);
  result.addOperands(indices);
  result.addTypes(memrefType.getElementType());
}

static LogicalResult
verifySubgroupBlockReadINTELOp(SubgroupBlockReadINTELOp op) {
  if (op.getNumOperands() != 1 + op.getMemRefType().getRank())
    return op.emitOpError("incorrect number of indices for memory ref");
  return success();
}

// ---- SubgroupBlockWriteINTELOp ----

void SubgroupBlockWriteINTELOp::build(OpBuilder &builder,
                                      OperationState &result,
                                      Value valueToStore, Value memref) {
  result.addOperands(valueToStore);
  result.addOperands(memref);
}

static LogicalResult
verifySubgroupBlockWriteINTELOp(SubgroupBlockWriteINTELOp op) {
  if (op.getNumOperands() != 2 + op.getMemRefType().getRank())
    return op.emitOpError("subgroup block write index operand"
                          " count not equal to memref rank");

  return success();
}

Type StdXDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) {
    return Type();
  }
  if (keyword == "argpack") {
    return ArgpackType::get(getContext());
  }
  parser.emitError(parser.getNameLoc(), "unknown type: ") << keyword;
  return Type();
}

void StdXDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (type.isa<ArgpackType>()) {
    os << "argpack";
    return;
  }
  llvm_unreachable("unexpected stdx type kind");
}

void StdXDialect::initialize() {
  addTypes<ArgpackType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stdx/ir/ops.cc.inc" // NOLINT
      >();
}

} // namespace pmlc::dialect::stdx

#define GET_OP_CLASSES
#include "pmlc/dialect/stdx/ir/ops.cc.inc" // NOLINT
