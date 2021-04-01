// Copyright 2020 Intel Corporation

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/ir/util.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  auto oldType = getRankedTensorType(tensor().getType());
  auto newType = getRankedTensorType(result().getType());
  /// cast(x).type == type -> x
  if (oldType == newType) {
    return tensor();
  }
  Attribute attr;
  if (matchPattern(tensor(), m_Constant(&attr))) {
    return attr;
  }
  return {};
}

static OpFoldResult safeFold(Attribute attr, Value val) {
  if (attr)
    return attr;
  return val;
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
  /// add(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    return safeFold(operands[0], lhs());
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a + b; });
}

OpFoldResult DivOp::fold(ArrayRef<Attribute> operands) {
  // don't fold division by zero
  // modeling this choice on DivUIOp::Fold from the standard dialect
  if (matchPattern(rhs(), m_Zero())) {
    return {};
  }
  // div(x, 1) -> x
  if (matchPattern(rhs(), m_One())) {
    return safeFold(operands[0], lhs());
  }
  // div(0, x) -> 0
  if (matchPattern(lhs(), m_Zero())) {
    return operands[0];
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a / b; });
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> operands) {
  // mul(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero())) {
    return operands[1];
  }
  // mul(x, 1) -> x
  if (matchPattern(rhs(), m_One())) {
    return safeFold(operands[0], lhs());
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a * b; });
}

OpFoldResult SubOp::fold(ArrayRef<Attribute> operands) {
  // sub(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    return safeFold(operands[0], lhs());
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a - b; });
}

LogicalResult SelectOp::materializeOperands(OpBuilder &builder) {
  Operation *op = getOperation();
  return tile::materializeOperands(builder, op,
                                   op->getOpOperands().drop_front());
}

} // namespace pmlc::dialect::tile
