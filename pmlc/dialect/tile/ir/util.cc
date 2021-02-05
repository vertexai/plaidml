// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ir/util.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/ir/types.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

using llvm::ArrayRef;
using llvm::SmallVector;

// I64EnumAttrCase<"invalid", 0>,
// I64EnumAttrCase<"u1",      1>,
// I64EnumAttrCase<"i8",      2>,
// I64EnumAttrCase<"u8",      3>,
// I64EnumAttrCase<"i16",     4>,
// I64EnumAttrCase<"u16",     5>,
// I64EnumAttrCase<"i32",     6>,
// I64EnumAttrCase<"u32",     7>,
// I64EnumAttrCase<"i64",     8>,
// I64EnumAttrCase<"u64",     9>,
// I64EnumAttrCase<"f16",    11>,
// I64EnumAttrCase<"f32",    12>,
// I64EnumAttrCase<"f64",    13>,
unsigned typeScore(Type type) {
  if (!type || type.isIndex()) {
    return 0;
  }
  if (type.isa<APSignedIntegerType>()) {
    return 1;
  }
  if (type.isa<APUnsignedIntegerType>()) {
    return 2;
  }
  if (type.isa<APFloatType>()) {
    return 3;
  }
  if (type.isInteger(1)) {
    return 4;
  }
  if (type.isSignedInteger(8)) {
    return 5;
  }
  if (type.isUnsignedInteger(8)) {
    return 6;
  }
  if (type.isSignedInteger(16)) {
    return 7;
  }
  if (type.isUnsignedInteger(16)) {
    return 8;
  }
  if (type.isSignedInteger(32)) {
    return 9;
  }
  if (type.isUnsignedInteger(32)) {
    return 10;
  }
  if (type.isSignlessInteger(32)) {
    return 10;
  }
  if (type.isSignedInteger(64)) {
    return 11;
  }
  if (type.isUnsignedInteger(64)) {
    return 12;
  }
  if (type.isF16()) {
    return 13;
  }
  if (type.isF32()) {
    return 14;
  }
  if (type.isF64()) {
    return 15;
  }
  IVLOG(1, "Type: " << debugString(type));
  assert(false && "Undefined typeScore");
  return 0;
}

Type promoteTypes(Type lhs, Type rhs) {
  return typeScore(lhs) > typeScore(rhs) ? lhs : rhs;
}

RankedTensorType getRankedTensorType(Type type) {
  if (auto rankedType = type.dyn_cast<RankedTensorType>()) {
    return rankedType;
  }
  SmallVector<int64_t, 0> shape;
  if (type.isa<IndexType>()) {
    // TODO: reify this when we lower to a specific target.
    return RankedTensorType::get(
        shape, IntegerType::get(type.getContext(), 32,
                                IntegerType::SignednessSemantics::Signed));
  }
  return RankedTensorType::get(shape, type);
}

Attribute constFoldUnaryOp(ArrayRef<Attribute> operands,
                           UnaryCalculate calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (auto op = operands[0].dyn_cast_or_null<FloatAttr>()) {
    return FloatAttr::get(op.getType(), calculate(op.getValueAsDouble()));
  }
  if (auto op = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    return IntegerAttr::get(op.getType(), calculate(op.getInt()));
  }
  return {};
}

Attribute constFoldBinaryOp(ArrayRef<Attribute> operands,
                            BinaryCalculate calculate) {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (auto lhs = operands[0].dyn_cast_or_null<FloatAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<FloatAttr>()) {
      return FloatAttr::get(lhs.getType(), calculate(lhs.getValueAsDouble(),
                                                     rhs.getValueAsDouble()));
    }
    if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      return FloatAttr::get(lhs.getType(),
                            calculate(lhs.getValueAsDouble(), rhs.getInt()));
    }
  } else if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<FloatAttr>()) {
      return FloatAttr::get(rhs.getType(),
                            calculate(lhs.getInt(), rhs.getValueAsDouble()));
    }
    if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      return IntegerAttr::get(lhs.getType(),
                              calculate(lhs.getInt(), rhs.getInt()));
    }
  }
  return {};
}

bool ConstantValueMatcher::match(Operation *op) {
  Attribute attr;
  if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op)) {
    return false;
  }
  if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
    return intAttr.getValue() == value;
  }
  if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
    return floatAttr.getValueAsDouble() == value;
  }
  return false;
}

Type toSignlessType(Type type) {
  if (auto integerType = type.dyn_cast<IntegerType>()) {
    return IntegerType::get(type.getContext(), integerType.getWidth());
  }
  return type;
}

LogicalResult materializeOperands(OpBuilder &builder, Operation *op,
                                  llvm::ArrayRef<OpOperand *> operands) {
  Type promotedType;
  for (OpOperand *operand : operands) {
    Type type = operand->get().getType();
    RankedTensorType rankedTensorType = getRankedTensorType(type);
    Type elementType = rankedTensorType.getElementType();
    promotedType = promoteTypes(promotedType, elementType);
  }

  for (OpOperand *operand : operands) {
    RankedTensorType rankedTensorType =
        getRankedTensorType(operand->get().getType());
    Type elementType = rankedTensorType.getElementType();
    if (elementType != promotedType &&
        (elementType.isa<APFloatType>() ||
         elementType.isa<APSignedIntegerType>() ||
         elementType.isa<APUnsignedIntegerType>())) {
      RankedTensorType newType =
          RankedTensorType::get(rankedTensorType.getShape(), promotedType);
      Value value =
          builder.create<CastOp>(op->getLoc(), newType, operand->get());
      operand->set(value);
    }
  }

  return success();
}

LogicalResult materializeOperands(OpBuilder &builder, Operation *op,
                                  llvm::MutableArrayRef<OpOperand> operands) {
  std::vector<OpOperand *> ptrs;
  ptrs.reserve(operands.size());
  for (OpOperand &operand : operands) {
    ptrs.push_back(&operand);
  }
  return materializeOperands(builder, op, ptrs);
}

LogicalResult materializeOperands(OpBuilder &builder, Operation *op) {
  return materializeOperands(builder, op, op->getOpOperands());
}

double getFloatIdentity(util::AggregationKind agg) {
  switch (agg) {
  case util::AggregationKind::assign:
  case util::AggregationKind::add:
    return 0.0;
  case util::AggregationKind::mul:
    return 1.0;
  case util::AggregationKind::min:
    return std::numeric_limits<double>::infinity();
  case util::AggregationKind::max:
    return -std::numeric_limits<double>::infinity();
  }
  llvm_unreachable("Invalid aggregation kind");
}

int64_t getSignedIntegerIdentity(util::AggregationKind agg) {
  switch (agg) {
  case util::AggregationKind::assign:
  case util::AggregationKind::add:
    return 0;
  case util::AggregationKind::mul:
    return 1;
  case util::AggregationKind::min:
    return std::numeric_limits<int64_t>::max();
  case util::AggregationKind::max:
    return std::numeric_limits<int64_t>::min();
  }
  llvm_unreachable("Invalid aggregation kind");
}

uint64_t getUnsignedIntegerIdentity(util::AggregationKind agg) {
  switch (agg) {
  case util::AggregationKind::assign:
  case util::AggregationKind::add:
    return 0;
  case util::AggregationKind::mul:
    return 1;
  case util::AggregationKind::min:
    return std::numeric_limits<uint64_t>::max();
  case util::AggregationKind::max:
    return 0;
  }
  llvm_unreachable("Invalid aggregation kind");
}

Value createIdentity(OpBuilder &builder, Location loc, Type elementType,
                     util::AggregationKind agg) {
  if (elementType.isa<FloatType>()) {
    return builder.create<tile::ConstantOp>(loc, elementType,
                                            getFloatIdentity(agg));
  }
  if (elementType.isSignedInteger()) {
    return builder.create<tile::ConstantOp>(loc, elementType,
                                            getSignedIntegerIdentity(agg));
  }
  return builder.create<tile::ConstantOp>(loc, elementType,
                                          getUnsignedIntegerIdentity(agg));
}

} // namespace pmlc::dialect::tile
