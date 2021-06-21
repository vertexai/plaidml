// Copyright 2020 Intel Corporation

#include "pmlc/ast/ast.h"

#include <algorithm>
#include <sstream>

#include "llvm/Support/FormatVariadic.h"

#include "pmlc/util/logging.h"

namespace pmlc::ast {

using util::DataType;
using util::TensorShape;

static llvm::StringRef getAffineOpStr(AffineOp op) {
  switch (op) {
  case AffineOp::Add:
    return "add";
  case AffineOp::Div:
    return "div";
  case AffineOp::Max:
    return "max";
  case AffineOp::Min:
    return "min";
  case AffineOp::Mul:
    return "mul";
  case AffineOp::Neg:
    return "neg";
  case AffineOp::Sub:
    return "sub";
  default:
    return "<invalid op>";
  }
  llvm_unreachable("getAffineOpStr");
}

//
// ExprNode
//

ExprNode::ExprNode(llvm::StringRef name) : name(name) {}

//
// ExprNodeCast
//

ExprNodeCast::ExprNodeCast(DataType dtype, const ExprNodePtr &expr)
    : dtype(dtype), expr(expr) {}

std::string ExprNodeCast::str() const { return "cast"; }

//
// ExprNodeConstSsigned
//

ExprNodeConstSigned::ExprNodeConstSigned(int64_t value) : value(value) {}

std::string ExprNodeConstSigned::str() const {
  return llvm::formatv("{0}:six", value);
}

//
// ExprNodeConstUnsigned
//

ExprNodeConstUnsigned::ExprNodeConstUnsigned(uint64_t value) : value(value) {}

std::string ExprNodeConstUnsigned::str() const {
  return llvm::formatv("{0}:uix", value);
}

//
// ExprNodeConstFloat
//

ExprNodeConstFloat::ExprNodeConstFloat(double value) : value(value) {}

std::string ExprNodeConstFloat::str() const {
  return llvm::formatv("{0}:fx", value);
}

//
// ExprNodeConstTensor
//

ExprNodeConstTensor::ExprNodeConstTensor(const util::BufferPtr &buffer,
                                         llvm::StringRef name)
    : Base(name), buffer(buffer) {}

std::string ExprNodeConstTensor::str() const {
  return llvm::formatv("constant_tensor({0})", name);
}

std::string Constraint::str() const {
  return llvm::formatv("{0} < {1}", lhs->str(), rhs->str());
}

//
// ExprNodeContraction
//

ExprNodeContraction::ExprNodeContraction(llvm::StringRef name) : Base(name) {}

std::string ExprNodeContraction::str() const {
  std::stringstream ss;
  ss << util::stringifyAggregationKind(aggKind).str() << '(';
  // for (auto item : llvm::enumerate(srcs)) {
  //   if (item.index()) {
  //     ss << ", ";
  //   }
  //   ss << item.value().ref->str();
  // }
  ss << ')';
  return ss.str();
}

//
// ExprNodeDim
//

ExprNodeDim::ExprNodeDim(const DimNodePtr &dim) : dim(dim) {}

std::string ExprNodeDim::str() const { return dim->str(); }

//
// ExprNodeElement
//

ExprNodeElement::ExprNodeElement(const ExprNodePtr &expr, size_t ordinal)
    : expr(expr), ordinal(ordinal) {}

std::string ExprNodeElement::str() const {
  return llvm::formatv("element({0}, {1})", expr->str(), ordinal);
}

//
// ExprNodeInput
//

ExprNodeInput::ExprNodeInput(const TensorShape &shape, llvm::StringRef name)
    : Base(name), shape(shape) {
  if (shape.elementType == DataType::invalid) {
    throw std::runtime_error("DType::INVALID not appropriate here");
  }
}

std::string ExprNodeInput::str() const {
  if (name.size()) {
    return llvm::formatv("input({0}, \"{1}\")", shape.str(), name);
  }
  return llvm::formatv("input({0})", shape.str());
}

//
// ExprNodeIntrinsic
//

ExprNodeIntrinsic::ExprNodeIntrinsic(llvm::StringRef op,
                                     llvm::ArrayRef<ExprNodePtr> operands)
    : op(op), operands(operands) {}

std::string ExprNodeIntrinsic::str() const {
  std::stringstream ss;
  ss << op << '(';
  // for (auto item : llvm::enumerate(operands)) {
  //   if (item.index()) {
  //     ss << ", ";
  //   }
  //   ss << item.value()->str();
  // }
  ss << ')';
  return ss.str();
}

//
// ExprNodeLayer
//

ExprNodeLayer::ExprNodeLayer(llvm::StringRef op,
                             llvm::ArrayRef<ExprNodePtr> operands,
                             const llvm::StringMap<VarNodePtr> &attrs)
    : op(op), operands(operands), attrs(attrs) {}

std::string ExprNodeLayer::str() const {
  return llvm::formatv("layer({0})", op);
}

//
// ExprNodeLoop
//

ExprNodeLoop::ExprNodeLoop(llvm::StringRef op,
                           llvm::ArrayRef<ExprNodePtr> operands,
                           llvm::ArrayRef<ExprNodePtr> results)
    : op(op), operands(operands), results(results) {}

std::string ExprNodeLoop::str() const { return llvm::formatv("loop({0})", op); }

//
// ExprNodePragma
//

ExprNodePragma::ExprNodePragma(const ExprNodePtr &expr, llvm::StringRef op,
                               const llvm::StringMap<VarNodePtr> &attrs)
    : expr(expr), op(op), attrs(attrs) {}

std::string ExprNodePragma::str() const { return op; }

//
// DimNode tree
//

std::string DimNodeLiteral::str() const { return llvm::formatv("{0}", value); }

std::string DimNodeOp::str() const {
  std::stringstream ss;
  ss << getAffineOpStr(op).str() << '(';
  for (auto item : llvm::enumerate(operands)) {
    if (item.index()) {
      ss << ", ";
    }
    ss << item.value()->str();
  }
  ss << ')';
  return ss.str();
}

std::string DimNodeRef::str() const {
  return llvm::formatv("dim({0}, {1})", ref->str(), dim);
}

//
// PolyNode tree
//

std::string PolyNodeDim::str() const { return dim->str(); }

std::string PolyNodeIndex::str() const {
  if (name.empty()) {
    return llvm::formatv("%{0}", this);
  }
  return llvm::formatv("%{0}", name);
}

std::string PolyNodeLiteral::str() const { return std::to_string(value); }

std::string PolyNodeOp::str() const {
  std::stringstream ss;
  ss << getAffineOpStr(op).str() << '(';
  for (auto item : llvm::enumerate(operands)) {
    if (item.index()) {
      ss << ", ";
    }
    ss << item.value()->str();
  }
  ss << ')';
  return ss.str();
}

//
// VarNode tree
//

std::string VarNodeTuple::str() const {
  std::stringstream ss;
  ss << '(';
  for (auto item : llvm::enumerate(values)) {
    if (item.index()) {
      ss << ", ";
    }
    ss << item.value()->str();
  }
  ss << ')';
  return ss.str();
}

} // namespace pmlc::ast
