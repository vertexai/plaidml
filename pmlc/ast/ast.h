// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include "pmlc/util/buffer.h"
#include "pmlc/util/enums.h"
#include "pmlc/util/shape.h"

namespace pmlc::ast {

struct ExprNode;
struct ExprNodeCast;
struct ExprNodeConstSigned;
struct ExprNodeConstUnsigned;
struct ExprNodeConstFloat;
struct ExprNodeConstTensor;
struct ExprNodeContraction;
struct ExprNodeDim;
struct ExprNodeElement;
struct ExprNodeInput;
struct ExprNodeIntrinsic;
struct ExprNodeLayer;
struct ExprNodeLoop;
struct ExprNodePragma;

struct DimNode;
struct DimNodeLiteral;
struct DimNodeNone;
struct DimNodeOp;
struct DimNodeRef;

struct PolyNode;
struct PolyNodeDim;
struct PolyNodeIndex;
struct PolyNodeLiteral;
struct PolyNodeOp;

struct VarNode;
struct VarNodeDim;
struct VarNodeExpr;
struct VarNodeFloat;
struct VarNodeInt;
struct VarNodeNone;
struct VarNodeString;
struct VarNodeTuple;

using ExprNodePtr = std::shared_ptr<ExprNode>;
using DimNodePtr = std::shared_ptr<DimNode>;
using PolyNodePtr = std::shared_ptr<PolyNode>;
using VarNodePtr = std::shared_ptr<VarNode>;

enum class AffineOp {
  Add,
  Div,
  Max,
  Min,
  Mul,
  Neg,
  Sub,
};

//
// Base AST Node
//

template <typename ConcreteT, typename BaseT>
struct NodeBase : BaseT, std::enable_shared_from_this<ConcreteT> {
  using BaseT::BaseT;

  mlir::TypeID getTypeID() const final {
    return mlir::TypeID::get<ConcreteT>();
  }

  /// Provide an implementation of 'classof' that compares the type id of the
  /// provided value with that of the concerete type.
  static bool classof(const BaseT *val) {
    return val->getTypeID() == mlir::TypeID::get<ConcreteT>();
  }

  std::shared_ptr<ConcreteT> as_ptr() { return this->shared_from_this(); }
};

//
// ExprNode tree
//

struct ExprNode {
  std::string name;
  ExprNodePtr parent;

  explicit ExprNode(llvm::StringRef name = "");
  virtual ~ExprNode() = default;
  virtual mlir::TypeID getTypeID() const = 0;
  virtual std::string str() const = 0;
};

struct ExprNodeCast : NodeBase<ExprNodeCast, ExprNode> {
  using Base = NodeBase<ExprNodeCast, ExprNode>;

  util::DataType dtype;
  ExprNodePtr expr;

  explicit ExprNodeCast(util::DataType dtype, const ExprNodePtr &expr);
  std::string str() const final;
};

struct ExprNodeConstSigned : NodeBase<ExprNodeConstSigned, ExprNode> {
  using Base = NodeBase<ExprNodeConstSigned, ExprNode>;

  int64_t value;

  explicit ExprNodeConstSigned(int64_t value);
  std::string str() const final;
};

struct ExprNodeConstUnsigned : NodeBase<ExprNodeConstUnsigned, ExprNode> {
  using Base = NodeBase<ExprNodeConstUnsigned, ExprNode>;

  uint64_t value;

  explicit ExprNodeConstUnsigned(uint64_t value);
  std::string str() const final;
};

struct ExprNodeConstFloat : NodeBase<ExprNodeConstFloat, ExprNode> {
  using Base = NodeBase<ExprNodeConstFloat, ExprNode>;

  double value;

  explicit ExprNodeConstFloat(double value);
  std::string str() const final;
};

struct ExprNodeConstTensor : NodeBase<ExprNodeConstTensor, ExprNode> {
  using Base = NodeBase<ExprNodeConstTensor, ExprNode>;

  util::BufferPtr buffer;

  explicit ExprNodeConstTensor(const util::BufferPtr &buffer,
                               llvm::StringRef name = "");
  std::string str() const final;
};

struct PolyMap {
  ExprNodePtr ref;
  std::vector<PolyNodePtr> idxs;
};

struct Constraint {
  PolyNodePtr lhs;
  DimNodePtr rhs;

  Constraint(const PolyNodePtr &lhs, const DimNodePtr &rhs)
      : lhs(lhs), rhs(rhs) {}
  std::string str() const;
};

struct ExprNodeContraction : NodeBase<ExprNodeContraction, ExprNode> {
  using Base = NodeBase<ExprNodeContraction, ExprNode>;

  util::AggregationKind aggKind;
  util::CombinationKind comboKind;
  std::vector<DimNodePtr> sinkDims;
  std::vector<PolyNodePtr> sinkIdxs;
  std::vector<PolyMap> srcs;
  std::vector<Constraint> constraints;
  ExprNodePtr init;

  explicit ExprNodeContraction(llvm::StringRef name = "");
  std::string str() const final;
};

struct ExprNodeDim : NodeBase<ExprNodeDim, ExprNode> {
  using Base = NodeBase<ExprNodeDim, ExprNode>;

  DimNodePtr dim;

  explicit ExprNodeDim(const DimNodePtr &dim);
  std::string str() const final;
};

struct ExprNodeElement : NodeBase<ExprNodeElement, ExprNode> {
  using Base = NodeBase<ExprNodeElement, ExprNode>;

  ExprNodePtr expr;
  size_t ordinal;

  ExprNodeElement(const ExprNodePtr &expr, size_t ordinal);
  std::string str() const final;
};

struct ExprNodeInput : NodeBase<ExprNodeInput, ExprNode> {
  using Base = NodeBase<ExprNodeInput, ExprNode>;

  util::TensorShape shape;

  explicit ExprNodeInput(const util::TensorShape &shape,
                         llvm::StringRef name = "");
  std::string str() const final;
};

struct ExprNodeIntrinsic : NodeBase<ExprNodeIntrinsic, ExprNode> {
  using Base = NodeBase<ExprNodeIntrinsic, ExprNode>;

  std::string op;
  std::vector<ExprNodePtr> operands;

  ExprNodeIntrinsic(llvm::StringRef op, llvm::ArrayRef<ExprNodePtr> operands);
  std::string str() const final;
};

struct ExprNodeLayer : NodeBase<ExprNodeLayer, ExprNode> {
  using Base = NodeBase<ExprNodeLayer, ExprNode>;

  std::string op;
  std::vector<ExprNodePtr> operands;
  std::vector<ExprNodePtr> results;
  llvm::StringMap<VarNodePtr> attrs;

  ExprNodeLayer(llvm::StringRef op, llvm::ArrayRef<ExprNodePtr> operands,
                const llvm::StringMap<VarNodePtr> &attrs);
  std::string str() const final;
};

struct ExprNodeLoop : NodeBase<ExprNodeLoop, ExprNode> {
  using Base = NodeBase<ExprNodeLoop, ExprNode>;

  std::string op;
  std::vector<ExprNodePtr> operands;
  std::vector<ExprNodePtr> results;

  ExprNodeLoop(llvm::StringRef op, llvm::ArrayRef<ExprNodePtr> operands,
               llvm::ArrayRef<ExprNodePtr> results);
  std::string str() const final;
};

struct ExprNodePragma : NodeBase<ExprNodePragma, ExprNode> {
  using Base = NodeBase<ExprNodePragma, ExprNode>;

  ExprNodePtr expr;
  std::string op;
  llvm::StringMap<VarNodePtr> attrs;

  ExprNodePragma(const ExprNodePtr &expr, llvm::StringRef op,
                 const llvm::StringMap<VarNodePtr> &attrs);
  std::string str() const final;
};

//
// DimNode tree
//

struct DimNode {
  virtual ~DimNode() = default;
  virtual mlir::TypeID getTypeID() const = 0;
  virtual std::string str() const = 0;
};

struct DimNodeLiteral : NodeBase<DimNodeLiteral, DimNode> {
  int64_t value;

  explicit DimNodeLiteral(int64_t value) : value(value) {}
  std::string str() const final;
};

struct DimNodeNone : NodeBase<DimNodeNone, DimNode> {
  std::string str() const final { return "?"; }
};

struct DimNodeOp : NodeBase<DimNodeOp, DimNode> {
  AffineOp op;
  std::vector<DimNodePtr> operands;

  DimNodeOp(AffineOp op, llvm::ArrayRef<DimNodePtr> operands)
      : op(op), operands(operands) {}
  std::string str() const final;
};

struct DimNodeRef : NodeBase<DimNodeRef, DimNode> {
  ExprNodePtr ref;
  size_t dim;

  DimNodeRef(const ExprNodePtr &ref, size_t dim) : ref(ref), dim(dim) {}
  std::string str() const final;
};

//
// PolyNode tree
//

struct PolyNode {
  virtual ~PolyNode() = default;
  virtual mlir::TypeID getTypeID() const = 0;
  virtual std::string str() const = 0;
};

struct PolyNodeDim : NodeBase<PolyNodeDim, PolyNode> {
  DimNodePtr dim;

  explicit PolyNodeDim(const DimNodePtr &dim) : dim(dim) {}
  std::string str() const final;
};

struct PolyNodeIndex : NodeBase<PolyNodeIndex, PolyNode> {
  std::string name;

  explicit PolyNodeIndex(llvm::StringRef name = "") : name(name) {}
  std::string str() const final;
};

struct PolyNodeLiteral : NodeBase<PolyNodeLiteral, PolyNode> {
  int64_t value;

  explicit PolyNodeLiteral(int64_t value) : value(value) {}
  std::string str() const final;
};

struct PolyNodeOp : NodeBase<PolyNodeOp, PolyNode> {
  AffineOp op;
  std::vector<PolyNodePtr> operands;

  PolyNodeOp(AffineOp op, llvm::ArrayRef<PolyNodePtr> operands)
      : op(op), operands(operands) {}
  std::string str() const final;
};

//
// VarNode tree
//

struct VarNode {
  virtual ~VarNode() = default;
  virtual mlir::TypeID getTypeID() const = 0;
  virtual std::string str() const = 0;
};

struct VarNodeDim : NodeBase<VarNodeDim, VarNode> {
  DimNodePtr value;
  explicit VarNodeDim(DimNodePtr value) : value(value) {}
  std::string str() const final { return value->str(); }
};

struct VarNodeExpr : NodeBase<VarNodeExpr, VarNode> {
  ExprNodePtr value;
  explicit VarNodeExpr(ExprNodePtr value) : value(value) {}
  std::string str() const final { return value->str(); }
};

struct VarNodeFloat : NodeBase<VarNodeFloat, VarNode> {
  double value;
  explicit VarNodeFloat(double value) : value(value) {}
  std::string str() const final { return std::to_string(value); };
};

struct VarNodeInt : NodeBase<VarNodeInt, VarNode> {
  int64_t value;
  explicit VarNodeInt(int64_t value) : value(value) {}
  std::string str() const final { return std::to_string(value); };
};

struct VarNodeNone : NodeBase<VarNodeNone, VarNode> {
  std::string str() const final { return "none"; };
};

struct VarNodeString : NodeBase<VarNodeString, VarNode> {
  std::string value;
  explicit VarNodeString(llvm::StringRef value) : value(value) {}
  std::string str() const final { return value; };
};

struct VarNodeTuple : NodeBase<VarNodeTuple, VarNode> {
  std::vector<VarNodePtr> values;
  std::string str() const final;
};

} // namespace pmlc::ast
