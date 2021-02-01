// Copyright 2019 Intel Corporation.

#include "plaidml/edsl/ffi.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <stack>
#include <string>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "plaidml/core/internal.h"
#include "pmlc/util/logging.h"

#include "pmlc/ast/ast.h"
#include "pmlc/ast/builder.h"
#include "pmlc/ast/eval.h"
#include "pmlc/compiler/registry.h"
#include "pmlc/util/enums.h"
#include "pmlc/util/env.h"

using plaidml::core::convertFromDataType;
using plaidml::core::convertIntoDataType;
using plaidml::core::ffi_vector;
using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using pmlc::compiler::Program;
using pmlc::util::AggregationKind;
using pmlc::util::CombinationKind;
using pmlc::util::TensorShape;

namespace ast = pmlc::ast;

namespace {

AggregationKind getAggregationKind(plaidml_agg_op agg_op) {
  switch (agg_op) {
    case PLAIDML_AGG_OP_ASSIGN:
      return AggregationKind::assign;
    case PLAIDML_AGG_OP_MAX:
      return AggregationKind::max;
    case PLAIDML_AGG_OP_MIN:
      return AggregationKind::min;
    case PLAIDML_AGG_OP_PROD:
      return AggregationKind::mul;
    case PLAIDML_AGG_OP_SUM:
      return AggregationKind::add;
    default:
      break;
  }
  throw std::runtime_error("Unsupported agg_op");
}

CombinationKind getCombinationKind(plaidml_combo_op combo_op) {
  switch (combo_op) {
    case PLAIDML_COMBO_OP_ADD:
      return CombinationKind::add;
    case PLAIDML_COMBO_OP_COND:
      return CombinationKind::cond;
    case PLAIDML_COMBO_OP_EQ:
      return CombinationKind::eq;
    case PLAIDML_COMBO_OP_MUL:
      return CombinationKind::mul;
    case PLAIDML_COMBO_OP_NONE:
      return CombinationKind::none;
    default:
      break;
  }
  throw std::runtime_error("Unsupported combo_op");
}

ast::AffineOp getAffineOp(plaidml_int_op op) {
  switch (op) {
    case PLAIDML_INT_OP_ADD:
      return ast::AffineOp::Add;
    case PLAIDML_INT_OP_DIV:
      return ast::AffineOp::Div;
    case PLAIDML_INT_OP_MUL:
      return ast::AffineOp::Mul;
    case PLAIDML_INT_OP_NEG:
      return ast::AffineOp::Neg;
    case PLAIDML_INT_OP_SUB:
      return ast::AffineOp::Sub;
    case PLAIDML_INT_OP_MAX:
      return ast::AffineOp::Max;
    case PLAIDML_INT_OP_MIN:
      return ast::AffineOp::Min;
  }
  throw std::runtime_error("Unknown polynomial op");
}

struct LayerContext {
  static LayerContext* get() {
    thread_local LayerContext context;
    return &context;
  }

  void addNode(const ast::ExprNodePtr& node) {
    if (stack.empty()) {
      return;
    }
    node->parent = stack.front();
  }

  void push(const std::shared_ptr<ast::ExprNodeLayer>& layer) { stack.push_front(layer); }

  void pop() { stack.pop_front(); }

  std::deque<std::shared_ptr<ast::ExprNodeLayer>> stack;
};

}  // namespace

extern "C" {

struct plaidml_expr {
  ast::ExprNodePtr node;
};

struct plaidml_poly_expr {
  ast::PolyNodePtr node;
};

struct plaidml_dim_expr {
  ast::DimNodePtr node;
};

void plaidml_expr_free(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_free> " << expr->node->str());
    delete expr;
  });
}

void* plaidml_expr_ptr(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  return ffi_wrap<void*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_ptr");
    return expr->node.get();
  });
}

plaidml_datatype plaidml_expr_get_dtype(  //
    plaidml_error* err,                   //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {
    IVLOG(3, "plaidml_expr_get_dtype");
    ast::Evaluator evaluator;
    TensorShape shape = evaluator.getShape(expr->node);
    return convertIntoDataType(shape.elementType);
  });
}

size_t plaidml_expr_get_rank(  //
    plaidml_error* err,        //
    plaidml_expr* expr) {
  return ffi_wrap<size_t>(err, 0, [&] {
    ast::Evaluator evaluator;
    TensorShape shape = evaluator.getShape(expr->node);
    return shape.getRank();
  });
}

plaidml_shape* plaidml_expr_get_shape(  //
    plaidml_error* err,                 //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_shape*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_get_shape");
    ast::Evaluator evaluator;
    TensorShape shape = evaluator.getShape(expr->node);
    return new plaidml_shape{shape};
  });
}

void plaidml_expr_bind_dims(  //
    plaidml_error* err,       //
    plaidml_expr* expr,       //
    size_t rank,              //
    plaidml_dim_expr** dims) {
  return ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_bind_dims> " << expr->node->str());
    llvm::SmallVector<ast::DimNodePtr*, 8> into;
    for (size_t i = 0; i < rank; i++) {
      into.push_back(&dims[i]->node);
    }
    ast::Evaluator evaluator;
    evaluator.bindDims(expr->node, into);
  });
}

plaidml_string* plaidml_expr_repr(  //
    plaidml_error* err,             //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    // IVLOG(3, "plaidml_expr_repr");
    return new plaidml_string{expr->node->str()};
  });
}

plaidml_expr* plaidml_expr_dim(  //
    plaidml_error* err,          //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_dim");
    auto node = std::make_shared<ast::ExprNodeDim>(expr->node);
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

plaidml_expr* plaidml_expr_input(  //
    plaidml_error* err,            //
    plaidml_shape* shape,          //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_input: " << shape->shape.str());
    auto node = std::make_shared<ast::ExprNodeInput>(shape->shape, name);
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

plaidml_expr* plaidml_expr_constant(  //
    plaidml_error* err,               //
    plaidml_buffer* buffer,           //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_constant(name: " << name << ")");
    auto node = std::make_shared<ast::ExprNodeConstTensor>(buffer->buffer, name);
    // Constants cannot be added to layers, they must be defined in the global scope because they will eventually become
    // program arguments.
    // However, we need to ensure that all parent layers include the constant node as an operand.
    for (ast::ExprNodePtr layerNode : LayerContext::get()->stack) {
      auto* layer = llvm::dyn_cast<ast::ExprNodeLayer>(layerNode.get());
      layer->operands.push_back(node);
    }
    return new plaidml_expr{node};
  });
}

plaidml_expr* plaidml_expr_clone(  //
    plaidml_error* err,            //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_clone> " << expr->node->str());
    return new plaidml_expr{expr->node};
  });
}

plaidml_dim_expr* plaidml_expr_get_dim(  //
    plaidml_error* err,                  //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_get_dim> " << expr->node->str());
    auto* node = llvm::dyn_cast<ast::ExprNodeDim>(expr->node.get());
    if (!node) {
      throw std::bad_cast();
    }
    return new plaidml_dim_expr{node->dim};
  });
}

plaidml_expr* plaidml_expr_uint(  //
    plaidml_error* err,           //
    uint64_t value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_uint> " << value);
    auto node = std::make_shared<ast::ExprNodeConstUnsigned>(value);
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

plaidml_expr* plaidml_expr_int(  //
    plaidml_error* err,          //
    int64_t value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_int> " << value);
    auto node = std::make_shared<ast::ExprNodeConstSigned>(value);
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

plaidml_expr* plaidml_expr_float(  //
    plaidml_error* err,            //
    double value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_float");
    auto node = std::make_shared<ast::ExprNodeConstFloat>(value);
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

plaidml_expr* plaidml_expr_cast(  //
    plaidml_error* err,           //
    plaidml_expr* expr,           //
    plaidml_datatype dtype) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_cast");
    auto node = std::make_shared<ast::ExprNodeCast>(convertFromDataType(dtype), expr->node);
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

plaidml_expr* plaidml_expr_element(  //
    plaidml_error* err,              //
    plaidml_expr* expr,              //
    size_t ordinal) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_element");
    auto node = std::make_shared<ast::ExprNodeElement>(expr->node, ordinal);
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

plaidml_expr* plaidml_expr_pragma(  //
    plaidml_error* err,             //
    plaidml_expr* expr,             //
    const char* op,                 //
    size_t nattrs,                  //
    plaidml_attr** raw_attrs) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_pragma");
    llvm::StringMap<ast::VarNodePtr> attrs;
    for (size_t i = 0; i < nattrs; i++) {
      plaidml_attr* attr = raw_attrs[i];
      attrs[attr->key] = attr->value->node;
    }
    auto node = std::make_shared<ast::ExprNodePragma>(expr->node, op, attrs);
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

plaidml_expr* plaidml_expr_intrinsic(  //
    plaidml_error* err,                //
    const char* fn,                    //
    size_t nargs,                      //
    plaidml_expr** args) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_intrinsic: " << fn);
    std::vector<ast::ExprNodePtr> operands(nargs);
    for (size_t i = 0; i < nargs; i++) {
      operands[i] = args[i]->node;
    }
    auto node = std::make_shared<ast::ExprNodeIntrinsic>(fn, operands);
    ast::Evaluator evaluator;
    evaluator.verify(node);
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

plaidml_expr* plaidml_expr_contraction(  //
    plaidml_error* err,                  //
    plaidml_agg_op agg_op,               //
    plaidml_combo_op combo_op,           //
    size_t rank,                         //
    plaidml_poly_expr** idxs,            //
    plaidml_dim_expr** dims,             //
    plaidml_expr* init,                  //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_contraction");
    auto node = std::make_shared<ast::ExprNodeContraction>(name);
    node->aggKind = getAggregationKind(agg_op);
    node->comboKind = getCombinationKind(combo_op);
    for (size_t i = 0; i < rank; i++) {
      node->sinkDims.push_back(dims[i]->node);
      node->sinkIdxs.push_back(idxs[i]->node);
    }
    if (init) {
      node->init = init->node;
    }
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

void plaidml_contraction_add_operand(  //
    plaidml_error* err,                //
    plaidml_expr* expr,                //
    plaidml_expr* ref,                 //
    size_t rank,                       //
    plaidml_poly_expr** idxs) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_contraction_add_operand");
    auto* node = llvm::dyn_cast<ast::ExprNodeContraction>(expr->node.get());
    if (!node) {
      throw std::bad_cast();
    }
    ast::PolyMap map;
    map.ref = ref->node;
    for (size_t i = 0; i < rank; i++) {
      map.idxs.push_back(idxs[i]->node);
    }
    node->srcs.emplace_back(map);
  });
}

void plaidml_contraction_add_constraint(  //
    plaidml_error* err,                   //
    plaidml_expr* expr,                   //
    plaidml_poly_expr* lhs,               //
    plaidml_dim_expr* rhs) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_contraction_add_constraint");
    auto* node = llvm::dyn_cast<ast::ExprNodeContraction>(expr->node.get());
    if (!node) {
      throw std::bad_cast();
    }
    node->constraints.emplace_back(lhs->node, rhs->node);
  });
}

void plaidml_contraction_build(  //
    plaidml_error* err,          //
    plaidml_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_contraction_build");
    ast::Evaluator evaluator;
    evaluator.verify(expr->node);
  });
}

void plaidml_poly_expr_free(plaidml_error* err, plaidml_poly_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_poly_expr_free> " << expr->node->str());
    delete expr;
  });
}

plaidml_string* plaidml_poly_expr_repr(  //
    plaidml_error* err,                  //
    plaidml_poly_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_repr");
    return new plaidml_string{expr->node->str()};
  });
}

plaidml_poly_expr* plaidml_poly_expr_dim(  //
    plaidml_error* err,                    //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_dim");
    return new plaidml_poly_expr{std::make_shared<ast::PolyNodeDim>(expr->node)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_index(  //
    plaidml_error* err,                      //
    const char* name) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_index");
    return new plaidml_poly_expr{std::make_shared<ast::PolyNodeIndex>(name)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_literal(  //
    plaidml_error* err,                        //
    int64_t value) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_literal> " << value);
    return new plaidml_poly_expr{std::make_shared<ast::PolyNodeLiteral>(value)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_op(  //
    plaidml_error* err,                   //
    plaidml_int_op op,                    //
    size_t nargs,                         //
    plaidml_poly_expr** args) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_op> " << op);
    std::vector<ast::PolyNodePtr> operands(nargs);
    for (size_t i = 0; i < nargs; i++) {
      operands[i] = args[i]->node;
    }
    return new plaidml_poly_expr{std::make_shared<ast::PolyNodeOp>(getAffineOp(op), operands)};
  });
}

void plaidml_dim_expr_free(  //
    plaidml_error* err,      //
    plaidml_dim_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_dim_expr_free> " << expr->node->str());
    delete expr;
  });
}

plaidml_string* plaidml_dim_expr_repr(  //
    plaidml_error* err,                 //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_repr");
    return new plaidml_string{expr->node->str()};
  });
}

plaidml_dim_expr* plaidml_dim_expr_none(  //
    plaidml_error* err                    //
) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_none");
    return new plaidml_dim_expr{std::make_shared<ast::DimNodeNone>()};
  });
}

plaidml_dim_expr* plaidml_dim_expr_int(  //
    plaidml_error* err,                  //
    int64_t value) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_int> " << value);
    return new plaidml_dim_expr{std::make_shared<ast::DimNodeLiteral>(value)};
  });
}

plaidml_dim_expr* plaidml_dim_expr_op(  //
    plaidml_error* err,                 //
    plaidml_int_op op,                  //
    size_t nargs,                       //
    plaidml_dim_expr** args) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_op> " << op);
    std::vector<ast::DimNodePtr> operands(nargs);
    for (size_t i = 0; i < nargs; i++) {
      operands[i] = args[i]->node;
    }
    return new plaidml_dim_expr{std::make_shared<ast::DimNodeOp>(getAffineOp(op), operands)};
  });
}

void plaidml_tuple_free(  //
    plaidml_error* err,   //
    plaidml_tuple* tuple) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_tuple_free");
    for (size_t i = 0; i < tuple->size; i++) {
      delete tuple->elts[i];
    }
    delete[] tuple->elts;
    delete tuple;
  });
}

void plaidml_value_free(  //
    plaidml_error* err,   //
    plaidml_value* value) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_value_free: " << value->node->str());
    delete value;
  });
}

plaidml_value* plaidml_value_clone(  //
    plaidml_error* err,              //
    plaidml_value* value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_clone");
    return new plaidml_value{*value};
  });
}

plaidml_value_kind plaidml_value_get_kind(  //
    plaidml_error* err,                     //
    plaidml_value* value) {
  return ffi_wrap<plaidml_value_kind>(err, PLAIDML_VALUE_NONE, [&] {
    IVLOG(3, "plaidml_value_get_kind");
    return llvm::TypeSwitch<ast::VarNode*, plaidml_value_kind>(value->node.get())
        .Case<ast::VarNodeDim>([](auto*) { return PLAIDML_VALUE_DIM; })
        .Case<ast::VarNodeExpr>([](auto*) { return PLAIDML_VALUE_EXPR; })
        .Case<ast::VarNodeFloat>([](auto*) { return PLAIDML_VALUE_FLOAT; })
        .Case<ast::VarNodeInt>([](auto*) { return PLAIDML_VALUE_INT; })
        .Case<ast::VarNodeNone>([](auto*) { return PLAIDML_VALUE_NONE; })
        .Case<ast::VarNodeString>([](auto*) { return PLAIDML_VALUE_STR; })
        .Case<ast::VarNodeTuple>([](auto*) { return PLAIDML_VALUE_TUPLE; })
        .Default([](auto*) -> plaidml_value_kind { throw std::bad_cast(); });
  });
}

plaidml_value* plaidml_value_none(  //
    plaidml_error* err              //
) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_none");
    return new plaidml_value{std::make_shared<ast::VarNodeNone>()};
  });
}

plaidml_value* plaidml_value_int(  //
    plaidml_error* err,            //
    int64_t value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_int> " << value);
    return new plaidml_value{std::make_shared<ast::VarNodeInt>(value)};
  });
}

plaidml_value* plaidml_value_dim(  //
    plaidml_error* err,            //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_dim");
    if (!expr) {
      throw std::runtime_error("plaidml_value_dim requires non-null expr");
    }
    return new plaidml_value{std::make_shared<ast::VarNodeDim>(expr->node)};
  });
}

plaidml_value* plaidml_value_expr(  //
    plaidml_error* err,             //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_expr");
    if (!expr) {
      throw std::runtime_error("plaidml_value_expr requires non-null expr");
    }
    return new plaidml_value{std::make_shared<ast::VarNodeExpr>(expr->node)};
  });
}

plaidml_value* plaidml_value_float(  //
    plaidml_error* err,              //
    double value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_float");
    return new plaidml_value{std::make_shared<ast::VarNodeFloat>(value)};
  });
}

plaidml_value* plaidml_value_str(  //
    plaidml_error* err,            //
    const char* value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_str> " << value);
    return new plaidml_value{std::make_shared<ast::VarNodeString>(value)};
  });
}

plaidml_value* plaidml_value_tuple(  //
    plaidml_error* err,              //
    size_t size,                     //
    plaidml_value** elts) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_tuple: " << size);
    auto tuple = std::make_shared<ast::VarNodeTuple>();
    for (size_t i = 0; i < size; i++) {
      tuple->values.push_back(elts[i]->node);
    }
    return new plaidml_value{tuple};
  });
}

plaidml_dim_expr* plaidml_value_dim_get(  //
    plaidml_error* err,                   //
    plaidml_value* value) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_dim_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeDim>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    return new plaidml_dim_expr{node->value};
  });
}

plaidml_expr* plaidml_value_expr_get(  //
    plaidml_error* err,                //
    plaidml_value* value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_expr_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeExpr>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    return new plaidml_expr{node->value};
  });
}

double plaidml_value_float_get(  //
    plaidml_error* err,          //
    plaidml_value* value) {
  return ffi_wrap<double>(err, 0, [&] {
    IVLOG(3, "plaidml_value_float_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeFloat>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    return node->value;
  });
}

int64_t plaidml_value_int_get(  //
    plaidml_error* err,         //
    plaidml_value* value) {
  return ffi_wrap<int64_t>(err, 0, [&] {
    IVLOG(3, "plaidml_value_int_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeInt>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    return node->value;
  });
}

plaidml_tuple* plaidml_value_tuple_get(  //
    plaidml_error* err,                  //
    plaidml_value* value) {
  return ffi_wrap<plaidml_tuple*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_tuple_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeTuple>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    auto size = node->values.size();
    auto elts = new plaidml_value*[size];
    for (size_t i = 0; i < size; i++) {
      elts[i] = new plaidml_value{node->values[i]};
    }
    return new plaidml_tuple{size, elts};
  });
}

plaidml_string* plaidml_value_str_get(  //
    plaidml_error* err,                 //
    plaidml_value* value) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_str_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeString>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    return new plaidml_string{node->value};
  });
}

plaidml_string* plaidml_value_repr(  //
    plaidml_error* err,              //
    plaidml_value* value) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_repr");
    return new plaidml_string{value->node->str()};
  });
}

plaidml_strings* plaidml_targets_get(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_strings*>(err, nullptr, [&] {
    const auto& targets = pmlc::compiler::listTargets();
    auto strs = new plaidml_string*[targets.size()];
    for (unsigned i = 0; i < targets.size(); i++) {
      strs[i] = new plaidml_string{targets[i].str()};
    }
    return new plaidml_strings{targets.size(), strs};
  });
}

plaidml_program* plaidml_build(  //
    plaidml_error* err,          //
    const char* name,            //
    size_t ninputs,              //
    plaidml_expr** inputs,       //
    plaidml_shape** shapes,      //
    size_t noutputs,             //
    plaidml_expr** outputs) {
  return ffi_wrap<plaidml_program*>(err, nullptr, [&] {
    ast::ProgramArguments args;
    for (size_t i = 0; i < ninputs; i++) {
      args.inputs.push_back(inputs[i]->node);
      if (shapes) {
        args.shapes.push_back(shapes[i]->shape);
      }
    }
    for (size_t i = 0; i < noutputs; i++) {
      args.outputs.push_back(outputs[i]->node);
    }
    return new plaidml_program{ast::buildProgram(name, args)};
  });
}

void plaidml_exprs_free(  //
    plaidml_error* err,   //
    plaidml_exprs* exprs) {
  ffi_wrap_void(err, [&] {
    delete[] exprs->elts;
    delete exprs;
  });
}

plaidml_expr* plaidml_expr_layer_begin(  //
    plaidml_error* err,                  //
    const char* op,                      //
    size_t ninputs,                      //
    plaidml_expr** inputs,               //
    size_t nattrs,                       //
    plaidml_attr** raw_attrs) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_layer_begin(" << op << ", inputs: " << ninputs << ")");
    std::vector<ast::ExprNodePtr> operands(ninputs);
    for (size_t i = 0; i < ninputs; i++) {
      operands[i] = inputs[i]->node;
    }
    llvm::StringMap<ast::VarNodePtr> attrs;
    for (size_t i = 0; i < nattrs; i++) {
      plaidml_attr* attr = raw_attrs[i];
      attrs[attr->key] = attr->value->node;
    }
    auto node = std::make_shared<ast::ExprNodeLayer>(op, operands, attrs);
    LayerContext::get()->addNode(node);
    LayerContext::get()->push(node);
    return new plaidml_expr{node};
  });
}

plaidml_exprs* plaidml_expr_layer_end(  //
    plaidml_error* err,                 //
    plaidml_expr* expr,                 //
    size_t noutputs,                    //
    plaidml_expr** outputs) {
  return ffi_wrap<plaidml_exprs*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_layer_end");
    auto node = std::dynamic_pointer_cast<ast::ExprNodeLayer>(expr->node);
    if (!node) {
      throw std::bad_cast();
    }
    LayerContext::get()->pop();
    std::vector<ast::ExprNodePtr> outerResults;
    outerResults.reserve(noutputs);
    node->results.clear();
    node->results.reserve(noutputs);
    for (size_t i = 0; i < noutputs; i++) {
      node->results.push_back(outputs[i]->node);
      outerResults.push_back(std::make_shared<ast::ExprNodeElement>(node, i));
    }
    return ffi_vector<plaidml_exprs, plaidml_expr>(outerResults);
  });
}

plaidml_expr* plaidml_expr_loop(  //
    plaidml_error* err,            //
    const char* op,                //
    size_t nindex,                 //
    plaidml_expr** indexs,         //
    size_t ninputs,                //
    plaidml_expr** inputs,         //
    size_t noutputs,               //
    plaidml_expr** outputs) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_loop (" << op << ", inputs: " << ninputs << ")");
    std::vector<ast::ExprNodePtr> operands(ninputs);
    for (size_t i = 0; i < ninputs; i++) {
      operands[i] = inputs[i]->node;
    }
    std::vector<ast::ExprNodePtr> results(noutputs);
    for (size_t i = 0; i < noutputs; i++) {
      results[i] = outputs[i]->node;
    }
    std::vector<ast::ExprNodePtr> loopIndex(nindex);
    for (size_t i = 0; i < nindex; i++) {
      loopIndex[i] = indexs[i]->node;
    }
    auto node = std::make_shared<ast::ExprNodeLoop>(op, loopIndex, operands, results);
    ast::Evaluator evaluator;
    evaluator.verify(node);
    LayerContext::get()->addNode(node);
    return new plaidml_expr{node};
  });
}

}  // extern "C"
