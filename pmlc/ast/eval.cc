// Copyright 2020 Intel Corporation

#include "pmlc/ast/eval.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "pmlc/ast/ast_ops.h"
#include "pmlc/util/logging.h"

namespace pmlc::ast {

using util::DataType;
using util::TensorShape;
using util::TensorShapes;

static DataType inferElementType(Evaluator *evaluator,
                                 util::CombinationKind combo,
                                 llvm::ArrayRef<PolyMap> srcs) {
  if (combo == util::CombinationKind::eq) {
    return DataType::i1;
  }
  if (combo == util::CombinationKind::cond) {
    return evaluator->getShape(srcs[2].ref).elementType;
  }
  llvm::SmallVector<TensorShape, 3> shapes;
  for (const PolyMap &src : srcs) {
    shapes.push_back(evaluator->getShape(src.ref));
  }
  return ast::inferElementType(shapes);
}

static int64_t getTypeScore(DataType type) {
  return static_cast<int64_t>(type);
}

static DataType promoteTypes(DataType lhs, DataType rhs) {
  return getTypeScore(lhs) > getTypeScore(rhs) ? lhs : rhs;
}

static bool mergeShapes(TensorShape *into, const TensorShape &from,
                        DataType dtype) {
  // To compute the resulting broadcasted shape, we compare operand shapes
  // element-wise: starting with the trailing dimensions, and working our
  // way backward. Two dimensions are compatible when
  //   1. they are equal, or
  //   2. one of them is 1
  // The result shape has the maximum among the two inputs at every
  // dimension index.
  std::vector<int64_t> resultShape;
  const std::vector<int64_t> &shape1 = into->sizes;
  const std::vector<int64_t> &shape2 = from.sizes;
  IVLOG(6, "  Checking compatibility between " << shape1 << " and " << shape2);
  if (shape1.size() > shape2.size()) {
    std::copy(shape1.begin(), shape1.end(), std::back_inserter(resultShape));
  } else {
    std::copy(shape2.begin(), shape2.end(), std::back_inserter(resultShape));
  }

  auto i1 = shape1.rbegin(), e1 = shape1.rend();
  auto i2 = shape2.rbegin(), e2 = shape2.rend();
  auto iR = resultShape.rbegin();

  // Check each dimension is consistent.
  for (; i1 != e1 && i2 != e2; ++i1, ++i2, ++iR) {
    if (*i1 == 0 || *i2 == 0) {
      // One or both dimensions is unknown. Follow TensorFlow behavior:
      // - If either dimension is greater than 1, we assume that the program is
      //   correct, and the other dimension will be broadcast to match it.
      // - If either dimension is 1, the other dimension is the output.
      if (*i1 > 1) {
        *iR = *i1;
      } else if (*i2 > 1) {
        *iR = *i2;
      } else if (*i1 == 1) {
        *iR = *i2;
      } else if (*i2 == 1) {
        *iR = *i1;
      } else {
        *iR = 0;
      }
    } else {
      if (*i1 == *i2 || *i2 == 1) {
        *iR = *i1;
      } else if (*i1 == 1) {
        *iR = *i2;
      } else {
        // This dimension of the two operand types is incompatible.
        return false;
      }
    }
  }

  if (dtype == DataType::invalid) {
    dtype = promoteTypes(into->elementType, from.elementType);
  }
  *into = TensorShape{dtype, resultShape};
  IVLOG(6, "  Resulting shape: " << into->str());
  return true;
}

DataType inferElementType(llvm::ArrayRef<TensorShape> shapes) {
  DataType ret = DataType::invalid;
  for (const TensorShape &shape : shapes) {
    ret = promoteTypes(ret, shape.elementType);
  }
  return ret;
}

TensorShape inferShape(llvm::ArrayRef<TensorShape> operands,
                       DataType override) {
  TensorShape ret = operands.front();
  if (override != DataType::invalid) {
    ret.elementType = override;
  }
  for (const TensorShape &operand : operands.drop_front()) {
    if (!mergeShapes(&ret, operand, override)) {
      std::stringstream ss;
      ss << "Incompatible types: (";
      for (size_t i = 0; i < operands.size(); i++) {
        if (i) {
          ss << ", ";
        }
        ss << operands[i].str();
      }
      ss << ")";
      throw std::runtime_error(ss.str());
    }
  }
  return ret;
}

bool isAmbiguousDataType(DataType dtype) {
  return dtype == DataType::fx || dtype == DataType::six ||
         dtype == DataType::uix;
}

//
// Evaluator
//

void Evaluator::verify(const ExprNodePtr &node) { (void)getShapes(node); }

int64_t Evaluator::evaluate(const DimNodePtr &node) {
  return evaluate(node.get());
}

int64_t Evaluator::evaluate(const DimNode *node) {
  return llvm::TypeSwitch<const DimNode *, int64_t>(node)
      .Case<DimNodeLiteral>([&](const auto *node) { return node->value; })
      .Case<DimNodeNone>([&](const auto *node) -> int64_t {
        throw std::runtime_error("DimNodeNone cannot be evaluated");
      })
      .Case<DimNodeOp>([&](const auto *node) {
        if (node->op == AffineOp::Neg) {
          if (node->operands.size() != 1) {
            throw std::runtime_error("AffineOp::Neg expected single operand");
          }
          return -evaluate(node->operands[0]);
        }
        if (node->operands.size() != 2) {
          throw std::runtime_error("Binary DimNodeOp expected 2 operands");
        }
        int64_t lhs = evaluate(node->operands[0]);
        int64_t rhs = evaluate(node->operands[1]);
        switch (node->op) {
        case AffineOp::Add:
          return lhs + rhs;
        case AffineOp::Sub:
          return lhs - rhs;
        case AffineOp::Mul:
          return lhs * rhs;
        case AffineOp::Div:
          return lhs / rhs;
        case AffineOp::Max:
          return std::max(lhs, rhs);
        case AffineOp::Min:
          return std::min(lhs, rhs);
        default:
          throw std::runtime_error("Invalid AffineOp");
        }
      })
      .Case<DimNodeRef>([&](const auto *node) {
        TensorShape shape = getShape(node->ref);
        if (shape.getRank() < node->dim) {
          throw std::runtime_error("DimNodeRef out of bounds");
        }
        return shape.sizes[node->dim];
      });
}

TensorShape Evaluator::getShape(const ExprNodePtr &node) {
  return getShape(node.get());
}

TensorShape Evaluator::getShape(const ExprNodePtr &node, size_t ordinal) {
  return getShape(node.get(), ordinal);
}

llvm::ArrayRef<TensorShape> Evaluator::getShapes(const ExprNodePtr &node) {
  return getShapes(node.get());
}

TensorShape Evaluator::getShape(const ExprNode *node) {
  auto shapes = getShapes(node);
  if (shapes.size() != 1) {
    throw std::runtime_error(
        "Missing element needed to resolve multlple results.");
  }
  return shapes[0];
}

TensorShape Evaluator::getShape(const ExprNode *node, size_t ordinal) {
  auto shapes = getShapes(node);
  if (shapes.size() < ordinal) {
    throw std::runtime_error("Out of bounds element ordinal");
  }
  return shapes[ordinal];
}

llvm::ArrayRef<TensorShape> Evaluator::getShapes(const ExprNode *node) {
  auto it = shapesCache.find(node);
  if (it == shapesCache.end()) {
    std::tie(it, std::ignore) = shapesCache.emplace(node, computeShapes(node));
  }
  return it->second;
}

TensorShapes Evaluator::computeShapes(const ExprNode *node) {
  TensorShapes shapes =
      llvm::TypeSwitch<const ExprNode *, TensorShapes>(node)
          .Case<ExprNodeCast>([&](const auto *node) {
            TensorShape shape = getShape(node->expr);
            shape.elementType = node->dtype;
            return TensorShapes{shape};
          })
          .Case<ExprNodeConstSigned>([&](const auto *node) {
            return TensorShapes{TensorShape{DataType::six}};
          })
          .Case<ExprNodeConstUnsigned>([&](const auto *node) {
            return TensorShapes{TensorShape{DataType::uix}};
          })
          .Case<ExprNodeConstFloat>([&](const auto *node) {
            return TensorShapes{TensorShape{DataType::fx}};
          })
          .Case<ExprNodeConstTensor>([&](const auto *node) {
            return TensorShapes{node->buffer->shape()};
          })
          .Case<ExprNodeContraction>([&](const auto *node) {
            DataType elementType =
                inferElementType(this, node->comboKind, node->srcs);
            if (isAmbiguousDataType(elementType)) {
              throw std::runtime_error(
                  "'contraction' operand data type is ambiguous, use a cast");
            }
            TensorShape shape{elementType};
            for (const DimNodePtr &dim : node->sinkDims) {
              shape.sizes.push_back(evaluate(dim));
            }
            return TensorShapes{shape};
          })
          .Case<ExprNodeDim>([&](const auto *node) {
            return TensorShapes{TensorShape(DataType::six)};
          })
          .Case<ExprNodeElement>([&](const auto *node) {
            return TensorShapes{getShape(node->expr, node->ordinal)};
          })
          .Case<ExprNodeLayer>([&](const auto &node) {
            TensorShapes ret;
            for (const ExprNodePtr &result : node->results) {
              ret.push_back(getShape(result.get()));
            }
            return ret;
          })
          .Case<ExprNodeLoop>([&](const auto &node) {
            TensorShapes ret;
            for (const ExprNodePtr &result : node->results) {
              ret.push_back(getShape(result.get()));
            }
            return ret;
          })
          .Case<ExprNodeInput>(
              [&](const auto *node) { return TensorShapes{node->shape}; })
          .Case<ExprNodeIntrinsic>([&](const auto *node) {
            llvm::SmallVector<TensorShape, 8> shapes;
            for (const ExprNodePtr &operand : node->operands) {
              shapes.emplace_back(getShape(operand));
            }
            auto intrinsic = IntrinsicRegistry::Instance()->resolve(node->op);
            if (intrinsic) {
              return intrinsic->getShapes(this, node->operands, shapes);
            }
            return TensorShapes{inferShape(shapes)};
          })
          .Case<ExprNodePragma>([&](const auto *node) {
            auto shapes = getShapes(node->expr);
            return TensorShapes(shapes.begin(), shapes.end());
          });
  return shapes;
}

void Evaluator::bindDims(const ExprNodePtr &expr,
                         llvm::ArrayRef<DimNodePtr *> into) {
  TensorShape shape = getShape(expr);
  if (into.size() != shape.getRank()) {
    throw std::runtime_error("Rank mismatch in bind_dims");
  }
  bool failed = false;
  for (size_t i = 0; i < into.size(); i++) {
    DimNodePtr *ptr = into[i];
    int64_t dim = shape.sizes[i];
    llvm::TypeSwitch<const DimNode *>(ptr->get())
        .Case<DimNodeNone>([&](auto *node) {
          if (dim) {
            *ptr = std::make_shared<ast::DimNodeLiteral>(dim);
          } else {
            *ptr = std::make_shared<ast::DimNodeRef>(expr, i);
          }
        })
        .Case<DimNodeRef>([&](auto *node) {
          if (dim) {
            *ptr = std::make_shared<ast::DimNodeLiteral>(dim);
          }
        })
        .Case<DimNodeLiteral>([&](auto *node) {
          if (dim && dim != node->value) {
            failed = true;
          }
        })
        .Default([&](auto *node) {
          throw std::runtime_error("Can only bind dims to an unbound or "
                                   "previously bound TensorDim.");
        });
  }
  if (failed) {
    std::stringstream ss1;
    for (auto item : llvm::enumerate(shape.sizes)) {
      if (item.index()) {
        ss1 << ", ";
      }
      if (item.value()) {
        ss1 << item.value();
      } else {
        ss1 << '?';
      }
    }
    std::stringstream ss2;
    for (auto item : llvm::enumerate(into)) {
      if (item.index()) {
        ss2 << ", ";
      }
      auto value = evaluate(item.value()->get());
      if (value) {
        ss2 << value;
      } else {
        ss2 << '?';
      }
    }
    throw std::runtime_error(
        llvm::formatv("bind dims mismatch: shape=[{0}] != dims=[{1}]",
                      ss1.str(), ss2.str())
            .str());
  }
}

} // namespace pmlc::ast
