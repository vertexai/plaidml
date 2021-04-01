// Copyright 2020 Intel Corporation

#include "pmlc/ast/builder.h"

#include <limits>
#include <mutex>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "pmlc/ast/ast.h"
#include "pmlc/ast/ast_ops.h"
#include "pmlc/ast/eval.h"
#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/ir/util.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::ast {

using compiler::Program;
using pmlc::util::DataType;
using pmlc::util::TensorShape;
namespace layer = pmlc::dialect::layer;
namespace tile = pmlc::dialect::tile;

namespace {

static constexpr const char *kEntrypoint = "main";

static StringRef getDiagKindStr(DiagnosticSeverity kind) {
  switch (kind) {
  case DiagnosticSeverity::Note:
    return "note";
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  case DiagnosticSeverity::Remark:
    return "remark";
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}

class OpBuilder : public mlir::OpBuilder {
public:
  using mlir::OpBuilder::OpBuilder;

  Type getElementType(DataType dtype) {
    switch (dtype) {
    case DataType::i1:
      return getI1Type();
    case DataType::si8:
      return getIntegerType(8, /*isSigned=*/true);
    case DataType::ui8:
      return getIntegerType(8, /*isSigned=*/false);
    case DataType::si16:
      return getIntegerType(16, /*isSigned=*/true);
    case DataType::ui16:
      return getIntegerType(16, /*isSigned=*/false);
    case DataType::si32:
      return getIntegerType(32, /*isSigned=*/true);
    case DataType::ui32:
      return getIntegerType(32, /*isSigned=*/false);
    case DataType::si64:
    case DataType::six:
      return getIntegerType(64, /*isSigned=*/true);
    case DataType::ui64:
    case DataType::uix:
      return getIntegerType(64, /*isSigned=*/false);
    case DataType::f16:
      return getF16Type();
    case DataType::f32:
      return getF32Type();
    case DataType::f64:
    case DataType::fx:
      return getF64Type();
    default:
      throw std::runtime_error(llvm::formatv("OpBuilder> Invalid DataType: {0}",
                                             util::stringifyDataType(dtype)));
      break;
    }
  }

  RankedTensorType getRankedTensorType(const TensorShape &shape) {
    Type elementType = getElementType(shape.elementType);
    return RankedTensorType::get(shape.sizes, elementType);
  }

  tile::APFloatType getAPFloatType() { return tile::APFloatType::get(context); }

  tile::APSignedIntegerType getAPSignedIntegerType() {
    return tile::APSignedIntegerType::get(context);
  }

  tile::APUnsignedIntegerType getAPUnsignedIntegerType() {
    return tile::APUnsignedIntegerType::get(context);
  }

  Value lookupNode(const ExprNodePtr &node) {
    auto it = exprMap.find(node.get());
    if (it == exprMap.end()) {
      if (isa<ExprNodeInput>(node.get())) {
        // NOTE: this can happen if the user forgets to add an input to the
        // edsl::Program constructor.
        throw std::runtime_error(llvm::formatv(
            "Missing placeholder during program build: {0}", node->str()));
      }
      throw std::runtime_error(
          llvm::formatv("ExprNode not found: {0}", node->str()));
    }
    return it->second;
  }

  Value lookupElement(const ExprNodePtr &node, size_t ordinal) {
    auto it = exprTuples.find(node.get());
    if (it == exprTuples.end()) {
      throw std::runtime_error(
          llvm::formatv("Element node not found: {0}", node->str()));
    }
    if (ordinal >= it->second.size()) {
      throw std::runtime_error("Out of range ordinal for element node");
    }
    return it->second[ordinal];
  }

  void addNode(const ExprNodePtr &node, Value value) {
    exprMap[node.get()] = value;
  }

  Attribute getAttribute(const VarNodePtr &node) {
    return TypeSwitch<VarNode *, Attribute>(node.get())
        .Case<VarNodeFloat>(
            [&](VarNodeFloat *node) { return getF64FloatAttr(node->value); })
        .Case<VarNodeInt>(
            [&](VarNodeInt *node) { return getI64IntegerAttr(node->value); })
        .Case<VarNodeString>(
            [&](VarNodeString *node) { return getStringAttr(node->value); })
        .Case<VarNodeTuple>([&](VarNodeTuple *node) {
          SmallVector<Attribute, 8> attrs;
          for (const VarNodePtr &value : node->values) {
            attrs.push_back(getAttribute(value));
          }
          return getArrayAttr(attrs);
        })
        .Default([](VarNode *node) -> Attribute {
          throw std::runtime_error(
              llvm::formatv("Unsupported VarNode: {0}", node->str()));
        });
  }

  DenseMap<const ExprNode *, Value> exprMap;
  DenseMap<const ExprNode *, SmallVector<Value, 4>> exprTuples;
};

class AstTraversal {
private:
  struct Entry {
    ExprNodePtr expr;
    bool post;
  };

  std::stack<Entry> stack;
  std::vector<ExprNodePtr> flat;
  std::unordered_set<const ExprNode *> visited;

public:
  explicit AstTraversal(const std::vector<ExprNodePtr> &outputs,
                        const ExprNodeLayer *layer = nullptr) {
    for (const ExprNodePtr &expr : llvm::reverse(outputs)) {
      push(expr);
    }
    while (stack.size()) {
      Entry entry = stack.top();
      stack.pop();
      if (entry.post) {
        flat.push_back(entry.expr);
      } else if (!visited.count(entry.expr.get())) {
        visited.emplace(entry.expr.get());
        if (layer && entry.expr->parent.get() != layer) {
          if (std::find(layer->operands.begin(), layer->operands.end(),
                        entry.expr) == layer->operands.end()) {
            throw std::runtime_error(llvm::formatv(
                "Implicit operand detected within layer body: {0}",
                entry.expr->str()));
          }
        } else {
          push(entry.expr, /*post=*/true);
          visit(entry.expr.get());
        }
      }
    }
  }

  const std::vector<ExprNodePtr> &getFlat() const { return flat; }

private:
  void visit(ExprNode *node) {
    TypeSwitch<ExprNode *>(node)
        .Case<ExprNodeCast>([&](ExprNodeCast *expr) { push(expr->expr); })
        .Case<ExprNodeContraction>([&](ExprNodeContraction *expr) {
          // Push inputs from right-to-left so they eventually get processed in
          // left-to-right order.
          for (const PolyMap &map : llvm::reverse(expr->srcs)) {
            push(map.ref);
          }
          if (expr->init) {
            push(expr->init);
          }
        })
        .Case<ExprNodeElement>([&](ExprNodeElement *expr) { push(expr->expr); })
        .Case<ExprNodeLayer>([&](ExprNodeLayer *expr) {
          for (const ExprNodePtr &node : llvm::reverse(expr->results)) {
            push(node);
          }
          for (const ExprNodePtr &node : llvm::reverse(expr->operands)) {
            push(node);
          }
        })
        .Case<ExprNodeIntrinsic>([&](ExprNodeIntrinsic *expr) {
          // Push operands from right-to-left so they eventually get processed
          // in left-to-right order.
          for (const ExprNodePtr &node : llvm::reverse(expr->operands)) {
            push(node);
          }
        })
        .Case<ExprNodePragma>([&](ExprNodePragma *expr) { push(expr->expr); });
  }

private:
  void push(const ExprNodePtr &expr, bool post = false) {
    if (!expr) {
      throw std::runtime_error("Invalid expression in AstTraversal::push");
    }
    IVLOG(4, "AstTraversal::push> " << expr->str());
    stack.emplace(Entry{expr, post});
  }
};

static std::vector<ExprNodePtr> getFlatAst(const ProgramArguments &args) {
  AstTraversal traversal(args.outputs);
  return traversal.getFlat();
}

template <typename T, typename R = void>
class PolyVisitor {
protected:
  PolyVisitor() = default;

public:
  R visit(PolyNode *node) {
    return TypeSwitch<PolyNode *, R>(node)
        .template Case<PolyNodeDim>([this](auto *node) {
          return static_cast<T *>(this)->visitDim(node);
        })
        .template Case<PolyNodeIndex>([this](auto *node) {
          return static_cast<T *>(this)->visitIndex(node);
        })
        .template Case<PolyNodeLiteral>([this](auto *node) {
          return static_cast<T *>(this)->visitLiteral(node);
        })
        .template Case<PolyNodeOp>([this](auto *node) {
          for (const PolyNodePtr &operand : node->operands) {
            visit(operand.get());
          }
          return static_cast<T *>(this)->visitOp(node);
        })
        .Default([](PolyNode *) {
          llvm_unreachable("Invalid PolyNode");
          return R();
        });
  }

  void visitDim(const PolyNodeDim *node) {}
  void visitIndex(const PolyNodeIndex *node) {}
  void visitLiteral(const PolyNodeLiteral *node) {}
  void visitOp(const PolyNodeOp *node) {}
};

struct IndexCollector : PolyVisitor<IndexCollector> {
  llvm::SetVector<const PolyNodeIndex *> idxs;
  void visitIndex(const PolyNodeIndex *node) { idxs.insert(node); }
};

struct ContractionBuilder : PolyVisitor<ContractionBuilder, AffineExpr> {
  ContractionBuilder(OpBuilder &builder, Evaluator *evaluator,
                     ExprNodeContraction *node)
      : builder(builder), context(builder.getContext()), evaluator(evaluator),
        node(node),
        resultType(builder.getRankedTensorType(evaluator->getShape(node))) {
    // First collect all idxs to determine the number of dimensions needed
    // overall for the Contraction.
    for (const PolyNodePtr &idx : node->sinkIdxs) {
      collector.visit(idx.get());
    }

    for (const PolyMap &src : node->srcs) {
      for (const PolyNodePtr &idx : src.idxs) {
        collector.visit(idx.get());
      }
    }

    for (const Constraint &constraint : node->constraints) {
      collector.visit(constraint.lhs.get());
    }

    // Now construct the AffineMap for each source and the sink.
    for (const PolyNodePtr &idx : node->sinkIdxs) {
      AffineExpr expr = makeExpr(idx);
      sinkExprs.push_back(expr);
    }

    for (const PolyMap &src : node->srcs) {
      SmallVector<AffineExpr, 8> srcExprs;
      for (const PolyNodePtr &idx : src.idxs) {
        AffineExpr expr = makeExpr(idx);
        srcExprs.push_back(expr);
      }
      AffineMap srcMap = makeMap(srcExprs);
      srcs.push_back(srcMap);
      tensors.push_back(builder.lookupNode(src.ref));
    }

    // Convert the constraints.
    for (const Constraint &constraint : node->constraints) {
      consExprs.emplace_back(simplifyAffineExpr(makeExpr(constraint.lhs),
                                                collector.idxs.size(), 0));
      consExprs.emplace_back(makeConstraint(constraint));
    }
  }

  tile::ContractionOp build() {
    AffineMap sinkMap = makeMap(sinkExprs);
    return builder.create<tile::ContractionOp>( //
        builder.getUnknownLoc(),
        /*resultType=*/resultType,
        /*init=*/getInit(),
        /*tensors=*/tensors,
        /*agg=*/node->aggKind,
        /*combo=*/node->comboKind,
        /*sink=*/sinkMap,
        /*srcs=*/srcs,
        /*cons=*/getConstraints(),
        /*name=*/node->name);
  }

  Value getInit() {
    if (!node->init) {
      node->init = createIdentity();
      node->init->parent = node->parent;
    }
    return builder.lookupNode(node->init);
  }

  ExprNodePtr createIdentity() {
    Type elementType = resultType.getElementType();
    if (elementType.isa<FloatType>()) {
      auto init = std::make_shared<ast::ExprNodeConstFloat>(
          tile::getFloatIdentity(node->aggKind));
      Value value = builder.create<tile::ConstantOp>(builder.getUnknownLoc(),
                                                     elementType, init->value);
      builder.addNode(init, value);
      return init;
    }
    if (elementType.isSignedInteger()) {
      auto init = std::make_shared<ast::ExprNodeConstSigned>(
          tile::getSignedIntegerIdentity(node->aggKind));
      Value value = builder.create<tile::ConstantOp>(builder.getUnknownLoc(),
                                                     elementType, init->value);
      builder.addNode(init, value);
      return init;
    }
    auto init = std::make_shared<ast::ExprNodeConstUnsigned>(
        tile::getUnsignedIntegerIdentity(node->aggKind));
    Value value = builder.create<tile::ConstantOp>(builder.getUnknownLoc(),
                                                   elementType, init->value);
    builder.addNode(init, value);
    return init;
  }

  AffineExpr makeExpr(const PolyNodePtr &node) { return visit(node.get()); }

  AffineMap makeMap(ArrayRef<AffineExpr> exprs) {
    return AffineMap::get(collector.idxs.size(), 0, exprs, context);
  }

  AffineExpr makeConstraint(const Constraint &constraint) {
    // Constraints are in the form `lhs < rhs`
    // IntegerSets are in the form `x >= 0`
    // lhs < rhs
    // -lhs > -rhs             multiply -1
    // rhs - lhs > 0           add rhs
    // rhs - lhs - 1 >= 0      subtract 1
    int64_t rhs = evaluator->evaluate(constraint.rhs);
    auto expr = rhs - makeExpr(constraint.lhs) - 1;
    return simplifyAffineExpr(expr, collector.idxs.size(), 0);
  }

  IntegerSet getConstraints() {
    if (consExprs.empty()) {
      return IntegerSet::getEmptySet(collector.idxs.size(), 0, context);
    }
    SmallVector<bool, 4> flags(consExprs.size(), false);
    return IntegerSet::get(collector.idxs.size(), 0, consExprs, flags);
  }

  AffineExpr visitDim(const PolyNodeDim *node) {
    int64_t value = evaluator->evaluate(node->dim);
    return builder.getAffineConstantExpr(value);
  }

  AffineExpr visitIndex(const PolyNodeIndex *node) {
    auto pos =
        std::distance(collector.idxs.begin(), llvm::find(collector.idxs, node));
    return builder.getAffineDimExpr(pos);
  }

  AffineExpr visitLiteral(const PolyNodeLiteral *node) {
    return builder.getAffineConstantExpr(node->value);
  }

  AffineExpr visitOp(const PolyNodeOp *node) {
    switch (node->op) {
    case AffineOp::Add:
      return visit(node->operands[0].get()) + visit(node->operands[1].get());
    case AffineOp::Div:
      return visit(node->operands[0].get())
          .floorDiv(visit(node->operands[1].get()));
    case AffineOp::Max:
      throw std::runtime_error("NYI: AffineOp::Max"); // TODO
    case AffineOp::Min:
      throw std::runtime_error("NYI: AffineOp::Min"); // TODO
    case AffineOp::Mul:
      return visit(node->operands[0].get()) * visit(node->operands[1].get());
    case AffineOp::Neg:
      return -visit(node->operands[0].get());
    case AffineOp::Sub:
      return visit(node->operands[0].get()) - visit(node->operands[1].get());
    default:
      break;
    }
    llvm_unreachable("Invalid PolyNodeOp, invalid AffineOp");
  }

  OpBuilder &builder;
  MLIRContext *context;
  Evaluator *evaluator;
  ExprNodeContraction *node;
  IndexCollector collector;
  SmallVector<Value, 3> tensors;
  SmallVector<AffineMap, 3> srcs;
  SmallVector<AffineExpr, 8> sinkExprs;
  SmallVector<AffineExpr, 8> consExprs;
  RankedTensorType resultType;
};

struct ProgramBuilder {
  explicit ProgramBuilder(llvm::StringRef name)
      : program(std::make_shared<compiler::Program>(name)),
        context(program->context.get()), loc(UnknownLoc::get(context)),
        module(*program->module), builder(module) {
    context->getOrLoadDialect<dialect::tile::TileDialect>();
    context->getOrLoadDialect<dialect::layer::LayerDialect>();
    context->getOrLoadDialect<math::MathDialect>();
    context->getOrLoadDialect<StandardOpsDialect>();
  }

  std::shared_ptr<Program> build(const ProgramArguments &args) {
    std::vector<Type> inputTypes;
    std::vector<ExprNodePtr> inputNodes;
    for (auto item : llvm::enumerate(args.inputs)) {
      if (args.shapes.size()) {
        auto *node = llvm::cast<ast::ExprNodeInput>(item.value().get());
        node->shape = args.shapes[item.index()];
      }
      TensorShape shape = evaluator.getShape(item.value().get());
      RankedTensorType rankedTensorType = builder.getRankedTensorType(shape);
      inputTypes.push_back(rankedTensorType);
      program->inputs.emplace_back(rankedTensorType);
      inputNodes.push_back(item.value());
    }

    for (const ExprNodePtr &node : args.outputs) {
      TensorShape shape = evaluator.getShape(node.get());
      RankedTensorType rankedTensorType = builder.getRankedTensorType(shape);
      program->outputs.emplace_back(rankedTensorType);
    }

    std::vector<ExprNodePtr> flat = getFlatAst(args);
    for (const ExprNodePtr &node : flat) {
      if (auto *constTensor = llvm::dyn_cast<ExprNodeConstTensor>(node.get())) {
        TensorShape shape = constTensor->buffer->shape();
        RankedTensorType rankedTensorType = builder.getRankedTensorType(shape);
        program->constants.emplace_back(
            compiler::ConstantArgument{rankedTensorType, constTensor->buffer});
        inputTypes.emplace_back(rankedTensorType);
        inputNodes.push_back(node);
      }
    }

    FunctionType funcType =
        FunctionType::get(context, inputTypes, program->outputs);
    FuncOp funcOp = FuncOp::create(loc, kEntrypoint, funcType, {});
    size_t numInputs = inputTypes.size() - program->constants.size();
    for (size_t i = 0; i < program->constants.size(); i++) {
      funcOp.setArgAttr(numInputs + i, "tile.const", builder.getIndexAttr(i));
    }
    Block *body = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(body);
    module.push_back(funcOp);

    for (auto [node, blockArg] : llvm::zip(inputNodes, funcOp.getArguments())) {
      builder.addNode(node, blockArg);
    }

    for (const ExprNodePtr &node : flat) {
      Value value =
          TypeSwitch<ExprNode *, Value>(node.get())
              .Case<ExprNodeCast>(
                  [&](ExprNodeCast *node) { return handleCast(node); })
              .Case<ExprNodeConstSigned>([&](ExprNodeConstSigned *node) {
                return handleConstSigned(node);
              })
              .Case<ExprNodeConstUnsigned>([&](ExprNodeConstUnsigned *node) {
                return handleConstUnsigned(node);
              })
              .Case<ExprNodeConstFloat>([&](ExprNodeConstFloat *node) {
                return handleConstFloat(node);
              })
              .Case<ExprNodeConstTensor>(
                  [&](ExprNodeConstTensor *node) { return nullptr; })
              .Case<ExprNodeContraction>([&](ExprNodeContraction *node) {
                return handleContraction(node);
              })
              .Case<ExprNodeDim>(
                  [&](ExprNodeDim *node) { return handleDim(node); })
              .Case<ExprNodeElement>(
                  [&](ExprNodeElement *node) { return handleElement(node); })
              .Case<ExprNodeInput>([&](ExprNodeInput *node) { return nullptr; })
              .Case<ExprNodeIntrinsic>([&](ExprNodeIntrinsic *node) {
                return handleIntrinsic(node);
              })
              .Case<ExprNodeLayer>(
                  [&](ExprNodeLayer *node) { return handleLayer(node); })
              .Case<ExprNodePragma>(
                  [&](ExprNodePragma *node) { return handlePragma(node); });
      if (value) {
        builder.addNode(node, value);
      }
    }

    llvm::SetVector<Value> returnOperands;
    for (const ExprNodePtr &node : args.outputs) {
      Value value = builder.lookupNode(node);
      if (!value) {
        throw std::runtime_error("Output not found while building program.");
      }
      auto defOp = value.getDefiningOp();
      if (!defOp || isa<tile::ReshapeOp>(defOp) ||
          returnOperands.count(value)) {
        value = builder.create<tile::IdentOp>(loc, value.getType(), value);
      }
      returnOperands.insert(value);
    }
    builder.create<ReturnOp>(loc, returnOperands.getArrayRef());

    program->entry = kEntrypoint;

    IVLOG(3, "\n" << debugString(module));

    PassManager pm(context);
    ScopedDiagnosticHandler diagHandler(pm.getContext(), [&](Diagnostic &diag) {
      llvm::errs() << getDiagKindStr(diag.getSeverity()) << ": " << diag.str()
                   << "\n";
      for (auto &note : diag.getNotes()) {
        llvm::errs() << "  " << note << "\n";
      }
      return success();
    });

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<FuncOp>(tile::createMaterializePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    auto result = pm.run(module);

    program->tileIR = debugString(module);
    IVLOG(2, "\n" << program->tileIR);
    if (failed(result)) {
      throw std::runtime_error("Program build failure.");
    }

    return program;
  }

  Value handleCast(ExprNodeCast *node) {
    TensorShape shape = evaluator.getShape(node);
    Type elementType = builder.getElementType(shape.elementType);
    RankedTensorType resultType =
        RankedTensorType::get(shape.sizes, elementType);
    Value tensor = builder.lookupNode(node->expr);
    return builder.create<tile::CastOp>(loc, resultType, tensor);
  }

  Value handleConstFloat(ExprNodeConstFloat *node) {
    Type type = builder.getAPFloatType();
    return builder.create<tile::ConstantOp>(builder.getUnknownLoc(), type,
                                            node->value);
  }

  Value handleConstSigned(ExprNodeConstSigned *node) {
    Type type = builder.getAPSignedIntegerType();
    return builder.create<tile::ConstantOp>(builder.getUnknownLoc(), type,
                                            node->value);
  }

  Value handleConstUnsigned(ExprNodeConstUnsigned *node) {
    Type type = builder.getAPUnsignedIntegerType();
    return builder.create<tile::ConstantOp>(builder.getUnknownLoc(), type,
                                            node->value);
  }

  Value handleContraction(ExprNodeContraction *node) {
    return ContractionBuilder(builder, &evaluator, node).build();
  }

  Value handleDim(ExprNodeDim *node) {
    int64_t value = evaluator.evaluate(node->dim);
    Type type = builder.getAPUnsignedIntegerType();
    return builder.create<tile::ConstantOp>(builder.getUnknownLoc(), type,
                                            value);
  }

  Value handleElement(ExprNodeElement *node) {
    return builder.lookupElement(node->expr, node->ordinal);
  }

  Value handleIntrinsic(ExprNodeIntrinsic *node) {
    SmallVector<Value, 8> operands;
    SmallVector<Type, 2> resultTypes;
    auto resultShapes = evaluator.getShapes(node);
    for (auto shape : resultShapes) {
      resultTypes.push_back(builder.getRankedTensorType(shape));
    }
    for (const ExprNodePtr &operand : node->operands) {
      Value value = builder.lookupNode(operand);
      operands.push_back(value);
    }
    using IntrinsicBuilder = std::function<Value()>;
    auto intrinsicBuilder =
        llvm::StringSwitch<IntrinsicBuilder>(node->op)
            .Case("argsort", [&]() { return makeArgSortOp(node, operands); })
            .Case("index", [&]() { return makeIndexOp(node, operands); })
            .Case("prng", [&]() { return makePrngOp(node, operands); })
            .Case("reshape", [&]() { return makeReshapeOp(node, operands); })
            .Case("scatter", [&]() { return makeScatterOp(node, operands); })
            .Case("gather", [&]() { return makeGatherOp(node, operands); })
            .Default([&]() {
              const AbstractOperation *abstractOp = lookupOperation(node->op);
              OperationState state(loc, abstractOp->name);
              state.addOperands(operands);
              state.addTypes(resultTypes);
              Operation *op = builder.createOperation(state);
              return op->getResult(0);
            });
    return intrinsicBuilder();
  }

  Value handleLayer(ExprNodeLayer *node) {
    AstTraversal traversal(node->results, node);
    SmallVector<Value, 8> operands;
    for (const ExprNodePtr &operand : node->operands) {
      operands.push_back(builder.lookupNode(operand));
    }
    llvm::SmallVector<Value> results;
    for (const ExprNodePtr &result : node->results) {
      results.push_back(builder.lookupNode(result));
    }
    llvm::SmallVector<Type, 4> resultTypes;
    for (Value val : results) {
      resultTypes.push_back(val.getType());
    }
    std::vector<NamedAttribute> attrs;
    for (const auto &kvp : node->attrs) {
      Attribute value = builder.getAttribute(kvp.getValue());
      attrs.push_back(builder.getNamedAttr(kvp.getKey(), value));
    }
    auto layerOp = builder.create<layer::BoxOp>(
        loc, node->op, operands, resultTypes, builder.getDictionaryAttr(attrs));
    BlockAndValueMapping mapper;
    OpBuilder bodyBuilder(layerOp.body());
    for (auto tuple : llvm::zip(operands, layerOp.body().getArguments())) {
      Value outer, inner;
      std::tie(outer, inner) = tuple;
      mapper.map(outer, inner);
    }
    llvm::SmallVector<Value, 4> innerResults(results.size());
    llvm::SetVector<Operation *> toRemove;
    for (const ExprNodePtr &node : traversal.getFlat()) {
      Value value = builder.lookupNode(node);
      Operation *op = value.getDefiningOp();
      if (toRemove.contains(op)) {
        // this has already been visited
        continue;
      }
      assert(op && "Unexpected block argument");
      Operation *clonedOp = bodyBuilder.clone(*op, mapper);

      auto itLayerResults = std::find(results.begin(), results.end(), value);
      if (itLayerResults != results.end()) {
        if (auto opResultVal = value.cast<OpResult>()) {
          auto idxLayerResults = std::distance(results.begin(), itLayerResults);
          innerResults[idxLayerResults] =
              clonedOp->getResult(opResultVal.getResultNumber());
        }
      }
      toRemove.insert(op);
    }
    bodyBuilder.create<layer::ReturnOp>(loc, innerResults);
    for (auto it = toRemove.rbegin(); it != toRemove.rend(); ++it) {
      (*it)->erase();
    }
    SmallVector<Value, 4> tuple;
    for (OpResult result : layerOp.getResults()) {
      tuple.push_back(result);
    }
    builder.exprTuples[node] = tuple;
    return nullptr;
  }

  Value handlePragma(ExprNodePragma *node) {
    Value tensor = builder.lookupNode(node->expr);
    std::vector<NamedAttribute> attrs;
    for (const auto &kvp : node->attrs) {
      Attribute value = builder.getAttribute(kvp.getValue());
      attrs.push_back(builder.getNamedAttr(kvp.getKey(), value));
    }
    return builder
        .create<tile::PragmaOp>(loc, tensor, node->op,
                                builder.getDictionaryAttr(attrs))
        .result();
  }

  Value makeGatherOp(ExprNodeIntrinsic *node, ArrayRef<Value> operands) {
    TensorShape shape = evaluator.getShape(node);
    RankedTensorType resultType = builder.getRankedTensorType(shape);
    IntegerAttr axis;
    IntegerAttr interpolationMode;
    IntegerAttr nearestMode;
    FloatAttr cubeCoeff;
    IntegerAttr mode;
    IntegerAttr batchDims;
    if (!matchPattern(operands[2], m_Constant(&axis))) {
      throw std::runtime_error("'gather' primitive expects the 'axis' argument "
                               "to be a constant integer");
    }
    if (!matchPattern(operands[3], m_Constant(&interpolationMode))) {
      throw std::runtime_error(
          "'gather' primitive expects the 'interpolationMode' argument "
          "to be a constant integer");
    }
    if (!matchPattern(operands[4], m_Constant(&nearestMode))) {
      throw std::runtime_error(
          "'gather' primitive expects the 'nearestMode' argument "
          "to be a constant integer");
    }
    if (!matchPattern(operands[5], m_Constant(&cubeCoeff))) {
      throw std::runtime_error(
          "'gather' primitive expects the 'cubeCoeff' argument "
          "to be a constant float");
    }
    if (!matchPattern(operands[6], m_Constant(&mode))) {
      throw std::runtime_error("'gather' primitive expects the 'mode' argument "
                               "to be a constant integer");
    }
    if (!matchPattern(operands[7], m_Constant(&batchDims))) {
      throw std::runtime_error(
          "'gather' primitive expects the 'batchDims' argument "
          "to be a constant integer");
    }
    auto op = builder.create<tile::GatherOp>(
        loc, resultType, operands.take_front(2),
        builder.getIndexAttr(axis.getInt()), interpolationMode, nearestMode,
        cubeCoeff, mode, builder.getIndexAttr(batchDims.getInt()));
    return op.result();
  }

  Value makeArgSortOp(ExprNodeIntrinsic *node, ArrayRef<Value> operands) {
    TensorShape shape = evaluator.getShape(node);
    auto i32Type = builder.getIntegerType(32, /*isSigned=*/true);
    RankedTensorType resultType = RankedTensorType::get(shape.sizes, i32Type);
    Value tensor = operands[0];
    Value axis = operands[1];
    Value direction = operands[2];
    IntegerAttr axisAttr;
    if (!matchPattern(axis, m_Constant(&axisAttr))) {
      throw std::runtime_error(
          "'argsort' requires operand #2 to be a constant integer.");
    }
    IntegerAttr dirIntAttr;
    if (!matchPattern(direction, m_Constant(&dirIntAttr))) {
      throw std::runtime_error(
          "'argsort' requires operand #3 to be a constant integer.");
    }
    Optional<tile::SortDirection> dir =
        tile::symbolizeSortDirection(dirIntAttr.getInt());
    if (!dir) {
      throw std::runtime_error(
          "'argsort' requires operand #3 to be a valid SortDirection.");
    }
    auto dirAttr = tile::SortDirectionAttr::get(context, *dir);
    IntegerAttr axisIndexAttr = builder.getIndexAttr(axisAttr.getInt());
    auto op = builder.create<tile::ArgSortOp>(loc, resultType, tensor,
                                              axisIndexAttr, dirAttr);
    return op.result();
  }

  Value makeReshapeOp(ExprNodeIntrinsic *node, ArrayRef<Value> operands) {
    TensorShape shape = evaluator.getShape(node);
    RankedTensorType resultType = builder.getRankedTensorType(shape);
    auto op = builder.create<tile::ReshapeOp>(loc, resultType, operands[0]);
    return op.result();
  }

  Value makeScatterOp(ExprNodeIntrinsic *node, ArrayRef<Value> operands) {
    TensorShape shape = evaluator.getShape(node);
    IntegerAttr axis;
    IntegerAttr mode;

    if (!matchPattern(operands[3], m_Constant(&axis))) {
      throw std::runtime_error(
          "'scatter' primitive expects the 'axis' argument "
          "to be a constant integer");
    }
    if (!matchPattern(operands[4], m_Constant(&mode))) {
      throw std::runtime_error(
          "'scatter' primitive expects the 'mode' argument "
          "to be a constant integer");
    }
    RankedTensorType resultType = builder.getRankedTensorType(shape);
    auto op = builder.create<tile::ScatterOp>(
        loc, resultType, operands.take_front(3),
        builder.getIndexAttr(axis.getInt()), mode);
    return op.result();
  }

  Value makeIndexOp(ExprNodeIntrinsic *node, ArrayRef<Value> operands) {
    if (operands.size() < 1) {
      throw std::runtime_error(
          "'index' primitive expects at least one operand");
    }
    Value axis = operands.front();
    IntegerAttr axisAttr;
    if (!m_Constant(&axisAttr).match(axis.getDefiningOp())) {
      throw std::runtime_error(
          "'index' primitive expects argument 1 to be a constant integer");
    }
    TensorShape shape = evaluator.getShape(node);
    RankedTensorType resultType = builder.getRankedTensorType(shape);
    IntegerAttr indexAttr = builder.getIndexAttr(axisAttr.getInt());
    auto op = builder.create<tile::IndexOp>(loc, resultType, indexAttr);
    return op.result();
  }

  Value makePrngOp(ExprNodeIntrinsic *node, ArrayRef<Value> operands) {
    if (operands.size() < 1) {
      throw std::runtime_error("'prng' primitive expects at least one operand");
    }
    Value state = operands.front();
    SmallVector<Type, 2> resultTypes;
    for (const TensorShape &shape : evaluator.getShapes(node)) {
      resultTypes.push_back(builder.getRankedTensorType(shape));
    }
    auto op = builder.create<tile::PrngOp>(loc, resultTypes, state);
    SmallVector<Value, 4> tuple;
    for (OpResult result : op.getResults()) {
      tuple.push_back(result);
    }
    builder.exprTuples[node] = tuple;
    return nullptr;
  }

  const AbstractOperation *lookupOperation(StringRef op) {
    auto opName = tile::TileDialect::getCanonicalOpName(op);
    auto abstractOp = AbstractOperation::lookup(opName, context);
    if (!abstractOp) {
      throw std::runtime_error("Unknown EDSL primitive: " + op.str());
    }
    return abstractOp;
  }

  std::string name;
  std::shared_ptr<Program> program;
  MLIRContext *context;
  Location loc;
  ModuleOp module;
  OpBuilder builder;
  Evaluator evaluator;
};

} // namespace

std::shared_ptr<Program> buildProgram(llvm::StringRef name,
                                      const ProgramArguments &args) {
  if (name.empty()) {
    name = "module";
  }
  return ProgramBuilder(name).build(args);
}

} // namespace pmlc::ast
