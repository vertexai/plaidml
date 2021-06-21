// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ir/ops.h"

#include <vector>

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/MathExtras.h"

#include "pmlc/dialect/tile/ir/util.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

using llvm::SmallVector;

LogicalResult ArgSortOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult ContractionOp::materializeOperands(OpBuilder &builder) {
  Operation *op = getOperation();
  if (combo() == CombinationKind::cond) {
    auto operands = op->getOpOperands();
    return success(
        succeeded(tile::materializeOperands(
            builder, op,
            llvm::ArrayRef<OpOperand *>{&operands[0], &operands[3]})) &&
        succeeded(tile::materializeOperands(
            builder, op,
            llvm::ArrayRef<OpOperand *>{&operands[1], &operands[2]})));
  }
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult GatherOp::materializeOperands(OpBuilder &builder) {
  Operation *op = getOperation();
  return tile::materializeOperands(builder, op,
                                   op->getOpOperands().take_front());
}

LogicalResult ReshapeOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult ScatterOp::materializeOperands(OpBuilder &builder) {
  Operation *op = getOperation();
  return tile::materializeOperands(builder, op,
                                   op->getOpOperands().take_front());
}

LogicalResult ShapeOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult PragmaOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

// ---- ReshapeOp ----

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  // reshape(x, x.shape) -> x
  if (tensor().getType() == getType()) {
    return tensor();
  }
  return {};
}

// ---- ContractionOp ----

unsigned ContractionOp::getNumTensors(CombinationKind combo) {
  switch (combo) {
  case CombinationKind::none:
    return 1;
  case CombinationKind::add:
  case CombinationKind::eq:
  case CombinationKind::mul:
    return 2;
  case CombinationKind::cond:
    return 3;
  default:
    throw std::runtime_error("Invalid combination op");
  }
}

void ContractionOp::build(OpBuilder &builder, OperationState &result,
                          Type resultType, Value init, ValueRange tensors,
                          AggregationKind agg, CombinationKind combo,
                          AffineMap sink, ArrayRef<AffineMap> srcs,
                          IntegerSet cons, StringRef name) {
  result.addOperands(init);
  result.addOperands(tensors);
  result.addTypes(resultType);
  result.addAttribute("agg",
                      builder.getI64IntegerAttr(static_cast<int64_t>(agg)));
  result.addAttribute("combo",
                      builder.getI64IntegerAttr(static_cast<int64_t>(combo)));
  result.addAttribute(getSinkAttrName(), AffineMapAttr::get(sink));
  result.addAttribute(getSourcesAttrName(),
                      builder.getAffineMapArrayAttr(srcs));
  if (!cons.isEmptyIntegerSet()) {
    result.addAttribute(getConstraintsAttrName(), IntegerSetAttr::get(cons));
  }
  if (name.size()) {
    result.addAttribute("name", builder.getStringAttr(name));
  }
}

AffineMap ContractionOp::getSourceMap(unsigned i) {
  return srcs().getValue()[i].cast<AffineMapAttr>().getValue();
}

void ContractionOp::setLowerBounds(ArrayRef<int64_t> bounds) {
  SmallVector<AffineExpr, 6> exprs;
  for (auto dim : bounds) {
    exprs.push_back(getAffineConstantExpr(dim, getContext()));
  }
  auto map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, exprs, getContext());
  (*this)->setAttr(getLowerBoundsAttrName(), AffineMapAttr::get(map));
}

void ContractionOp::setUpperBounds(ArrayRef<int64_t> bounds) {
  SmallVector<AffineExpr, 6> exprs;
  for (auto dim : bounds) {
    exprs.push_back(getAffineConstantExpr(dim, getContext()));
  }
  auto map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, exprs, getContext());
  (*this)->setAttr(getUpperBoundsAttrName(), AffineMapAttr::get(map));
}

void ContractionOp::setSink(AffineMap sink) {
  (*this)->setAttr(getSinkAttrName(), AffineMapAttr::get(sink));
}

void ContractionOp::setSources(ArrayRef<AffineMap> srcs) {
  SmallVector<Attribute, 4> attrs;
  for (auto src : srcs) {
    attrs.push_back(AffineMapAttr::get(src));
  }
  (*this)->setAttr(getSourcesAttrName(), ArrayAttr::get(getContext(), attrs));
}

void ContractionOp::setConstraints(IntegerSet cons) {
  if (cons.isEmptyIntegerSet()) {
    (*this)->removeAttr(getConstraintsAttrName());
  } else {
    (*this)->setAttr(getConstraintsAttrName(), IntegerSetAttr::get(cons));
  }
}

unsigned ContractionOp::getNumTensors() { return getNumTensors(combo()); }

unsigned ContractionOp::getNumSymbols() {
  return getNumOperands() - 1 - getNumTensors();
}

Value ContractionOp::getTensor(unsigned i) {
  return *std::next(operands().begin(), i);
}

Value ContractionOp::getSymbol(unsigned i) {
  return *std::next(operands().begin(), getNumTensors() + i);
}

void printContractionOp(OpAsmPrinter *printer, ContractionOp op) {
  SmallVector<StringRef, 3> elidedAttrs = {"agg", "combo", "name"};
  *printer << op.getOperation()->getName() << ' ';
  *printer << util::stringifyAggregationKind(op.agg());
  *printer << ", ";
  *printer << util::stringifyCombinationKind(op.combo());
  *printer << ", ";
  printer->printOperand(op.init());
  auto numTensors = op.getNumTensors();
  for (unsigned i = 0; i < numTensors; i++) {
    *printer << ", ";
    printer->printOperand(op.getTensor(i));
  }
  auto numSymbols = op.getNumSymbols();
  if (numSymbols) {
    *printer << " [";
    for (unsigned i = 0; i < numSymbols; i++) {
      if (i) {
        *printer << ", ";
      }
      printer->printOperand(op.getSymbol(i));
    }
    *printer << ']';
  }
  printer->printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  *printer << " : ";
  printer->printType(op.init().getType());
  *printer << ", ";
  for (unsigned i = 0; i < numTensors; i++) {
    if (i) {
      *printer << ", ";
    }
    printer->printType(op.getTensor(i).getType());
  }
  *printer << " -> ";
  printer->printType(op.result().getType());
}

ParseResult parseContractionOp(OpAsmParser *parser, OperationState &result) {
  StringRef strAgg;
  StringRef strCombo;
  OpAsmParser::OperandType init;
  SmallVector<OpAsmParser::OperandType, 3> tensors;
  SmallVector<OpAsmParser::OperandType, 8> symbols;
  SmallVector<Type, 4> types;
  Type resultType;
  if (parser->parseKeyword(&strAgg) || parser->parseComma() ||
      parser->parseKeyword(&strCombo) || parser->parseComma() ||
      parser->parseOperand(init) || parser->parseComma()) {
    return failure();
  }

  auto agg = util::symbolizeAggregationKind(strAgg);
  if (!agg) {
    return failure();
  }
  result.addAttribute("agg", parser->getBuilder().getI64IntegerAttr(
                                 static_cast<int64_t>(agg.getValue())));

  auto combo = util::symbolizeCombinationKind(strCombo);
  if (!combo) {
    return failure();
  }
  result.addAttribute("combo", parser->getBuilder().getI64IntegerAttr(
                                   static_cast<int64_t>(combo.getValue())));

  auto numTensors = ContractionOp::getNumTensors(combo.getValue());
  if (parser->parseOperandList(tensors, numTensors) ||
      parser->parseOperandList(symbols,
                               OpAsmParser::Delimiter::OptionalSquare) ||
      parser->parseOptionalAttrDict(result.attributes) ||
      parser->parseColonTypeList(types) || parser->parseArrow() ||
      parser->parseType(resultType)) {
    return failure();
  }

  // TODO: parse a FunctionType here

  auto loc = parser->getCurrentLocation();
  auto indexType = parser->getBuilder().getIndexType();
  auto tensorTypes = llvm::makeArrayRef(types).drop_front();
  if (parser->resolveOperand(init, types.front(), result.operands) ||
      parser->resolveOperands(tensors, tensorTypes, loc, result.operands) ||
      parser->resolveOperands(symbols, indexType, result.operands)) {
    return failure();
  }

  result.addTypes(resultType);
  return success();
}

bool isAnyScalar(Type type) {
  return type.isIndex() || type.isa<FloatType>() || type.isInteger(1) ||
         type.isSignedInteger() || type.isUnsignedInteger();
}

bool isEltwiseAny(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    auto elementType = rankedTensorType.getElementType();
    return isAnyScalar(elementType);
  }
  return isAnyScalar(type);
}

LogicalResult verifyContractionOp(ContractionOp op) {
  auto numTensors = op.getNumTensors();
  auto numSymbols = op.getNumSymbols();
  SmallVector<Value, 8> variadic(op.operands());
  if (variadic.size() < numTensors) {
    return op.emitOpError("combo '")
           << util::stringifyCombinationKind(op.combo()) << "' requires "
           << numTensors << " tensor operands";
  }
  auto shape = op.shape();
  auto resultType = op.result().getType().cast<RankedTensorType>();
  if (!resultType.hasStaticShape() && !shape.hasValue()) {
    return op.emitOpError(
        "attribute 'shape' is required when result type is dynamic");
  }
  unsigned expectedSymbols = op.sink().getNumSymbols();
  if (shape.hasValue()) {
    expectedSymbols += shape->getNumSymbols();
  }
  for (auto src : op.srcs()) {
    auto map = src.cast<AffineMapAttr>();
    expectedSymbols += map.getValue().getNumSymbols();
  }
  if (op.cons().hasValue()) {
    expectedSymbols += op.cons().getValue().getNumSymbols();
  }
  if (expectedSymbols != numSymbols) {
    return op.emitOpError("has incorrect number of symbols: expected ")
           << expectedSymbols << " but found " << numSymbols;
  }
  for (unsigned i = 0; i < numTensors; i++) {
    auto type = op.getTensor(i).getType();
    if (!isEltwiseAny(type)) {
      return op.emitOpError("tensor #")
             << i << " must be eltwise-any, but got " << type;
    }
  }
  for (unsigned i = 0; i < numSymbols; i++) {
    auto type = op.getSymbol(i).getType();
    if (!type.isa<IndexType>()) {
      return op.emitOpError("symbol #")
             << i << " must be index, but got " << type;
    }
  }
  return success();
}

LogicalResult verifyReshapeOp(ReshapeOp op) {
  auto inType = op.tensor().getType().cast<RankedTensorType>();
  auto outType = op.result().getType().cast<RankedTensorType>();
  if (inType.getElementType() != outType.getElementType()) {
    return op.emitOpError("element type mismatch");
  }
  if (inType.getNumElements() != outType.getNumElements()) {
    return op.emitOpError("element count mismatch");
  }
  return success();
}

void GatherOp::build(OpBuilder &builder, OperationState &result,
                     Type resultType, ValueRange operands, IntegerAttr axis,
                     IntegerAttr interpolationMode, IntegerAttr nearestMode,
                     FloatAttr cubeCoeff, IntegerAttr mode,
                     IntegerAttr batchDims) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  result.addOperands(operands);
  result.addAttribute("axis", axis);
  result.addAttribute("interpolationMode", interpolationMode);
  result.addAttribute("nearestMode", nearestMode);
  result.addAttribute("cubeCoeff", cubeCoeff);
  result.addAttribute("mode", mode);
  result.addAttribute("batchDims", batchDims);
  result.addTypes(resultType);
}

void ScatterOp::build(OpBuilder &builder, OperationState &result,
                      Type resultType, ValueRange operands, IntegerAttr axis,
                      IntegerAttr mode) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  result.addOperands(operands);
  result.addAttribute("axis", axis);
  result.addAttribute("mode", mode);
  result.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(LoopOp op) {
  // Check that the body defines as single block argument for the induction
  // variable.
  auto opNumResults = op.getNumResults();
  if (opNumResults == 0)
    return success();
  // If LoopOp defines values, check that the number and types of
  // the defined values match LoopOp initial iter operands and backedge
  // basic block arguments.
  if (op.getNumIterOperands() != opNumResults)
    return op.emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (op.getNumRegionIterArgs() != opNumResults)
    return op.emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = op.getIterOperands();
  auto iterArgs = op.getRegionIterArgs();
  auto opResults = op.getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter region arg and defined value";

    i++;
  }

  return RegionBranchOpInterface::verifyTypes(op);
}

/// Prints the initialization list in the form of
///   <prefix>(%inner = %outer, %inner2 = %outer2, <...>)
/// where 'inner' values are assumed to be region arguments and 'outer' values
/// are regular SSA values.
static void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

static void print(OpAsmPrinter &p, LoopOp op) {
  p << op.getOperationName() << " " << op.maxTripCount();

  printInitializationList(p, op.getRegionIterArgs(), op.getIterOperands(),
                          " iter_args");
  if (!op.getIterOperands().empty())
    p << " -> (" << op.getIterOperands().getTypes() << ')';
  p.printRegion(op.region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/op.hasIterOperands());
  p.printOptionalAttrDict(op.getAttrs());
}

static ParseResult parseLoopOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType maxTripCount;
  auto tensorType = RankedTensorType::get(ArrayRef<int64_t>{1},
                                          builder.getIntegerType(32, true));
  if (parser.parseOperand(maxTripCount) ||
      parser.resolveOperand(maxTripCount, tensorType, result.operands)) {
    return failure();
  }
  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::OperandType, 4> regionArgs, operands;
  SmallVector<Type, 4> argTypes;

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
    // Resolve input operands.
    for (auto operand_type : llvm::zip(operands, result.types))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return failure();
  }
  argTypes.push_back(tensorType);
  // Loop carried variables
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  Region *body = result.addRegion();
  if (regionArgs.size() != argTypes.size() - 1)
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  if (parser.parseRegion(*body, regionArgs,
                         {argTypes.begin() + 1, argTypes.end()}))
    return failure();

  LoopOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}
void LoopOp::build(OpBuilder &builder, OperationState &result,
                   Value maxTripCount, ValueRange iterArgs,
                   BodyBuilderFn bodyBuilder) {
  result.addOperands(maxTripCount);
  result.addOperands(iterArgs);
  for (Value v : iterArgs)
    result.addTypes(v.getType());
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  for (Value v : iterArgs)
    bodyBlock.addArgument(v.getType());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  if (iterArgs.empty() && !bodyBuilder) {
    LoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, {},
                bodyBlock.getArguments().drop_front());
  }
}
Region &LoopOp::getLoopBody() { return region(); }

bool LoopOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult LoopOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto op : ops)
    op->moveBefore(*this);
  return success();
}

LoopOp getForInductionVarOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return LoopOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast_or_null<LoopOp>(containingOp);
}

/// Return operands used when entering the region at 'index'. These operands
/// correspond to the loop iterator operands, i.e., those excluding the
/// induction variable. LoopOp only has one region, so 0 is the only valid value
/// for `index`.
OperandRange LoopOp::getSuccessorEntryOperands(unsigned index) {
  assert(index == 0 && "invalid region index");

  // The initial operands map to the loop arguments after the induction
  // variable.
  return initArgs();
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void LoopOp::getSuccessorRegions(Optional<unsigned> index,
                                 ArrayRef<Attribute> operands,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the LoopOp, branch into the body using the iterator
  // arguments.
  if (!index.hasValue()) {
    regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
    return;
  }

  // Otherwise, the loop may branch back to itself or the parent operation.
  assert(index.getValue() == 0 && "expected loop region");
  regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
  regions.push_back(RegionSuccessor(getResults()));
}

void LoopOp::getNumRegionInvocations(ArrayRef<Attribute> operands,
                                     SmallVectorImpl<int64_t> &countPerRegion) {
  assert(countPerRegion.empty());
  countPerRegion.resize(1);

  auto maxTripCount = operands[0].dyn_cast_or_null<IntegerAttr>();

  // Loop bounds are not known statically.
  if (!maxTripCount) {
    countPerRegion[0] = -1;
    return;
  }

  countPerRegion[0] = ceilDiv(maxTripCount.getValue().getSExtValue(), 1);
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.mergeBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

namespace {
// Fold away LoopOp iter arguments that are also yielded by the op.
// These arguments must be defined outside of the LoopOp region and can just be
// forwarded after simplifying the op inits, yields and returns.
//
// The implementation uses `mergeBlockBefore` to steal the content of the
// original LoopOp and avoid cloning.
struct LoopOpIterArgsFolder : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loopOp,
                                PatternRewriter &rewriter) const final {
    bool canonicalize = false;
    Block &block = loopOp.region().front();
    auto yieldOp = cast<YieldOp>(block.getTerminator());

    // An internal flat vector of block transfer
    // arguments `newBlockTransferArgs` keeps the 1-1 mapping of original to
    // transformed block argument mappings. This plays the role of a
    // BlockAndValueMapping for the particular use case of calling into
    // `mergeBlockBefore`.
    SmallVector<bool, 4> keepMask;
    keepMask.reserve(yieldOp.getNumOperands());
    SmallVector<Value, 4> newBlockTransferArgs, newIterArgs, newYieldValues,
        newResultValues;
    newBlockTransferArgs.reserve(loopOp.getNumIterOperands());
    newBlockTransferArgs.push_back(Value()); // iv placeholder with null value
    newIterArgs.reserve(loopOp.getNumIterOperands());
    newYieldValues.reserve(yieldOp.getNumOperands());
    newResultValues.reserve(loopOp.getNumResults());
    for (auto it : llvm::zip(loopOp.getIterOperands(),   // iter from outside
                             loopOp.getRegionIterArgs(), // iter inside region
                             yieldOp.getOperands())      // iter yield
    ) {
      // Forwarded is `true` when the region `iter` argument is yielded.
      bool forwarded = (std::get<1>(it) == std::get<2>(it));
      keepMask.push_back(!forwarded);
      canonicalize |= forwarded;
      if (forwarded) {
        newBlockTransferArgs.push_back(std::get<0>(it));
        newResultValues.push_back(std::get<0>(it));
        continue;
      }
      newIterArgs.push_back(std::get<0>(it));
      newYieldValues.push_back(std::get<2>(it));
      newBlockTransferArgs.push_back(Value()); // placeholder with null value
      newResultValues.push_back(Value());      // placeholder with null value
    }

    if (!canonicalize)
      return failure();

    LoopOp newLoopOp = rewriter.create<LoopOp>(
        loopOp.getLoc(), loopOp.maxTripCount(), newIterArgs);
    Block &newBlock = newLoopOp.region().front();

    // Replace the null placeholders with newly constructed values.
    newBlockTransferArgs[0] = newBlock.getArgument(0); // iv
    for (unsigned idx = 0, collapsedIdx = 0, e = newResultValues.size();
         idx != e; ++idx) {
      Value &blockTransferArg = newBlockTransferArgs[idx];
      Value &newResultVal = newResultValues[idx];
      assert((blockTransferArg && newResultVal) ||
             (!blockTransferArg && !newResultVal));
      if (!blockTransferArg) {
        blockTransferArg = newLoopOp.getRegionIterArgs()[collapsedIdx];
        newResultVal = newLoopOp.getResult(collapsedIdx++);
      }
    }

    Block &oldBlock = loopOp.region().front();
    assert(oldBlock.getNumArguments() == newBlockTransferArgs.size() &&
           "unexpected argument size mismatch");

    // No results case: the tile::loopOp builder already created a zero
    // result terminator. Merge before this terminator and just get rid of the
    // original terminator that has been merged in.
    if (newIterArgs.empty()) {
      auto newYieldOp = cast<YieldOp>(newBlock.getTerminator());
      rewriter.mergeBlockBefore(&oldBlock, newYieldOp, newBlockTransferArgs);
      rewriter.eraseOp(newBlock.getTerminator()->getPrevNode());
      rewriter.replaceOp(loopOp, newResultValues);
      return success();
    }

    // No terminator case: merge and rewrite the merged terminator.
    auto cloneFilteredTerminator = [&](YieldOp mergedTerminator) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(mergedTerminator);
      SmallVector<Value, 4> filteredOperands;
      filteredOperands.reserve(newResultValues.size());
      for (unsigned idx = 0, e = keepMask.size(); idx < e; ++idx)
        if (keepMask[idx])
          filteredOperands.push_back(mergedTerminator.getOperand(idx));
      rewriter.create<YieldOp>(mergedTerminator.getLoc(), filteredOperands);
    };

    rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);
    auto mergedYieldOp = cast<YieldOp>(newBlock.getTerminator());
    cloneFilteredTerminator(mergedYieldOp);
    rewriter.eraseOp(mergedYieldOp);
    rewriter.replaceOp(loopOp, newResultValues);
    return success();
  }
};

/// Rewriting pattern that erases loops that are known not to iterate and
/// replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter &rewriter) const override {
    auto maxTripCount = op.maxTripCount().getDefiningOp<ConstantOp>();
    if (!maxTripCount)
      return failure();

    // If the loop is known to have 0 iterations, remove it.
    llvm::APInt maxTripCountValue =
        maxTripCount.getValue().cast<IntegerAttr>().getValue();
    if (maxTripCountValue == 0) {
      rewriter.replaceOp(op, op.getIterOperands());
      return success();
    }

    // If the loop is known to have 1 iteration, inline its body and remove the
    // loop.
    if (maxTripCountValue == 1) {
      SmallVector<Value, 4> blockArgs;
      blockArgs.reserve(op.getNumIterOperands());
      llvm::append_range(blockArgs, op.getIterOperands());
      replaceOpWithRegion(rewriter, op, op.getLoopBody(), blockArgs);
      return success();
    }

    return failure();
  }
};
} // namespace

void LoopOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<LoopOpIterArgsFolder, SimplifyTrivialLoops>(context);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static ParseResult parseYieldOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  SmallVector<Type, 4> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  // Parse variadic operands list, their types, and resolve operands to SSA
  // values.
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalColonTypeList(types) ||
      parser.resolveOperands(operands, types, loc, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, tile::YieldOp op) {
  p << op.getOperationName();
  if (op.getNumOperands() != 0)
    p << ' ' << op.getOperands() << " : " << op.getOperandTypes();
}

} // namespace pmlc::dialect::tile

#include "pmlc/dialect/tile/ir/enums.cc.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ir/ops.cc.inc"

#include "pmlc/dialect/tile/ir/interfaces.cc.inc"
