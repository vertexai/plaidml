// Copyright 2020, Intel Corporation

#include <limits>
#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/tile_to_pxa/pass_detail.h"
#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/padding.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

#include "pmlc/util/ident.h"

namespace pmlc::conversion::tile_to_pxa {

namespace tile = dialect::tile;
namespace layer = dialect::layer;
namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;

using namespace mlir; // NOLINT

using util::AggregationKind;
using util::CombinationKind;
using util::GatherMode;
using util::InterpolationMode;
using util::NearestMode;
using util::ScatterMode;

namespace {

struct TypeConverter : public mlir::TypeConverter {
  TypeConverter() {
    addConversion([](FunctionType type) { return type; });
    addConversion([](FloatType type) { return type; });
    addConversion([](IntegerType type) { return tile::toSignlessType(type); });
    addConversion([](IndexType type) { return type; });
    addConversion([](MemRefType type) { return type; });
    addConversion([](stdx::ArgpackType type) { return type; });
    addConversion([this](RankedTensorType type) {
      auto elementType = type.getElementType();
      auto newType = convertType(elementType);
      assert(newType && "could not convert type");
      return MemRefType::get(type.getShape(), newType);
    });
  }
};

static Type getElementType(Type type) {
  if (auto tensorType = type.dyn_cast<TensorType>()) {
    return tensorType.getElementType();
  } else if (auto memRefType = type.dyn_cast<MemRefType>()) {
    return memRefType.getElementType();
  }
  return type;
}

static Type getElementType(Value value) {
  return getElementType(value.getType());
}

static RankedTensorType getRankedTensorType(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    return rankedTensorType;
  }
  return RankedTensorType::get({}, type);
}

static llvm::APFloat convertFloatUsingType(llvm::APFloat value,
                                           FloatType type) {
  bool losesInfo = false;
  value.convert(type.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                &losesInfo);
  return value;
}

struct ConstantOpConversion : public OpConversionPattern<tile::ConstantOp> {
  using OpConversionPattern<tile::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto stdType = tile::toSignlessType(getElementType(op));
    auto value = op.getValue();
    if (auto floatType = stdType.dyn_cast<FloatType>()) {
      auto floatAttr = value.cast<FloatAttr>();
      auto floatValue = convertFloatUsingType(floatAttr.getValue(), floatType);
      value = FloatAttr::get(floatType, floatValue);
    } else if (auto intType = stdType.dyn_cast<IntegerType>()) {
      auto intAttr = value.cast<IntegerAttr>();
      value = IntegerAttr::get(intType, intAttr.getInt());
    } else {
      llvm_unreachable("Invalid scalar constant op");
    }
    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, value);
    return success();
  }
};

static Value createCastOp(OpBuilder &builder, Location loc, Value from,
                          bool fromSigned, Type intoType, bool intoSigned) {
  auto fromType = from.getType();
  if (fromType == intoType) {
    return from;
  }
  if (auto intoFloatType = intoType.dyn_cast<FloatType>()) {
    if (auto fromFloatType = fromType.dyn_cast<FloatType>()) {
      if (fromFloatType.getWidth() < intoFloatType.getWidth()) {
        // FPExtOp: FloatType -> wider FloatType
        return builder.create<mlir::FPExtOp>(loc, from, intoType).getResult();
      }
      // FPTruncOp: FloatType -> narrower FloatType
      return builder.create<mlir::FPTruncOp>(loc, from, intoType).getResult();
    }
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromSigned) {
        // SIToFPOp: IntegerType -> FloatType
        return builder.create<mlir::SIToFPOp>(loc, from, intoType).getResult();
      } else {
        // UIToFPOp: IntegerType -> FloatType
        return builder.create<mlir::UIToFPOp>(loc, intoType, from).getResult();
      }
    }
    if (auto fromIndexType = fromType.dyn_cast<IndexType>()) {
      auto i64Type = builder.getIntegerType(64);
      auto intCastOp = builder.create<mlir::IndexCastOp>(loc, from, i64Type);
      return builder.create<mlir::SIToFPOp>(loc, intCastOp, intoType)
          .getResult();
    }
  }
  if (auto intoIntType = intoType.dyn_cast<IntegerType>()) {
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromIntType.getWidth() < intoIntType.getWidth()) {
        if (fromSigned) {
          // SignExtendIOp: IntegerType -> wider signed int
          return builder.create<mlir::SignExtendIOp>(loc, from, intoType)
              .getResult();
        }
        // ZeroExtendIOp: IntegerType -> wider unsigned int
        return builder.create<mlir::ZeroExtendIOp>(loc, from, intoType)
            .getResult();
      }
      // TruncateIOp: IntegerType -> narrower IntegerType
      return builder.create<mlir::TruncateIOp>(loc, from, intoType).getResult();
    }
    if (auto fromFloatType = fromType.dyn_cast<FloatType>()) {
      if (intoSigned) {
        // FPToSIOp: FloatType -> signed IntegerType
        return builder.create<mlir::FPToSIOp>(loc, from, intoType).getResult();
      } else {
        // FPToUIOp: FloatType -> unsigned IntegerType
        return builder.create<mlir::FPToUIOp>(loc, from, intoType).getResult();
      }
    }
    if (auto fromIndexType = fromType.dyn_cast<IndexType>()) {
      auto intType = builder.getIntegerType(intoIntType.getWidth());
      return builder.create<mlir::IndexCastOp>(loc, from, intType);
    }
  }
  llvm_unreachable("Unsupported cast op");
}

struct Matcher {
  LogicalResult operator()(Operation *op) { return success(match(op)); }
  virtual bool match(Operation *op) const { return false; }
};

struct AlwaysTrue : Matcher {
  bool match(Operation *op) const final { return true; }
};

template <typename InnerPredicate>
struct ResultIs : Matcher {
  bool match(Operation *op) const final {
    InnerPredicate pred;
    return pred.match(op->getResult(0).getType());
  }
};

template <typename InnerPredicate>
struct AnyOperandIs : Matcher {
  bool match(Operation *op) const final {
    for (auto operand : op->getOperands()) {
      InnerPredicate pred;
      if (pred.match(operand.getType())) {
        return true;
      }
    }
    return false;
  }
};

template <typename InnerPredicate>
struct OperandsAre : Matcher {
  bool match(Operation *op) const final {
    for (auto operand : op->getOperands()) {
      InnerPredicate pred;
      if (!pred.match(operand.getType())) {
        return false;
      }
    }
    return true;
  }
};

template <typename InnerPredicate>
struct FirstOperandIs : Matcher {
  bool match(Operation *op) const final {
    InnerPredicate pred;
    if (op->getNumOperands() == 0) {
      return false;
    }
    return pred.match(op->getOperand(0).getType());
  }
};

template <typename InnerPredicate>
struct AnyComparandIs : Matcher {
  bool match(Operation *op) const final {
    SmallVector<Value, 4> allOperands(op->getOperands());
    tile::ContractionOpAdaptor adaptor(allOperands);
    auto operands = adaptor.operands();
    InnerPredicate pred;
    return pred.match(operands[0].getType()) ||
           pred.match(operands[1].getType());
  }
};

template <typename InnerPredicate>
struct ComparandsAre : Matcher {
  bool match(Operation *op) const final {
    SmallVector<Value, 4> allOperands(op->getOperands());
    tile::ContractionOpAdaptor adaptor(allOperands);
    auto operands = adaptor.operands();
    InnerPredicate pred;
    return pred.match(operands[0].getType()) &&
           pred.match(operands[1].getType());
  }
};

template <typename InnerPredicate>
struct Not {
  bool match(Type type) const {
    InnerPredicate pred;
    return !pred.match(type);
  }
};

struct EltwiseFloat {
  bool match(Type type) const { return getElementType(type).isa<FloatType>(); }
};

struct EltwiseInteger {
  bool match(Type type) const {
    return getElementType(type).isa<IntegerType>();
  }
};

struct EltwiseSigned {
  bool match(Type type) const { return getElementType(type).isSignedInteger(); }
};

struct EltwiseUnsigned {
  bool match(Type type) const {
    return getElementType(type).isUnsignedInteger();
  }
};

struct FirstOperand {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    return operands.front();
  }
};

static Type promoteTypes(ConversionPatternRewriter &rewriter, Location loc,
                         ArrayRef<Value> operands, ArrayRef<Type> types,
                         SmallVectorImpl<Value> *into) {
  // First, determine the 'final' type that wins the promotion
  Type bestType;
  for (auto type : types) {
    bestType = tile::promoteTypes(bestType, type);
  }
  // Next, cast each operand to the 'final' type
  bool intoSigned = bestType.isSignedInteger();
  auto targetType = tile::toSignlessType(bestType);
  for (unsigned i = 0; i < operands.size(); i++) {
    auto dtype = types[i];
    auto operand = operands[i];
    auto castedValue =
        createCastOp(rewriter, loc, operand, dtype.isSignedInteger(),
                     targetType, intoSigned);
    into->push_back(castedValue);
  }
  return bestType;
}

struct NegIOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    auto zero = rewriter.create<mlir::ConstantIntOp>(loc, 0, resultType);
    auto neg = rewriter.create<mlir::SubIOp>(loc, zero, operands[0]);
    return neg.getResult();
  }
};

struct NotOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    // -(x + 1) = -1 - x
    auto negOne = rewriter.create<mlir::ConstantIntOp>(loc, -1, resultType);
    auto sub = rewriter.create<mlir::SubIOp>(loc, negOne, operands[0]);
    return sub.getResult();
  }
};

template <typename OpType>
struct StdOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands, types, &promoted);
    auto attrs = ArrayRef<NamedAttribute>{};
    auto resultTypes = llvm::makeArrayRef(resultType);
    auto op = rewriter.create<OpType>(loc, resultTypes, promoted, attrs);
    return op.getOperation()->getResult(0);
  }
};

struct SelectOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands.drop_front(), types.drop_front(),
                 &promoted);
    auto op = rewriter.create<mlir::SelectOp>(loc, operands[0], promoted[0],
                                              promoted[1]);
    return op.getResult();
  }
};

template <CmpFPredicate predicate>
struct CmpFloatOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands, types, &promoted);
    return rewriter
        .create<mlir::CmpFOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <CmpIPredicate predicate>
struct CmpIntOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands, types, &promoted);
    return rewriter
        .create<mlir::CmpIOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <CmpIPredicate signedPred, CmpIPredicate unsignedPred>
struct CmpIntInequalityOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    auto bestType = promoteTypes(rewriter, loc, operands, types, &promoted);
    auto predicate = bestType.isSignedInteger() ? signedPred : unsignedPred;
    return rewriter
        .create<mlir::CmpIOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <typename OpType>
struct LogicalOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    for (unsigned i = 0; i < operands.size(); ++i) {
      auto &operand = operands[i];
      auto fromType = operand.getType();
      if (auto floatType = fromType.dyn_cast<FloatType>()) {
        auto value = convertFloatUsingType(llvm::APFloat(0.0), floatType);
        auto zero =
            rewriter.create<mlir::ConstantFloatOp>(loc, value, floatType);
        promoted.push_back(
            rewriter
                .create<mlir::CmpFOp>(loc, CmpFPredicate::ONE, operand, zero)
                .getResult());
      } else if (auto intType = fromType.dyn_cast<IntegerType>()) {
        auto zero = rewriter.create<mlir::ConstantIntOp>(loc, 0, intType);
        promoted.push_back(
            rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::ne, operand, zero)
                .getResult());
      } else {
        llvm_unreachable("Unknown type for LogicalOp");
      }
    }
    auto attrs = ArrayRef<NamedAttribute>{};
    Type boolType = rewriter.getI1Type();
    auto resultTypes = llvm::makeArrayRef(boolType);
    auto op = rewriter.create<OpType>(loc, resultTypes, promoted, attrs);
    return op.getOperation()->getResult(0);
  }
};

struct LogicalNotOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    auto input = operands[0];
    auto fromType = input.getType();
    if (auto floatType = fromType.dyn_cast<FloatType>()) {
      auto value = convertFloatUsingType(llvm::APFloat(0.0), floatType);
      auto zero = rewriter.create<mlir::ConstantFloatOp>(loc, value, floatType);
      return rewriter.create<mlir::CmpFOp>(loc, CmpFPredicate::OEQ, input, zero)
          .getResult();
    } else if (auto intType = fromType.dyn_cast<IntegerType>()) {
      auto zero = rewriter.create<mlir::ConstantIntOp>(loc, 0, intType);
      return rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::eq, input, zero)
          .getResult();
    } else {
      llvm_unreachable("Unknown type for LogicalNotOp");
    }
  }
};

static Value createInit(OpBuilder &builder, Location loc, Type type,
                        AggregationKind agg) {
  if (auto floatType = type.dyn_cast<FloatType>()) {
    switch (agg) {
    case AggregationKind::add: {
      auto value = convertFloatUsingType(llvm::APFloat(0.0), floatType);
      return builder.create<mlir::ConstantFloatOp>(loc, value, floatType);
    }
    case AggregationKind::mul: {
      auto value = convertFloatUsingType(llvm::APFloat(1.0), floatType);
      return builder.create<mlir::ConstantFloatOp>(loc, value, floatType);
    }
    case AggregationKind::min: {
      auto value = llvm::APFloat::getInf(floatType.getFloatSemantics(), false);
      return builder.create<mlir::ConstantFloatOp>(loc, value, floatType);
    }
    case AggregationKind::max: {
      auto value = llvm::APFloat::getInf(floatType.getFloatSemantics(), true);
      return builder.create<mlir::ConstantFloatOp>(loc, value, floatType);
    }
    default:
      llvm_unreachable("Unsupported aggregation for createInit");
    }
  } else if (auto intType = type.dyn_cast<IntegerType>()) {
    switch (agg) {
    case AggregationKind::add:
      return builder.create<mlir::ConstantIntOp>(loc, 0, intType);
    case AggregationKind::mul:
      return builder.create<mlir::ConstantIntOp>(loc, 1, intType);
    case AggregationKind::min:
      return builder.create<mlir::ConstantIntOp>(
          loc, std::numeric_limits<int>::max(), intType);
    case AggregationKind::max:
      return builder.create<mlir::ConstantIntOp>(
          loc, std::numeric_limits<int>::min(), intType);
    default:
      llvm_unreachable("Unsupported aggregation for createInit");
    }
  }
  llvm_unreachable("Unknown type for createInit");
}

template <typename CmpOpBuilder>
struct CondOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    CmpOpBuilder cmpOpBuilder;
    auto cmp = cmpOpBuilder.create(rewriter, loc, resultType,
                                   operands.take_front(2), types.take_front(2));
    auto zero = createInit(rewriter, loc, resultType, AggregationKind::add);
    return rewriter.create<mlir::SelectOp>(loc, cmp, operands[2], zero)
        .getResult();
  }
};

static void updateAffineMap(Operation *in, const tile::PaddingInfo &padding) {
  auto accMap = in->getAttr("map").cast<AffineMapAttr>().getValue();
  assert(padding.lower.size() == accMap.getNumResults());
  SmallVector<AffineExpr, 4> newExprs;
  for (unsigned j = 0; j < accMap.getNumResults(); j++) {
    newExprs.push_back(accMap.getResult(j) + padding.lower[j]);
  }
  accMap = AffineMap::get(accMap.getNumDims(), 0, newExprs, in->getContext());
  in->setAttr("map", AffineMapAttr::get(accMap));
}

static Value
buildBroadcastLoad(OpBuilder &builder, Location loc, Value operand,
                   unsigned outRank,
                   Optional<tile::PaddingInfo> maybePadding = llvm::None) {
  // Handle scalar values
  if (!operand.getType().isa<MemRefType>()) {
    return operand;
  }
  // handle broadcasts
  auto body = builder.getBlock();
  auto operandType = operand.getType().cast<MemRefType>();
  assert(operandType.getRank() <= outRank && "result rank < operand rank");
  auto shape = operandType.getShape();
  SmallVector<Value, 8> operandIdxs(operandType.getRank());
  for (unsigned i = 0; i < operandType.getRank(); i++) {
    unsigned j = outRank - i - 1;
    unsigned k = operandType.getRank() - i - 1;
    if (shape[k] == 1) {
      operandIdxs[k] = builder.create<mlir::ConstantIndexOp>(loc, 0);
    } else {
      operandIdxs[k] = body->getArgument(j);
    }
  }
  auto loadOp = builder.create<pxa::PxaLoadOp>(loc, operand, operandIdxs);
  if (maybePadding)
    updateAffineMap(loadOp, *maybePadding);
  return loadOp;
}

static AtomicRMWKind convertAgg(AggregationKind agg, Type type) {
  switch (agg) {
  case AggregationKind::assign:
    return AtomicRMWKind::assign;
  case AggregationKind::add:
    if (type.isa<FloatType>()) {
      return AtomicRMWKind::addf;
    } else {
      return AtomicRMWKind::addi;
    }
  case AggregationKind::mul:
    if (type.isa<FloatType>()) {
      return AtomicRMWKind::mulf;
    } else {
      return AtomicRMWKind::muli;
    }
  case AggregationKind::min:
    if (type.isa<FloatType>()) {
      return AtomicRMWKind::minf;
    } else if (type.isSignedInteger()) {
      return AtomicRMWKind::mins;
    } else {
      return AtomicRMWKind::minu;
    }
  case AggregationKind::max:
    if (type.isa<FloatType>()) {
      return AtomicRMWKind::maxf;
    } else if (type.isSignedInteger()) {
      return AtomicRMWKind::maxs;
    } else {
      return AtomicRMWKind::maxu;
    }
  }
  llvm_unreachable("Invalid agg type in convertAgg");
}

static Value buildSimpleStore(OpBuilder &builder, Location loc, Value scalar,
                              Value memRef,
                              Optional<tile::PaddingInfo> maybePadding) {
  auto body = builder.getBlock();
  auto memRefType = memRef.getType().cast<MemRefType>();
  auto elementType = memRefType.getElementType();
  if (elementType != scalar.getType()) {
    scalar = createCastOp(builder, loc, scalar, false, elementType, false);
  }
  auto aggOp = AtomicRMWKind::assign;
  auto idMap = builder.getMultiDimIdentityMap(memRefType.getRank());
  auto storeOp = builder.create<pxa::PxaReduceOp>(loc, aggOp, scalar, memRef,
                                                  idMap, body->getArguments());
  if (maybePadding)
    updateAffineMap(storeOp, *maybePadding);
  return storeOp;
}

struct BufferAllocator {
  Value resultMemRef;
  RankedTensorType rankedTensorType;
  MemRefType memRefType;
  Type elementType;

  BufferAllocator(OpBuilder &builder, Operation *op, Type resultType) {
    // Gather some basic info
    TypeConverter typeConverter;
    auto loc = op->getLoc();
    rankedTensorType = getRankedTensorType(resultType);
    elementType = typeConverter.convertType(rankedTensorType.getElementType());
    auto originalShape = rankedTensorType.getShape();
    auto shape = llvm::to_vector<8>(originalShape);

    // If padding is detected, expand the shape to accomodate.
    auto maybePadding = tile::getPaddingInfo(op);
    if (maybePadding) {
      for (unsigned i = 0, e = shape.size(); i < e; ++i) {
        shape[i] += maybePadding->lower[i] + maybePadding->upper[i];
      }
    }

    // Make an allocation for the output
    memRefType = MemRefType::get(shape, elementType);
    resultMemRef = builder.create<AllocOp>(loc, memRefType);
    if (maybePadding) {
      auto initValue = createInit(builder, loc, elementType, maybePadding->agg);
      auto parallel = builder.create<AffineParallelOp>(
          loc,
          /*resultTypes=*/ArrayRef<Type>{memRefType},
          /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
          /*ranges=*/shape);
      auto parallelBuilder = parallel.getBodyBuilder();
      auto load =
          buildBroadcastLoad(parallelBuilder, loc, initValue, shape.size());
      auto stored = buildSimpleStore(parallelBuilder, loc, load, resultMemRef,
                                     llvm::None);
      parallelBuilder.create<AffineYieldOp>(loc, ValueRange{stored});
      resultMemRef = parallel.getResult(0);
    }
  }
};

struct PrngOpConversion : public OpConversionPattern<tile::PrngOp> {
  using OpConversionPattern<tile::PrngOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::PrngOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    tile::PrngOpAdaptor transformed(operands);
    BufferAllocator allocResult(rewriter, op.getOperation(),
                                op.result().getType());
    BufferAllocator stateResult(rewriter, op.getOperation(),
                                op.state().getType());
    rewriter.replaceOpWithNewOp<pxa::PrngOp>(
        op, allocResult.memRefType, stateResult.memRefType, transformed.state(),
        allocResult.resultMemRef, stateResult.resultMemRef);
    return success();
  }
};

template <typename FromOpType, typename IntoOpBuilder,
          typename Matcher = AlwaysTrue>
struct EltwiseOpConversion : public OpConversionPattern<FromOpType> {
  using OpConversionPattern<FromOpType>::OpConversionPattern;

  LogicalResult match(Operation *op) const final {
    Matcher pred;
    return pred(op);
  }

  void rewrite(FromOpType op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    BufferAllocator alloc(rewriter, op.getOperation(), op.result().getType());

    // Make a parallel for loop to fill the result
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{alloc.memRefType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*ranges=*/alloc.rankedTensorType.getShape());
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);

    // Create the loads
    SmallVector<Value, 4> scalars;
    for (size_t i = 0; i < operands.size(); i++) {
      auto maybePadding = tile::getPaddingInfo(
          op.getOperation()->getOperand(i).getDefiningOp());
      scalars.push_back(buildBroadcastLoad(rewriter, loc, operands[i],
                                           alloc.memRefType.getRank(),
                                           maybePadding));
    }

    // Create the standard op
    SmallVector<Type, 4> operandTypes;
    for (auto type : op.getOperation()->getOperandTypes()) {
      operandTypes.push_back(getElementType(type));
    }
    IntoOpBuilder intoOpBuilder;
    auto result = intoOpBuilder.create(rewriter, loc, alloc.elementType,
                                       scalars, operandTypes);

    // Create the store
    auto stored = buildSimpleStore(rewriter, loc, result, alloc.resultMemRef,
                                   tile::getPaddingInfo(op));
    rewriter.create<AffineYieldOp>(loc, ValueRange{stored});

    // Replace output with the newly allocated buffer
    rewriter.replaceOp(op, forOp.getResult(0));
  }
};

template <CombinationKind comboKind, typename ComboBuilder,
          typename Matcher = AlwaysTrue>
struct ContractionOpConversion
    : public OpConversionPattern<tile::ContractionOp> {
  using OpConversionPattern<tile::ContractionOp>::OpConversionPattern;

  LogicalResult match(Operation *op) const final {
    if (auto cionOp = dyn_cast<tile::ContractionOp>(op)) {
      if (cionOp.combo() != comboKind) {
        return failure();
      }
      if (!cionOp.lowerBounds().hasValue() ||
          !cionOp.upperBounds().hasValue()) {
        cionOp.emitError("contraction bounds must be computed");
        return failure();
      }
      Matcher pred;
      return pred(cionOp);
    }
    return failure();
  }

  void rewrite(tile::ContractionOp op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    try {
      tryRewrite(op, operands, rewriter);
    } catch (const std::exception &ex) {
      op.emitError(ex.what());
    }
  }

  void tryRewrite(tile::ContractionOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const {
    // Create an adaptor
    tile::ContractionOpAdaptor cionAdaptor(operands);
    auto cionOperands = cionAdaptor.operands();

    auto loc = op.getLoc();
    BufferAllocator alloc(rewriter, op.getOperation(), op.result().getType());

    // Do initialization
    auto shape = alloc.rankedTensorType.getShape();
    auto parallel = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{alloc.memRefType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*ranges=*/shape);
    auto parallelBuilder = parallel.getBodyBuilder();
    auto maybePadding = tile::getPaddingInfo(op.init().getDefiningOp());
    auto load = buildBroadcastLoad(parallelBuilder, loc, cionAdaptor.init(),
                                   shape.size(), maybePadding);
    auto store = buildSimpleStore(parallelBuilder, loc, load,
                                  alloc.resultMemRef, tile::getPaddingInfo(op));
    if (maybePadding)
      updateAffineMap(store.getDefiningOp(), *maybePadding);
    parallelBuilder.create<AffineYieldOp>(loc, ValueRange{store});
    auto filled = parallel.getResult(0);

    // Determine lower and upper bounds.
    SmallVector<AffineExpr, 8> ubExprs;
    auto lowerBounds = op.lowerBounds().getValue();
    auto upperBounds = op.upperBounds().getValue();
    assert(lowerBounds.getNumResults() == upperBounds.getNumResults() &&
           "mismatched dims for lower and upper bounds");
    for (unsigned i = 0; i < lowerBounds.getNumResults(); i++) {
      auto ubExpr = upperBounds.getResult(i) + 1;
      auto upper = ubExpr.cast<AffineConstantExpr>().getValue();
      ubExprs.push_back(rewriter.getAffineConstantExpr(upper));
    }

    auto ubMap = AffineMap::get(0, 0, {ubExprs}, op.getContext());
    // Make the outer loops
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{alloc.memRefType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*lbMap=*/op.lowerBounds().getValue(),
        /*lbArgs=*/ArrayRef<Value>{},
        /*ubMap=*/ubMap,
        /*ubArgs=*/ArrayRef<Value>{});

    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);
    auto idxs = body->getArguments();

    // add constraints
    if (op.cons()) {
      auto cons = op.cons().getValue();
      auto ifOp = rewriter.create<AffineIfOp>(loc, TypeRange{alloc.memRefType},
                                              cons, idxs, true);
      rewriter.create<AffineYieldOp>(loc, ifOp.getOperation()->getResults());
      rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
      rewriter.create<AffineYieldOp>(loc, filled);
      rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
    }

    // Create the loads + casts
    SmallVector<Value, 4> scalars;
    auto srcs = op.srcs().getValue();
    for (size_t i = 0; i < srcs.size(); i++) {
      auto operand = cionOperands[i];
      if (!operand.getType().isa<MemRefType>()) {
        scalars.push_back(operand);
      } else {
        auto map = srcs[i].cast<AffineMapAttr>().getValue();
        auto loadOp = rewriter.create<pxa::PxaLoadOp>(loc, operand, map, idxs);
        auto maybePadding =
            tile::getPaddingInfo(op.operands()[i].getDefiningOp());
        if (maybePadding)
          updateAffineMap(loadOp, *maybePadding);
        scalars.push_back(loadOp);
      }
    }

    // Do the combination op
    ComboBuilder comboBuilder;
    SmallVector<Type, 4> operandTypes;
    for (auto type : op.operands().getTypes()) {
      operandTypes.push_back(getElementType(type));
    }
    auto combined = comboBuilder.create(rewriter, loc, alloc.elementType,
                                        scalars, operandTypes);

    // Create the store
    auto resultMap = op.sink();
    pxa::PxaReduceOp reduceOp;
    auto agg = convertAgg(op.agg(), alloc.elementType);
    if (resultMap.isEmpty()) {
      SmallVector<Value, 0> emptyIdxs;
      reduceOp = rewriter.create<pxa::PxaReduceOp>(loc, agg, combined, filled,
                                                   resultMap, emptyIdxs);
    } else {
      reduceOp = rewriter.create<pxa::PxaReduceOp>(loc, agg, combined, filled,
                                                   resultMap, idxs);
    }
    maybePadding = tile::getPaddingInfo(op);
    if (maybePadding)
      updateAffineMap(reduceOp, *maybePadding);
    rewriter.create<AffineYieldOp>(loc, ValueRange{reduceOp});

    // Replace the op
    rewriter.replaceOp(op, forOp.getResult(0));
  }
};

struct ArgSortOpConversion : public OpConversionPattern<tile::ArgSortOp> {
  using OpConversionPattern<tile::ArgSortOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ArgSortOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IVLOG(3, "ArgSortOpConversion::matchAndRewrite");
    tile::ArgSortOpAdaptor adaptor(operands);
    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto indexType = rewriter.getIndexType();

    // Axis represents one of the tensor dimensions.
    int64_t axisAttr = op.axis().getSExtValue();

    // Special case value -1 indicates the last axis.
    Value tensor = adaptor.tensor();
    auto shape = tensor.getType().cast<MemRefType>().getShape();
    size_t tensorDims = shape.size();
    if (axisAttr < 0) {
      axisAttr += static_cast<int64_t>(tensorDims);
    }
    size_t axis = static_cast<size_t>(axisAttr);
    if (axisAttr < 0 || axis >= tensorDims) {
      return failure();
    }
    auto elementType = tensor.getType().cast<MemRefType>().getElementType();

    // Allocate an output tensor to contain the sorted argument indices.
    auto resultType = MemRefType::get(shape, i32Type);
    Value result = rewriter.create<AllocOp>(loc, resultType).getResult();

    llvm::SmallVector<Type, 4> resultTypes;
    resultTypes.push_back(resultType);
    llvm::SmallVector<NamedAttribute, 4> attrs;
    llvm::SmallVector<Value, 2> layerInputs{result, tensor};
    auto layerOp =
        rewriter.create<layer::BoxOp>(loc, "argsort", layerInputs, resultTypes,
                                      rewriter.getDictionaryAttr(attrs));
    rewriter.setInsertionPointToStart(&layerOp.body().front());

    // Inside the box, the output tensor comes from the first argument,
    // and the input data tensor comes from the next. Outputs must come
    // first in the layer.box operands list.
    result = layerOp.body().getArguments()[0];
    tensor = layerOp.body().getArguments()[1];

    auto icon0 =
        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)).getResult();
    auto icon1 =
        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1)).getResult();

    // Create a loop nest over all the dimensions, including the one we are
    // sorting on. We will use a single iteration for the sort axis, since the
    // body of the nest will contain the sorting loop, but we will keep the
    //  number of IVs equal to the tensor rank to simplify accounting.
    SmallVector<Value, 4> ops;
    for (size_t i = 0; i < tensorDims; ++i) {
      if (i == axis) {
        ops.push_back(icon0);
        continue;
      }
      auto limit =
          rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(shape[i]))
              .getResult();
      auto loop = rewriter.create<scf::ForOp>(loc, icon0, limit, icon1);
      ops.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    auto iconN =
        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(shape[axis]))
            .getResult();

    // For loads and stores from the minIdxVar and minValVar, we'll need a
    // single, zero-value index, because that's the only way to create a memref
    SmallVector<Value, 1> zeroIndex;
    zeroIndex.push_back(icon0);

    // Initialize result tensor using index values in ascending order
    {
      auto initLoop = rewriter.create<scf::ForOp>(loc, icon0, iconN, icon1);
      auto initIV = initLoop.getInductionVar();
      rewriter.setInsertionPointToStart(initLoop.getBody());
      // The induction var is the index value across the sort axis.
      // Write the value of the index to its position in the sorted output.
      // This will be our initial argument index into the source tensor.
      // Combine the index value with the loop dimension indexes to create the
      // destination affine map.
      auto indexVal =
          rewriter.create<IndexCastOp>(loc, initIV, i32Type).getResult();
      ops[axis] = initIV;
      rewriter.create<StoreOp>(loc, indexVal, result, ops);
      rewriter.setInsertionPointAfter(initLoop);
    }

    // Stable selection sort:
    // for (int i = 0; i < n-1; i++) {
    //     int min_idx = i;
    //     for (int j = i+1; j < n; j++) {
    //         if (arr[j] < arr[min_idx]) {
    //             min_idx = j;
    //         }
    //     }
    //     int min_val = arr[min_idx];
    //     while (min_idx > i) {
    //         arr[min_idx] = arr[min_idx - 1];
    //         min_idx--;
    //     }
    //     arr[i] = min_val;
    // }

    // Build inner sorting loop
    if (shape[axis] > 1) {
      auto sortUB = rewriter.create<ConstantOp>(
          loc, rewriter.getIndexAttr(shape[axis] - 1));
      auto sortLoop = rewriter.create<scf::ForOp>(loc, icon0, sortUB, icon1);
      auto sortIV = sortLoop.getInductionVar();
      rewriter.setInsertionPointToStart(sortLoop.getBody());

      // Get the value associated with the current minimum index position.
      // create a MemRefType which is a single-element index
      MemRefType minIdxVarType = MemRefType::get({1}, indexType);
      auto minIdxVar = rewriter.create<AllocaOp>(loc, minIdxVarType);
      rewriter.create<StoreOp>(loc, sortIV, minIdxVar, zeroIndex);

      ops[axis] = sortIV;
      auto minValIdxInt = rewriter.create<LoadOp>(loc, result, ops).getResult();
      auto minValIdx =
          rewriter.create<IndexCastOp>(loc, minValIdxInt, indexType)
              .getResult();
      ops[axis] = minValIdx;
      Value minVal = rewriter.create<LoadOp>(loc, tensor, ops).getResult();
      MemRefType minValVarType = MemRefType::get({1}, elementType);
      auto minValVar = rewriter.create<AllocaOp>(loc, minValVarType);
      rewriter.create<StoreOp>(loc, minVal, minValVar, zeroIndex);

      // Iterate over the remaining elements, looking for a smaller value.
      auto minLB = rewriter.create<AddIOp>(loc, sortIV, icon1);
      auto minLoop = rewriter.create<scf::ForOp>(loc, minLB, iconN, icon1);
      auto minIV = minLoop.getInductionVar();
      rewriter.setInsertionPointToStart(minLoop.getBody());

      // Get the comparison value for the search iterator position.
      ops[axis] = minIV;
      auto compValIdxInt =
          rewriter.create<LoadOp>(loc, result, ops).getResult();
      auto compValIdx =
          rewriter.create<IndexCastOp>(loc, compValIdxInt, indexType)
              .getResult();
      ops[axis] = compValIdx;
      Value compVal = rewriter.create<LoadOp>(loc, tensor, ops).getResult();

      // What is the current minimum value, stored in the min val var?
      minVal = rewriter.create<LoadOp>(loc, minValVar, zeroIndex).getResult();
      // Is compVal smaller than minVal? If so, update the allocs
      Value orderPred;
      orderPred = convertCmpOp(loc, rewriter, compVal, minVal, elementType,
                               op.direction());

      auto ifReorder = rewriter.create<scf::IfOp>(loc, orderPred, false);
      rewriter.setInsertionPointToStart(&ifReorder.thenRegion().front());
      // store minIV -> minIdxVar
      rewriter.create<StoreOp>(loc, minIV, minIdxVar, zeroIndex);
      // store compVal -> minValVar
      rewriter.create<StoreOp>(loc, compVal, minValVar, zeroIndex);
      // End the conditional block. We would set the insertion point after
      // the ifReorder block, except that we are about to...
      // End the inner sort loop, which found the smallest value in the
      // unsorted region, by setting the insertion point after its block.
      rewriter.setInsertionPointAfter(minLoop);

      auto finalMinPos =
          rewriter.create<LoadOp>(loc, minIdxVar, zeroIndex).getResult();
      ops[axis] = finalMinPos;
      auto finalMinIdx = rewriter.create<LoadOp>(loc, result, ops).getResult();

      // Move every element between [sortIV,finalMinPos) a step forward
      // and then move the minimum index to the head to keep it stable.
      auto moveLoop =
          rewriter.create<scf::ForOp>(loc, sortIV, finalMinPos, icon1);
      auto moveIV = moveLoop.getInductionVar();
      rewriter.setInsertionPointToStart(moveLoop.getBody());
      moveIV = rewriter.create<SubIOp>(loc, finalMinPos, moveIV);
      moveIV = rewriter.create<AddIOp>(loc, moveIV, sortIV);
      auto moveNext = rewriter.create<SubIOp>(loc, moveIV, icon1);
      ops[axis] = moveNext;
      auto moveNextVal = rewriter.create<LoadOp>(loc, result, ops).getResult();
      ops[axis] = moveIV;
      rewriter.create<StoreOp>(loc, moveNextVal, result, ops);
      rewriter.setInsertionPointAfter(moveLoop);
      ops[axis] = sortIV;
      rewriter.create<StoreOp>(loc, finalMinIdx, result, ops);

      // The minimum remaining value is now at the end of the sorted region.
      // Advance to the next minimum in the remaining unsorted region.
      rewriter.setInsertionPointAfter(sortLoop);
    }

    // for future reference, when N is a power of 2
    // bitonic sort:
    // for (int k = 2; k <= N; k = 2 * k) {
    //   for (int j = k >> 1; j > 0; j = j >> 1) {
    //     for (int i = 0; i < N; i++) {
    //       int ixj = i ^ j;
    //       if (ixj > i) {
    //         if ((i & k) == 0 && a[i] > a[ixj]) {
    //           swap(a[i], a[ixj]);
    //         }
    //         if ((i & k) != 0 && a[i] < a[ixj]) {
    //           swap(a[i], a[ixj]);
    //         }
    //       }
    //     }
    //   }
    // }

    rewriter.setInsertionPointToEnd(&layerOp.body().back());
    rewriter.create<layer::ReturnOp>(loc, ArrayRef<Value>{result});
    rewriter.replaceOp(op, layerOp.getResult(0));
    return success();
  }

  Value convertCmpOp(Location loc, ConversionPatternRewriter &rewriter,
                     Value compVal, Value minVal, Type type,
                     tile::SortDirection dir) const {
    if (type.isa<FloatType>()) {
      CmpFPredicate pred;
      switch (dir) {
      case tile::SortDirection::asc:
        pred = CmpFPredicate::OLT;
        break;
      case tile::SortDirection::desc:
        pred = CmpFPredicate::OGT;
        break;
      }
      return rewriter.create<mlir::CmpFOp>(loc, pred, compVal, minVal)
          .getResult();
    }
    CmpIPredicate pred;
    if (type.isSignedInteger()) {
      switch (dir) {
      case tile::SortDirection::asc:
        pred = CmpIPredicate::slt;
        break;
      case tile::SortDirection::desc:
        pred = CmpIPredicate::sgt;
        break;
      }
    } else {
      switch (dir) {
      case tile::SortDirection::asc:
        pred = CmpIPredicate::ult;
        break;
      case tile::SortDirection::desc:
        pred = CmpIPredicate::ugt;
        break;
      }
    }
    return rewriter.create<mlir::CmpIOp>(loc, pred, compVal, minVal)
        .getResult();
  }
};

struct GatherOpConversion : public OpConversionPattern<tile::GatherOp> {
  using OpConversionPattern<tile::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::GatherOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Create an adaptor, to interpret the operands
    tile::GatherOpAdaptor adaptor(operands);

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    // Input values
    auto tensor = adaptor.tensor();
    // Index values for the last dimension
    // this is a one-dimensional array of integers
    auto indices = adaptor.indices();

    TypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result().getType());
    auto memrefType = resultType.cast<MemRefType>();

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, memrefType).getResult();

    // We need an array of int64_t representing the results tensor's dims
    ArrayRef<int64_t> size = memrefType.getShape();

    auto loop = rewriter.create<AffineParallelOp>(
        loc, ArrayRef<Type>{memrefType},
        ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign}, size);
    rewriter.setInsertionPointToStart(loop.getBody());

    // Create an affine map for loading the index, using the leading counters
    size_t axis = *(op.axis().getRawData());
    size_t batchDims = *(op.batchDims().getRawData());
    auto idxShape = indices.getType().cast<MemRefType>().getShape();
    size_t idxDims = indices.getType().cast<MemRefType>().getShape().size();
    auto idxLoadMap = AffineMap::getMultiDimIdentityMap(idxDims, ctx);

    // Create default source map
    size_t dstDims = size.size();
    std::vector<Value> srcOps;
    Value interpVal;
    switch (op.mode()) {
    case GatherMode::normal: {
      auto idxLoadOps = loop.getIVs().slice(axis, idxDims);
      auto idx =
          rewriter.create<pxa::PxaLoadOp>(loc, indices, idxLoadMap, idxLoadOps)
              .getResult();
      for (size_t i = 0; i < axis; ++i) {
        srcOps.push_back(loop.getIVs()[i]);
      }

      for (size_t i = axis + idxDims - 1; i < dstDims; ++i) {
        srcOps.push_back(loop.getIVs()[i]);
      }

      // Create std ops for 1D interpolation
      if (idx.getType().isa<FloatType>()) {
        switch (op.interpolationMode()) {
        case InterpolationMode::nearest:
          interpVal = buildNearestInterpolationOps(
              loc, rewriter, tensor, idx, srcOps, axis, op.nearestMode());
          break;
        case InterpolationMode::linear:
          interpVal = buildLinearInterpolationOps(loc, rewriter, tensor, idx,
                                                  srcOps, axis);
          break;
        case InterpolationMode::cubic:
          interpVal = buildCubicInterpolationOps(
              loc, rewriter, tensor, idx, srcOps, axis,
              op.cubeCoeffAttr().getValueAsDouble());
          break;
        default:
          llvm_unreachable("Unsupported InterpolationMode");
        }
      } else {
        if (!idx.getType().isa<IndexType>()) {
          auto indexType = rewriter.getIndexType();
          // Cast from whatever integer type it has to index type
          idx = rewriter.create<mlir::IndexCastOp>(loc, idx, indexType)
                    .getResult();
        }
        srcOps.at(axis) = idx;
        interpVal = rewriter.create<mlir::LoadOp>(loc, tensor, srcOps);
      }
    } break;
    case GatherMode::nd: {
      std::vector<Value> idxs, combIdx(idxDims);
      for (size_t i = 0; i < idxDims - 1; ++i) {
        combIdx[i] = loop.getIVs()[i];
      }
      for (int64_t i = 0; i < idxShape[idxDims - 1]; ++i) {
        combIdx[idxDims - 1] = rewriter.create<mlir::ConstantIndexOp>(loc, i);
        auto idx =
            rewriter.create<pxa::PxaLoadOp>(loc, indices, idxLoadMap, combIdx)
                .getResult();
        if (!idx.getType().isa<IndexType>()) {
          auto indexType = rewriter.getIndexType();
          idx = rewriter.create<mlir::IndexCastOp>(loc, idx, indexType)
                    .getResult();
        }
        idxs.push_back(idx);
      }
      for (size_t i = 0; i < batchDims; ++i) {
        srcOps.push_back(loop.getIVs()[i]);
      }
      srcOps.insert(srcOps.end(), idxs.begin(), idxs.end());
      for (size_t i = idxDims - 1; i < dstDims; ++i) {
        srcOps.push_back(loop.getIVs()[i]);
      }
      interpVal = rewriter.create<mlir::LoadOp>(loc, tensor, srcOps);
    } break;
    default:
      llvm_unreachable("unrecognized gather mode");
    }

    // Create a destination map using all of the dimensions
    auto dstStoreMap = AffineMap::getMultiDimIdentityMap(dstDims, ctx);

    // Create a destination map from the whole loop
    auto stored = rewriter.create<pxa::PxaReduceOp>(loc, AtomicRMWKind::assign,
                                                    interpVal, resultMemRef,
                                                    dstStoreMap, loop.getIVs());
    rewriter.create<AffineYieldOp>(loc, ArrayRef<Value>{stored.getResult()});
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }

  Value buildNearestInterpolationOps(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     Value tensor, Value idx,
                                     std::vector<Value> &srcOps, size_t axis,
                                     NearestMode nearestMode) const {
    auto idxType = rewriter.getIndexType();
    auto i32Type = rewriter.getI32Type();
    auto bounds = GetIndexBounds(loc, rewriter, tensor, axis, i32Type);
    switch (nearestMode) {
    case NearestMode::round_prefer_floor: {
      auto cmp = isHalfWayFloat(loc, rewriter, idx);
      auto floor = floorFPToSI(loc, rewriter, idx, i32Type);
      auto round = roundFPToSI(loc, rewriter, idx, i32Type);
      idx = rewriter.create<mlir::SelectOp>(loc, cmp, floor, round).result();
    } break;
    case NearestMode::round_prefer_ceil: {
      auto cmp = isHalfWayFloat(loc, rewriter, idx);
      auto ceil = ceilFPToSI(loc, rewriter, idx, i32Type);
      auto round = roundFPToSI(loc, rewriter, idx, i32Type);
      idx = rewriter.create<mlir::SelectOp>(loc, cmp, ceil, round).result();
    } break;
    case NearestMode::floor:
      idx = floorFPToSI(loc, rewriter, idx, i32Type);
      break;
    case NearestMode::ceil:
      idx = ceilFPToSI(loc, rewriter, idx, i32Type);
      break;
    case NearestMode::simple:
      idx = rewriter.create<mlir::FPToSIOp>(loc, idx, i32Type).getResult();
      break;
    default:
      llvm_unreachable("Unsupported NearestMode");
    }
    idx = checkIntOutOfBounds(loc, rewriter, idx, bounds[0], bounds[1]);
    idx = rewriter.create<mlir::IndexCastOp>(loc, idx, idxType).getResult();
    srcOps.at(axis) = idx;
    return rewriter.create<mlir::LoadOp>(loc, tensor, srcOps);
  }

  Value buildLinearInterpolationOps(Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    Value tensor, Value idx,
                                    std::vector<Value> &srcOps,
                                    size_t axis) const {
    auto idxType = rewriter.getIndexType();
    auto i32Type = rewriter.getI32Type();
    auto elementType = tensor.getType().cast<MemRefType>().getElementType();
    auto bounds = GetIndexBounds(loc, rewriter, tensor, axis, i32Type);
    auto cst1F =
        rewriter
            .create<mlir::ConstantOp>(loc, elementType,
                                      rewriter.getFloatAttr(elementType, 1.0))
            .getResult();

    // Calculate interpolation nodes: floor and ceil
    auto floor = floorFPToSI(loc, rewriter, idx, i32Type);
    auto ceil = ceilFPToSI(loc, rewriter, idx, i32Type);
    floor = checkIntOutOfBounds(loc, rewriter, floor, bounds[0], bounds[1]);
    ceil = checkIntOutOfBounds(loc, rewriter, ceil, bounds[0], bounds[1]);
    floor = rewriter.create<mlir::IndexCastOp>(loc, floor, idxType).getResult();
    ceil = rewriter.create<mlir::IndexCastOp>(loc, ceil, idxType).getResult();

    // Load sample data g0 and g1 at interpolation nodes
    srcOps.at(axis) = ceil;
    auto g0 = rewriter.create<mlir::LoadOp>(loc, tensor, srcOps).getResult();
    srcOps.at(axis) = floor;
    auto g1 = rewriter.create<mlir::LoadOp>(loc, tensor, srcOps).getResult();

    // Calculate coefficients of g0 and g1
    auto floorF =
        rewriter.create<mlir::FloorFOp>(loc, elementType, idx).getResult();
    auto c0 = rewriter.create<mlir::SubFOp>(loc, idx, floorF).getResult();
    auto c1 = rewriter.create<mlir::SubFOp>(loc, cst1F, c0).getResult();

    // Return interpolation result (result = c0*g0 + c1*g1)
    auto p0 = rewriter.create<mlir::MulFOp>(loc, c0, g0).getResult();
    auto p1 = rewriter.create<mlir::MulFOp>(loc, c1, g1).getResult();
    return rewriter.create<mlir::AddFOp>(loc, p0, p1).getResult();
  }

  Value buildCubicInterpolationOps(Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   Value tensor, Value idx,
                                   std::vector<Value> &srcOps, size_t axis,
                                   double cubicCoeff) const {
    // Follow the algorithm used in ngraph cubic interpolation (also see, e.g.
    // [article](https://ieeexplore.ieee.org/document/1163711/).

    auto idxType = rewriter.getIndexType();
    auto i32Type = rewriter.getI32Type();
    auto elementType = tensor.getType().cast<MemRefType>().getElementType();
    auto bounds = GetIndexBounds(loc, rewriter, tensor, axis, i32Type);

    // Create constant a (cubeCoeff)
    auto a = rewriter
                 .create<mlir::ConstantOp>(
                     loc, rewriter.getF64Type(),
                     FloatAttr::get(rewriter.getF64Type(), cubicCoeff))
                 .getResult();
    if (!elementType.isa<mlir::Float64Type>()) {
      a = rewriter.create<mlir::FPTruncOp>(loc, elementType, a);
    }

    // Create integer constants
    SmallVector<Value, 4> cstI;
    for (auto i = 0; i <= 2; i++) {
      auto cstOp = rewriter.create<mlir::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, i));
      cstI.push_back(cstOp.getResult());
    }

    // Create float constants
    SmallVector<Value, 4> cstF;
    for (auto i = 0; i <= 3; i++) {
      auto cstOp = rewriter.create<mlir::ConstantOp>(
          loc, elementType, rewriter.getFloatAttr(elementType, i));
      cstF.push_back(cstOp.getResult());
    }

    // Calculate interpolation nodes x
    auto floorI = floorFPToSI(loc, rewriter, idx, i32Type);
    auto ceilI = ceilFPToSI(loc, rewriter, idx, i32Type);
    SmallVector<Value, 4> x;
    x.push_back(
        rewriter.create<mlir::SubIOp>(loc, floorI, cstI[1]).getResult());
    x.push_back(floorI);
    x.push_back(ceilI);
    x.push_back(rewriter.create<mlir::AddIOp>(loc, ceilI, cstI[1]).getResult());

    // Load sample data g at interpolation nodes
    SmallVector<Value, 4> g;
    for (size_t i = 0; i < x.size(); i++) {
      x[i] = checkIntOutOfBounds(loc, rewriter, x[i], bounds[0], bounds[1]);
      x[i] = rewriter.create<mlir::IndexCastOp>(loc, x[i], idxType).getResult();
      srcOps.at(axis) = x[i];
      auto loadOp = rewriter.create<mlir::LoadOp>(loc, tensor, srcOps);
      g.push_back(loadOp.getResult());
    }

    // Calculate intermediate terms
    SmallVector<Value, 4> p;
    auto floorF =
        rewriter.create<mlir::FloorFOp>(loc, idx.getType(), idx).getResult();
    auto s = rewriter.create<mlir::SubFOp>(loc, idx, floorF).getResult();
    auto s2 = rewriter.create<mlir::MulFOp>(loc, s, s).getResult();
    auto s3 = rewriter.create<mlir::MulFOp>(loc, s2, s).getResult();
    auto s_a = rewriter.create<mlir::MulFOp>(loc, a, s).getResult();
    auto s2_a = rewriter.create<mlir::MulFOp>(loc, a, s2).getResult();
    auto s3_a = rewriter.create<mlir::MulFOp>(loc, a, s3).getResult();
    auto s3_a2 = rewriter.create<mlir::AddFOp>(loc, a, cstF[2]).getResult();
    s3_a2 = rewriter.create<mlir::MulFOp>(loc, s3_a2, s3).getResult();
    auto s2_a3 = rewriter.create<mlir::AddFOp>(loc, a, cstF[3]).getResult();
    s2_a3 = rewriter.create<mlir::MulFOp>(loc, s2_a3, s2).getResult();
    auto s2_2a3 = rewriter.create<mlir::AddFOp>(loc, a, a).getResult();
    s2_2a3 = rewriter.create<mlir::AddFOp>(loc, s2_2a3, cstF[3]).getResult();
    s2_2a3 = rewriter.create<mlir::MulFOp>(loc, s2_2a3, s2).getResult();

    // Calculate 4 terms at interpolation nodes
    p.push_back(rewriter.create<mlir::MulFOp>(loc, s2_a, cstF[2]).getResult());
    p[0] = rewriter.create<mlir::SubFOp>(loc, s3_a, p[0]).getResult();
    p[0] = rewriter.create<mlir::AddFOp>(loc, p[0], s_a).getResult();

    p.push_back(rewriter.create<mlir::SubFOp>(loc, s3_a2, s2_a3).getResult());
    p[1] = rewriter.create<mlir::AddFOp>(loc, p[1], cstF[1]).getResult();

    p.push_back(rewriter.create<mlir::SubFOp>(loc, s2_2a3, s3_a2).getResult());
    p[2] = rewriter.create<mlir::SubFOp>(loc, p[2], s_a).getResult();

    p.push_back(rewriter.create<mlir::SubFOp>(loc, s2_a, s3_a).getResult());

    for (size_t i = 0; i < p.size(); i++) {
      p[i] = rewriter.create<mlir::MulFOp>(loc, p[i], g[i]).getResult();
    }

    // Return interpolation result (result = p0 + p1 + p2 + p3)
    auto r = rewriter.create<mlir::AddFOp>(loc, p[0], p[1]).getResult();
    r = rewriter.create<mlir::AddFOp>(loc, r, p[2]).getResult();
    return rewriter.create<mlir::AddFOp>(loc, r, p[3]).getResult();
  }

  SmallVector<Value, 2> GetIndexBounds(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Value tensor, size_t axis,
                                       IntegerType integerType) const {
    // Return lower and upper bounds of a tensor at an axis
    SmallVector<Value, 2> bounds;
    auto axisLen = tensor.getType().cast<MemRefType>().getShape()[axis];
    auto lower = rewriter.create<mlir::ConstantOp>(
        loc, integerType, rewriter.getIntegerAttr(integerType, 0));
    auto upper = rewriter.create<mlir::ConstantOp>(
        loc, integerType, rewriter.getIntegerAttr(integerType, axisLen - 1));
    bounds.push_back(lower.getResult());
    bounds.push_back(upper.getResult());
    return bounds;
  }

  Value checkIntOutOfBounds(Location loc, ConversionPatternRewriter &rewriter,
                            Value value, Value lowerBound,
                            Value upperBound) const {
    // Check if a mlir::IntegerType value is out of bounds. If it is, set it to
    // lower/upper bound.
    auto cmpLower = rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::slt,
                                                  value, lowerBound);
    auto cmpUpper = rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::slt,
                                                  value, upperBound);
    value = rewriter.create<mlir::SelectOp>(loc, cmpLower, lowerBound, value)
                .result();
    value = rewriter.create<mlir::SelectOp>(loc, cmpUpper, value, upperBound)
                .result();
    return value;
  }

  Value isHalfWayFloat(Location loc, ConversionPatternRewriter &rewriter,
                       Value value) const {
    // Check if the fractional part of a float value is 0.5
    auto floatType = value.getType();
    auto half = rewriter.create<mlir::ConstantOp>(
        loc, floatType, rewriter.getFloatAttr(floatType, 0.5));
    auto floor =
        rewriter.create<mlir::FloorFOp>(loc, floatType, value).getResult();
    auto floorPlusHalf = rewriter.create<mlir::AddFOp>(loc, floor, half);
    return rewriter
        .create<mlir::CmpFOp>(loc, CmpFPredicate::OEQ, value, floorPlusHalf)
        .getResult();
  }

  Value ceilFPToSI(Location loc, ConversionPatternRewriter &rewriter,
                   Value value, IntegerType integerType) const {
    auto ceilFloat =
        rewriter.create<mlir::CeilFOp>(loc, value.getType(), value).getResult();
    return rewriter.create<mlir::FPToSIOp>(loc, integerType, ceilFloat)
        .getResult();
  }

  Value floorFPToSI(Location loc, ConversionPatternRewriter &rewriter,
                    Value value, IntegerType integerType) const {
    auto floorFloat =
        rewriter.create<mlir::FloorFOp>(loc, value.getType(), value)
            .getResult();
    return rewriter.create<mlir::FPToSIOp>(loc, integerType, floorFloat)
        .getResult();
  }

  Value roundFPToSI(Location loc, ConversionPatternRewriter &rewriter,
                    Value value, IntegerType integerType) const {
    auto floatType = value.getType();
    auto half = rewriter.create<mlir::ConstantOp>(
        loc, floatType, rewriter.getFloatAttr(floatType, 0.5));
    auto valuePlusHalf = rewriter.create<mlir::AddFOp>(loc, value, half);
    return floorFPToSI(loc, rewriter, valuePlusHalf, integerType);
  }
};

struct IndexOpConversion : public OpConversionPattern<tile::IndexOp> {
  using OpConversionPattern<tile::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::IndexOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType =
        typeConverter.convertType(op.result().getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Make a parallel for loop to fill the result
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{resultType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*ranges=*/resultType.getShape());
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);
    auto idxs = body->getArguments();

    // Load the index value
    // TODO: add check that axis is within range in verifier
    auto axis = op.axis().getZExtValue();
    auto map = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)});
    auto apply = rewriter.create<mlir::AffineApplyOp>(loc, map, idxs[axis]);

    // Create the store
    auto cast = rewriter.create<mlir::IndexCastOp>(loc, apply,
                                                   rewriter.getIntegerType(32));
    auto stored = buildSimpleStore(rewriter, loc, cast, resultMemRef,
                                   tile::getPaddingInfo(op));
    rewriter.create<AffineYieldOp>(loc, ValueRange{stored});

    // Replace the op
    rewriter.replaceOp(op, forOp.getResult(0));

    return success();
  }
};

struct ReshapeOpConversion : public OpConversionPattern<tile::ReshapeOp> {
  using OpConversionPattern<tile::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ReshapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Create an adaptor, to interpret the operands
    tile::ReshapeOpAdaptor adaptor(operands);

    auto tensor = adaptor.tensor();

    TypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result().getType());

    rewriter.replaceOpWithNewOp<stdx::ReshapeOp>(op, resultType, tensor);
    return success();
  }
};

struct ShapeOpConversion : public OpConversionPattern<tile::ShapeOp> {
  using OpConversionPattern<tile::ShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ShapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Create an adaptor
    tile::ShapeOpAdaptor adaptor(operands);

    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType =
        typeConverter.convertType(op.result().getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto memRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Populate the buffer with the shape dims
    auto operandType = adaptor.tensor().getType().cast<MemRefType>();
    auto aggOp = AtomicRMWKind::assign;
    for (unsigned i = 0; i < operandType.getRank(); i++) {
      auto dim = rewriter.create<mlir::DimOp>(loc, adaptor.tensor(), i);
      auto cast = rewriter.create<mlir::IndexCastOp>(
          loc, dim, rewriter.getIntegerType(32));
      auto map = rewriter.getConstantAffineMap(i);
      memRef = rewriter.create<pxa::PxaReduceOp>(loc, aggOp, cast, memRef, map,
                                                 ArrayRef<Value>{});
    }

    // Replace the op
    rewriter.replaceOp(op, memRef);

    return success();
  }
};

struct ScatterOpConversion : public OpConversionPattern<tile::ScatterOp> {
  using OpConversionPattern<tile::ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ScatterOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Helpful explanation of scatter from tensorflow docs:
    // https://www.tensorflow.org/api_docs/python/tf/scatter_nd

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    TypeConverter typeConverter;

    // Create an adaptor, to interpret the operands
    tile::ScatterOpAdaptor adaptor(operands);
    // 'tensor' provides update values
    // 'dims' contains the destination indices
    // 'other' is the shape of the output
    // this is redundant because the result type also specifies output shape
    auto data = adaptor.data();
    auto indices = adaptor.indices();
    auto updates = adaptor.updates();

    // Make an allocation for the output
    auto resultType = typeConverter.convertType(op.result().getType());
    auto resultMemRefType = resultType.cast<MemRefType>();
    auto resultMemRef =
        rewriter.create<AllocOp>(loc, resultMemRefType).getResult();

    auto dataShape = data.getType().cast<MemRefType>().getShape();
    auto copyLoop = rewriter.create<AffineParallelOp>(
        loc, ArrayRef<Type>{data.getType()},
        ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign}, dataShape);
    rewriter.setInsertionPointToStart(copyLoop.getBody());
    size_t dataDims = dataShape.size();
    auto dataLoadMap = AffineMap::getMultiDimIdentityMap(dataDims, ctx);
    auto loadData = rewriter.create<pxa::PxaLoadOp>(loc, data, dataLoadMap,
                                                    copyLoop.getIVs());
    auto stored = buildSimpleStore(rewriter, loc, loadData, resultMemRef,
                                   tile::getPaddingInfo(op));

    rewriter.create<AffineYieldOp>(loc, ArrayRef<Value>{stored});
    rewriter.setInsertionPointAfter(copyLoop);

    // Get the shape of the update tensor and create a parallel loop over its
    // indexes; we will load each value from the updates, load its destination
    // from the indexes, and store the value to the result.
    auto updatesType = typeConverter.convertType(updates.getType());
    auto updatesMemRefType = updatesType.cast<MemRefType>();
    ArrayRef<int64_t> updatesShape = updatesMemRefType.getShape();

    auto loop = rewriter.create<AffineParallelOp>(
        loc, ArrayRef<Type>{resultMemRefType},
        ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign}, updatesShape);
    rewriter.setInsertionPointToStart(loop.getBody());

    // Load the source value from the updates tensor.
    // The affine map for locating the update value uses all loop dimensions.
    size_t srcDims = updatesShape.size();
    auto srcLoadMap = AffineMap::getMultiDimIdentityMap(srcDims, ctx);
    auto srcLoadOps = loop.getIVs();
    Value srcVal =
        rewriter.create<pxa::PxaLoadOp>(loc, updates, srcLoadMap, srcLoadOps)
            .getResult();

    // Load the location value from the indices tensor.
    // Create an affine map for loading the index, using leading counters.
    size_t axis = *(op.axis().getRawData());
    auto idxShape = indices.getType().cast<MemRefType>().getShape();
    size_t idxDims = idxShape.size();
    auto idxLoadMap = AffineMap::getMultiDimIdentityMap(idxDims, ctx);
    SmallVector<Value, 4> dstOps;

    switch (op.mode()) {
    case ScatterMode::update_nd: {
      std::vector<Value> idxs, combIdx(idxDims);
      for (size_t i = 0; i < idxDims - 1; ++i) {
        combIdx[i] = loop.getIVs()[i];
      }
      for (int64_t i = 0; i < idxShape[idxDims - 1]; ++i) {
        combIdx[idxDims - 1] = rewriter.create<mlir::ConstantIndexOp>(loc, i);
        auto indexVal =
            getIndexValue(loc, rewriter, indices, idxLoadMap, combIdx);
        idxs.push_back(indexVal);
      }
      dstOps.insert(dstOps.begin(), idxs.begin(), idxs.end());
      for (size_t i = idxDims - 1; i < srcDims; ++i) {
        dstOps.push_back(loop.getIVs()[i]);
      }
    } break;
    case ScatterMode::update_slice: {
      auto idxLoadOps = loop.getIVs().slice(axis, idxDims);
      auto idxStart = axis + idxDims - 1;
      auto indexVal =
          getIndexValue(loc, rewriter, indices, idxLoadMap, idxLoadOps);
      getOutputIndices(indexVal, axis, idxStart, srcDims, dstOps,
                       loop.getIVs());
    } break;
    case ScatterMode::normal:
    case ScatterMode::update_elt: {
      auto idxLoadOps = loop.getIVs().take_front(idxDims);
      auto idxStart = axis;
      auto indexVal =
          getIndexValue(loc, rewriter, indices, idxLoadMap, idxLoadOps);
      getOutputIndices(indexVal, axis, idxStart, srcDims, dstOps,
                       loop.getIVs());
    } break;
    default:
      llvm_unreachable("unrecognized scatter mode");
    }

    Value storeResult;
    if (op.mode() == ScatterMode::normal) {
      if (srcVal.getType().isa<FloatType>()) {
        storeResult = rewriter.create<pxa::PxaStoreOp>(
            loc, AtomicRMWKind::addf, srcVal, copyLoop.getResult(0), dstOps);
      } else if (srcVal.getType().isa<IntegerType>()) {
        storeResult = rewriter.create<pxa::PxaStoreOp>(
            loc, AtomicRMWKind::addi, srcVal, copyLoop.getResult(0), dstOps);
      } else {
        llvm_unreachable("Unsupported datatype in scatter.");
      }
    } else {
      storeResult = rewriter.create<pxa::PxaStoreOp>(
          loc, AtomicRMWKind::assign, srcVal, copyLoop.getResult(0), dstOps);
    }
    rewriter.create<AffineYieldOp>(loc, ArrayRef<Value>{storeResult});
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }

  Value getIndexValue(Location loc, ConversionPatternRewriter &rewriter,
                      Value indices, AffineMap idxLoadMap,
                      mlir::ValueRange idxLoadOps) const {
    Value indexVal =
        rewriter.create<pxa::PxaLoadOp>(loc, indices, idxLoadMap, idxLoadOps)
            .getResult();

    // Cast the index value from its integer type to the index type
    if (!indexVal.getType().isa<IndexType>()) {
      // cast from whatever integer type it has to index type
      auto indexType = rewriter.getIndexType();
      indexVal = rewriter.create<mlir::IndexCastOp>(loc, indexVal, indexType)
                     .getResult();
    }
    return indexVal;
  }

  void getOutputIndices(Value indexVal, size_t axis, size_t idxStart,
                        size_t end, SmallVector<Value, 4> &dstOps,
                        std::vector<BlockArgument> loopArgs) const {
    for (size_t i = 0; i < axis; ++i) {
      dstOps.push_back(loopArgs[i]);
    }

    for (size_t i = idxStart; i < end; ++i) {
      dstOps.push_back(loopArgs[i]);
    }

    dstOps[axis] = indexVal;
  }
};

struct CastOpConversion : public OpConversionPattern<tile::CastOp> {
  using OpConversionPattern<tile::CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::CastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto oldResultType = op.result().getType();
    auto resultType =
        typeConverter.convertType(oldResultType).cast<MemRefType>();
    auto operand = operands[0];
    if (resultType == operand.getType()) {
      rewriter.replaceOp(op, operand);
      return success();
    }

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Make a parallel for loop to fill the result
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{resultType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*ranges=*/resultType.getShape());
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);

    // Create the load
    auto scalar =
        buildBroadcastLoad(rewriter, loc, operand, resultType.getRank());

    // Create the standard cast op
    auto dtype = getElementType(op.tensor());
    bool resultIsSigned = getElementType(oldResultType).isSignedInteger();
    auto result = createCastOp(rewriter, loc, scalar, dtype.isSignedInteger(),
                               resultType.getElementType(), resultIsSigned);

    // Create the store
    auto stored = buildSimpleStore(rewriter, loc, result, resultMemRef,
                                   tile::getPaddingInfo(op));
    rewriter.create<AffineYieldOp>(loc, ValueRange{stored});

    // Replace the op
    rewriter.replaceOp(op, forOp.getResult(0));

    return success();
  }
};

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    FunctionType type = op.getType();

    // Convert the function signature
    TypeConverter typeConverter;
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      result.addInputs(i, {typeConverter.convertType(type.getInput(i))});
    }
    SmallVector<Type, 8> resultTypes;
    for (Type resultType : type.getResults()) {
      Type newResultType = typeConverter.convertType(resultType);
      if (!newResultType.isa<stdx::ArgpackType>()) {
        result.addInputs({newResultType});
      }
      resultTypes.push_back(newResultType);
    }

    // Create a new function with an updated signature.
    auto newOp = rewriter.cloneWithoutRegions(op);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    newOp.setType(FunctionType::get(op.getContext(), result.getConvertedTypes(),
                                    resultTypes));

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newOp.getBody(), result);

    // Finally cause the old func op to be erased
    rewriter.eraseOp(op);

    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto &block = op->getParentRegion()->front();
    auto funcOp = op->getParentOfType<FuncOp>();
    auto blockArg = funcOp.getType().getNumInputs() - op.getNumOperands();
    for (Value operand : operands) {
      // Find very initial allocation of memref
      auto def = pxa::getIndirectDef(operand);
      def.replaceAllUsesWith(block.getArgument(blockArg++));
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(op, operands);
    return success();
  }
};

struct PragmaOpConversion : public OpConversionPattern<tile::PragmaOp> {
  using OpConversionPattern<tile::PragmaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::PragmaOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.op() == "trace") {
      return failure();
    }
    tile::PragmaOpAdaptor adaptor(operands);
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }
};

struct TraceOpConversion : public OpConversionPattern<tile::PragmaOp> {
  using OpConversionPattern<tile::PragmaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::PragmaOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.op() != "trace") {
      return failure();
    }
    tile::PragmaOpAdaptor adaptor(operands);
    auto module = op->getParentOfType<ModuleOp>();
    auto msg = op.attrs().getNamed("msg");
    if (!msg) {
      return failure();
    }
    auto symbol = createStubFunc(module, msg->second.cast<StringAttr>());
    rewriter.create<CallOp>(op.getLoc(), symbol, ArrayRef<Type>{});
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }

  FlatSymbolRefAttr createStubFunc(ModuleOp module, StringAttr msg) const {
    static unsigned idCounter = 0;
    auto uniqueId = idCounter++;
    auto symbol = llvm::formatv("__trace_{0}", uniqueId).str();
    auto context = module.getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = FunctionType::get(context, {}, {});
    auto funcOp = builder.create<FuncOp>(module.getLoc(), symbol, funcType,
                                         ArrayRef<NamedAttribute>{});
    funcOp->setAttr("msg", msg);
    funcOp->setAttr("trace", builder.getUnitAttr());
    funcOp->setAttr("id", builder.getI64IntegerAttr(uniqueId));
    funcOp.setPrivate();
    return SymbolRefAttr::get(context, symbol);
  }
};

struct PackOpConversion : public OpConversionPattern<stdx::PackOp> {
  using OpConversionPattern<stdx::PackOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stdx::PackOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto argpackType = stdx::ArgpackType::get(op.getContext());
    // Some 0-dim tensors convert to 0-dim memrefs, and some convert to actual
    // scalars. To make the type mapping exact, we always convert 0-dim memrefs
    // to scalars via doing a load before packing.
    SmallVector<Value, 8> scalarizedOperands;
    for (auto val : operands) {
      // Handle cases that require a load.
      if (auto memrefType = val.getType().dyn_cast<MemRefType>()) {
        if (memrefType.getRank() == 0) {
          auto loadOp =
              rewriter.create<pxa::PxaLoadOp>(op.getLoc(), val, ValueRange{});
          scalarizedOperands.push_back(loadOp.getResult());
          continue;
        }
      }
      // Default case is a no-op
      scalarizedOperands.push_back(val);
    }
    rewriter.replaceOpWithNewOp<stdx::PackOp>(op, TypeRange(argpackType),
                                              scalarizedOperands);
    return success();
  }
};

struct UnpackOpConversion : public OpConversionPattern<stdx::UnpackOp> {
  using OpConversionPattern<stdx::UnpackOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stdx::UnpackOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type, 8> newResultTypes;
    TypeConverter typeConverter;
    for (auto type : op.getResultTypes()) {
      if (auto tensorType = type.dyn_cast<TensorType>()) {
        if (tensorType.getRank() == 0) {
          auto newType = typeConverter.convertType(tensorType.getElementType());
          newResultTypes.push_back(newType);
          continue;
        }
      }
      auto newType = typeConverter.convertType(type);
      newResultTypes.push_back(newType);
    }
    rewriter.replaceOpWithNewOp<stdx::UnpackOp>(op, newResultTypes,
                                                operands[0]);
    return success();
  }
};

struct ScfForOpConversion : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    scf::ForOpAdaptor oldFor(operands);
    auto &oldBodyOps = op.getBody()->getOperations();
    auto newOp = rewriter.create<scf::ForOp>(op.getLoc(), oldFor.lowerBound(),
                                             oldFor.upperBound(), oldFor.step(),
                                             oldFor.initArgs());
    auto &newBodyOps = newOp.getBody()->getOperations();
    newBodyOps.splice(std::prev(newBodyOps.end()), oldBodyOps,
                      oldBodyOps.begin(), oldBodyOps.end());
    auto oldArgs = op.getBody()->getArguments();
    auto newArgs = newOp.getBody()->getArguments();
    for (unsigned i = 0; i < oldArgs.size(); ++i) {
      oldArgs[i].replaceAllUsesWith(newArgs[i]);
    }
    rewriter.replaceOp(op, newOp.results());
    return success();
  }
};

struct LowerTileToPXAPass : public LowerTileToPXABase<LowerTileToPXAPass> {
  void runOnOperation() final {
    // Inject tile.ident ops for each return operand that needs it.
    // argument, a constant value, or a reshape op.
    getOperation().walk([&](ReturnOp op) {
      OpBuilder builder(op);
      for (OpOperand &operand : op.getOperation()->getOpOperands()) {
        Value value = operand.get();
        bool needsIdent =                                  //
            value.isa<BlockArgument>() ||                  // Block arguemnt
            matchPattern(value, m_Constant()) ||           // Constant op
            matchPattern(value, m_Op<stdx::UnpackOp>()) || // Direct from unpack
            matchPattern(value, m_Op<tile::ReshapeOp>());  // Reshape op
        if (needsIdent) {
          Value copy = builder.create<tile::IdentOp>(op.getLoc(),
                                                     value.getType(), value);
          operand.set(copy);
        }
      }
    });

    // Set up target (i.e. what is legal)
    ConversionTarget target(getContext());
    TypeConverter converter;
    target.addLegalDialect<mlir::AffineDialect,          //
                           mlir::math::MathDialect,      //
                           mlir::scf::SCFDialect,        //
                           mlir::StandardOpsDialect,     //
                           dialect::layer::LayerDialect, //
                           dialect::pxa::PXADialect,     //
                           dialect::stdx::StdXDialect>();
    target.addLegalOp<scf::ForOp, scf::YieldOp, scf::IfOp>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp, ReturnOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    target.addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp op) { return converter.isLegal(op); });
    target.addDynamicallyLegalOp<stdx::PackOp>([&](stdx::PackOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<stdx::UnpackOp>([&](stdx::UnpackOp op) {
      return converter.isLegal(op.getResultTypes());
    });
    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp op) { return converter.isLegal(op.getResultTypes()); });

    // Setup rewrite patterns
    using CmpIntLtOp =
        CmpIntInequalityOp<CmpIPredicate::slt, CmpIPredicate::ult>;
    using CmpIntLeOp =
        CmpIntInequalityOp<CmpIPredicate::sle, CmpIPredicate::ule>;
    using CmpIntGtOp =
        CmpIntInequalityOp<CmpIPredicate::sgt, CmpIPredicate::ugt>;
    using CmpIntGeOp =
        CmpIntInequalityOp<CmpIPredicate::sge, CmpIPredicate::uge>;
    OwningRewritePatternList patterns;
    patterns.insert<
        ArgSortOpConversion,  //
        CastOpConversion,     //
        ConstantOpConversion, //
        FuncOpConversion,     //
        GatherOpConversion,   //
        IndexOpConversion,    //
        PragmaOpConversion,   //
        PrngOpConversion,     //
        ReshapeOpConversion,  //
        ReturnOpConversion,   //
        ScatterOpConversion,  //
        ShapeOpConversion,    //
        TraceOpConversion,    //
        PackOpConversion,     //
        UnpackOpConversion,   //
        ScfForOpConversion,   //
        ContractionOpConversion<CombinationKind::none, FirstOperand>,
        ContractionOpConversion<CombinationKind::add, StdOp<mlir::AddFOp>,
                                ResultIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::add, StdOp<mlir::AddIOp>,
                                ResultIs<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::mul, StdOp<mlir::MulFOp>,
                                ResultIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::mul, StdOp<mlir::MulIOp>,
                                ResultIs<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::eq,
                                CmpFloatOp<CmpFPredicate::OEQ>,
                                AnyComparandIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::eq,
                                CmpIntOp<CmpIPredicate::eq>,
                                ComparandsAre<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::cond,
                                CondOp<CmpFloatOp<CmpFPredicate::OEQ>>,
                                AnyComparandIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::cond,
                                CondOp<CmpIntOp<CmpIPredicate::eq>>,
                                AnyComparandIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::ExpOp, StdOp<math::ExpOp>>,
        EltwiseOpConversion<tile::LogOp, StdOp<math::LogOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::PowOp, StdOp<stdx::PowOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::ErfOp, StdOp<stdx::ErfOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::CosOp, StdOp<math::CosOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::TanOp, StdOp<stdx::TanOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::SinHOp, StdOp<stdx::SinHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::CosHOp, StdOp<stdx::CosHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::SinOp, StdOp<math::SinOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::TanHOp, StdOp<math::TanhOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::ACosOp, StdOp<stdx::ACosOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::ASinOp, StdOp<stdx::ASinOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::ATanOp, StdOp<stdx::ATanOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::ACosHOp, StdOp<stdx::ACosHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::ASinHOp, StdOp<stdx::ASinHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::ATanHOp, StdOp<stdx::ATanHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::CeilOp, StdOp<mlir::CeilFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::FloorOp, StdOp<stdx::FloorOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::RoundOp, StdOp<stdx::RoundOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::NegOp, StdOp<mlir::NegFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::NegOp, NegIOp, ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::AddOp, StdOp<mlir::AddFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::AddOp, StdOp<mlir::AddIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::SubOp, StdOp<mlir::SubFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::SubOp, StdOp<mlir::SubIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::MulOp, StdOp<mlir::MulFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::MulOp, StdOp<mlir::MulIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::DivOp, StdOp<mlir::DivFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::DivOp, StdOp<mlir::SignedDivIOp>,
                            ResultIs<EltwiseSigned>>,
        EltwiseOpConversion<tile::DivOp, StdOp<mlir::UnsignedDivIOp>,
                            ResultIs<EltwiseUnsigned>>,
        EltwiseOpConversion<tile::SqrtOp, StdOp<math::SqrtOp>>,
        EltwiseOpConversion<tile::ModOp, StdOp<mlir::RemFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::ModOp, StdOp<mlir::SignedRemIOp>,
                            ResultIs<EltwiseSigned>>,
        EltwiseOpConversion<tile::ModOp, StdOp<mlir::UnsignedRemIOp>,
                            ResultIs<EltwiseUnsigned>>,
        EltwiseOpConversion<tile::CmpEqOp, CmpFloatOp<CmpFPredicate::OEQ>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpEqOp, CmpIntOp<CmpIPredicate::eq>,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpNeOp, CmpFloatOp<CmpFPredicate::ONE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpNeOp, CmpIntOp<CmpIPredicate::ne>,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpLtOp, CmpFloatOp<CmpFPredicate::OLT>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpLtOp, CmpIntLtOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpLeOp, CmpFloatOp<CmpFPredicate::OLE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpLeOp, CmpIntLeOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpGtOp, CmpFloatOp<CmpFPredicate::OGT>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpGtOp, CmpIntGtOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpGeOp, CmpFloatOp<CmpFPredicate::OGE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpGeOp, CmpIntGeOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::BitAndOp, StdOp<mlir::AndOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitOrOp, StdOp<mlir::OrOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitNotOp, NotOp>,
        EltwiseOpConversion<tile::BitXorOp, StdOp<mlir::XOrOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitShlOp, StdOp<mlir::ShiftLeftOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitShrOp, StdOp<mlir::SignedShiftRightOp>,
                            FirstOperandIs<EltwiseSigned>>,
        EltwiseOpConversion<tile::BitShrOp, StdOp<mlir::UnsignedShiftRightOp>,
                            FirstOperandIs<EltwiseUnsigned>>,
        EltwiseOpConversion<tile::LogicalAndOp, LogicalOp<mlir::AndOp>>,
        EltwiseOpConversion<tile::LogicalNotOp, LogicalNotOp>,
        EltwiseOpConversion<tile::LogicalOrOp, LogicalOp<mlir::OrOp>>,
        EltwiseOpConversion<tile::LogicalXorOp, LogicalOp<mlir::XOrOp>>,
        EltwiseOpConversion<tile::SelectOp, SelectOp>,
        EltwiseOpConversion<tile::IdentOp, FirstOperand>>(&getContext());
    // Run the conversion
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTileToPXAPass() {
  return std::make_unique<LowerTileToPXAPass>();
}

} // namespace pmlc::conversion::tile_to_pxa
