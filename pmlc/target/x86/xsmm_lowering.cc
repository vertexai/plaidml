// Copyright 2020, Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/strides.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace xsmm = dialect::xsmm;

namespace {

int aOffsetGlobalVarCount = 0;
int bOffsetGlobalVarCount = 0;

util::StrideArray getStrideArray(Value operand, AffineMap tileMap) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto type = operand.getType().cast<MemRefType>();
  // TODO: check LogicalResult
  (void)getStridesAndOffset(type, strides, offset);
  auto layoutMap =
      makeStridedLinearLayoutMap(strides, offset, operand.getContext());
  auto info = util::computeStrideArray(layoutMap.compose(tileMap));
  assert(info.hasValue() && "computeStrideArray must succeed");
  return *info;
}

struct PxaGemmOpConversion : public OpConversionPattern<pxa::PxaGemmOp> {
  using OpConversionPattern<pxa::PxaGemmOp>::OpConversionPattern;

  bool getIndices(pxa::PxaGemmOp op, ConversionPatternRewriter &rewriter,
                  pxa::PxaGemmOp::Adaptor &adaptor, AffineMap accessMap,
                  unsigned start, unsigned count,
                  SmallVectorImpl<Value> &into) const {
    auto operands = adaptor.mapOperands().slice(start, count);
    auto indices = expandAffineMap(rewriter, op.getLoc(), accessMap, operands);
    if (!indices)
      return false;
    into.append(indices->begin(), indices->end());
    return true;
  }

  void computeBRGemmOffsets(const SmallVector<int64_t, 4> &numSteps,
                            const SmallVector<int64_t, 4> &stepSizes,
                            const SmallVector<int64_t, 4> &aStrides,
                            const SmallVector<int64_t, 4> &bStrides,
                            SmallVector<int64_t, 4> &aOffsetsArray,
                            SmallVector<int64_t, 4> &bOffsetsArray) const {

    int numBatches = 1;

    for (size_t i = 0; i < numSteps.size(); i++) {
      numBatches *= numSteps[i];
    }

    aOffsetsArray = SmallVector<int64_t, 8>(numBatches, 0);
    bOffsetsArray = SmallVector<int64_t, 8>(numBatches, 0);

    assert((numSteps.size() == aStrides.size() &&
            numSteps.size() == bStrides.size()) &&
           "argument dimension mismatch for offset based BRGEMM");

    IVLOG(3, "numBatches in computeBRGEMM: " << numBatches);

    // variable to record the memory stride of the current index
    // within the offset array
    size_t innerStride = 1;

    for (size_t i = 0; i < numSteps.size(); i++) {
      // memory stride for array a
      int64_t aStride = aStrides[i];
      // memory stride for array b
      int64_t bStride = bStrides[i];
      // the iteration range of this index
      int64_t indexRange = stepSizes[i] * numSteps[i];
      // the number of batches for this index
      int64_t nSteps = numSteps[i];

      IVLOG(3, "aStride in computeBRGEMM: " << aStride);
      IVLOG(3, "bStride in computeBRGEMM: " << bStride);
      IVLOG(3, "indexRange in computeBRGEMM: " << indexRange);
      IVLOG(3, "numSteps in computeBRGEMM: " << nSteps);

      for (size_t k = 0; k < (size_t)numBatches;
           k += ((size_t)nSteps * innerStride)) {
        for (size_t j = 0; j < (size_t)nSteps; j++) {
          for (size_t l = 0; l < innerStride; l++) {
            aOffsetsArray[k + j * innerStride + l] +=
                (j * (indexRange / nSteps) * aStride);
            bOffsetsArray[k + j * innerStride + l] +=
                (j * (indexRange / nSteps) * bStride);
          }
        }
      }
      // update inner memory stride for next index
      innerStride *= (size_t)nSteps;
    }
  }

  LogicalResult
  matchAndRewrite(pxa::PxaGemmOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    pxa::PxaGemmOp::Adaptor transformed(operands);
    SmallVector<Value, 8> indices;
    auto aNumInputs = op.aAccessMap().getNumInputs();
    auto bNumInputs = op.bAccessMap().getNumInputs();
    auto cNumInputs = op.cAccessMap().getNumInputs();
    if (!getIndices(op, rewriter, transformed, op.cAccessMap(), 0, cNumInputs,
                    indices) ||
        !getIndices(op, rewriter, transformed, op.aAccessMap(), cNumInputs,
                    aNumInputs, indices) ||
        !getIndices(op, rewriter, transformed, op.bAccessMap(),
                    cNumInputs + aNumInputs, bNumInputs, indices))
      return failure();

    auto aInfo = getStrideArray(transformed.a(), op.aTileMap());
    auto bInfo = getStrideArray(transformed.b(), op.bTileMap());
    auto cInfo = getStrideArray(transformed.c(), op.cTileMap());
    auto leadingDimsAttr = rewriter.getI64ArrayAttr(ArrayRef<int64_t>{
        aInfo.strides[0], bInfo.strides[0], cInfo.strides[0]});

    auto numBatches = op.numBatches();
    SmallVector<int64_t, 4> numBatchesArr;
    for (auto i : numBatches.getValue()) {
      numBatchesArr.emplace_back(i.cast<IntegerAttr>().getInt());
    }
    // If numbatches only consists of 'k' index call xsmm gemm or xsmm brgemm
    if (numBatchesArr.size() == 1) {
      int numBatches = numBatchesArr[0];
      // If value of numbatches is 1 call xsmm gemm
      if (numBatches == 1) {

        auto dispatch = rewriter.create<xsmm::GemmDispatchF32Op>(
            op.getLoc(), rewriter.getI64Type(), op.tile(), leadingDimsAttr);

        rewriter.create<xsmm::GemmInvokeF32Op>(
            op.getLoc(), ArrayRef<Type>(), dispatch, transformed.c(),
            transformed.a(), transformed.b(), indices);

        // Else call batch reduce gemm when number of batches is greater than 1.
      } else if (numBatches > 1) {

        auto numBatchesAttr = rewriter.getI64IntegerAttr(numBatches);
        auto dispatch = rewriter.create<xsmm::BRGemmDispatchF32Op>(
            op.getLoc(), rewriter.getI64Type(), op.tile(), leadingDimsAttr);

        rewriter.create<xsmm::BRGemmInvokeF32Op>(
            op.getLoc(), ArrayRef<Type>(), dispatch, transformed.c(),
            transformed.a(), transformed.b(), numBatchesAttr, indices);
      }
    } else if (numBatchesArr.size() > 1) {
      // There are additional reduction indices
      // call offset based batch reduce gemm

      // offsets for index k in matrix multiply and
      // additional reduction indices, stepSizes are the tilesizes for each
      // dimension
      SmallVector<int64_t, 4> aStrides, bStrides, stepSizes;
      int64_t aElemSize = transformed.a()
                              .getType()
                              .cast<MemRefType>()
                              .getElementType()
                              .getIntOrFloatBitWidth() /
                          8;
      int64_t bElemSize = transformed.b()
                              .getType()
                              .cast<MemRefType>()
                              .getElementType()
                              .getIntOrFloatBitWidth() /
                          8;
      // Skip index i, start from k at index 1
      for (size_t i = 1; i < aInfo.strides.size(); i++) {
        aStrides.emplace_back(aInfo.strides[i] * aElemSize);
      }

      bStrides.emplace_back(bInfo.strides[0] * bElemSize);
      // Skip 1st index which is j.
      for (size_t i = 2; i < bInfo.strides.size(); i++) {
        bStrides.emplace_back(bInfo.strides[i] * bElemSize);
      }

      // Push the step size (tile size) for k
      int64_t kTile = (op.tile().getValue()[2]).cast<IntegerAttr>().getInt();
      stepSizes.emplace_back(kTile);

      // Rest of the reduction indices are unit step
      for (size_t i = 2; i < aInfo.strides.size(); i++)
        stepSizes.emplace_back(1);

      SmallVector<int64_t, 4> aOffsets, bOffsets;
      int64_t numSteps = 1;
      for (size_t i = 0; i < numBatchesArr.size(); i++) {
        numSteps *= numBatchesArr[i];
      }

      // Computation of offset table
      computeBRGemmOffsets(numBatchesArr, stepSizes, aStrides, bStrides,
                           aOffsets, bOffsets);
      auto dispatch = rewriter.create<xsmm::BRGemmOffsDispatchF32Op>(
          op.getLoc(), rewriter.getI64Type(), op.tile(), leadingDimsAttr);

      rewriter.create<xsmm::BRGemmOffsInvokeF32Op>(
          op.getLoc(), ArrayRef<Type>(), dispatch, transformed.c(),
          transformed.a(), transformed.b(),
          rewriter.getI64IntegerAttr(numSteps),
          rewriter.getI64ArrayAttr(ArrayRef<int64_t>(aOffsets)),
          rewriter.getI64ArrayAttr(ArrayRef<int64_t>(bOffsets)), indices);
    } else
      return failure();

    op.replaceAllUsesWith(transformed.c());
    rewriter.eraseOp(op);

    return success();
  }
};

struct XSMMGemmDispatchF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::GemmDispatchF32Op> {
  using ConvertOpToLLVMPattern<xsmm::GemmDispatchF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::GemmDispatchF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = getOrInsertFunc(op, rewriter);

    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    SmallVector<Value, 6> callOperands;

    // lda, ldb, ldc
    for (auto attr : op.tileld().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    // m, n, k
    for (auto attr : op.tile().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, int64Type, rewriter.getSymbolRefAttr(func), callOperands);
    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kGemmDispatchF32 = "plaidml_rt_xsmm_gemm_dispatch_f32";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kGemmDispatchF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kGemmDispatchF32,
        LLVM::LLVMFunctionType::get(int64Type,
                                    ArrayRef<Type>{int32Type,  // lda
                                                   int32Type,  // ldb
                                                   int32Type,  // ldc
                                                   int32Type,  // m
                                                   int32Type,  // n
                                                   int32Type}, // k
                                    /*isVarArg=*/false));
  }
};

struct XSMMGemmInvokeF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::GemmInvokeF32Op> {
  using ConvertOpToLLVMPattern<xsmm::GemmInvokeF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::GemmInvokeF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    xsmm::GemmInvokeF32Op::Adaptor transformed(operands);
    auto aType = op.a().getType().cast<MemRefType>();
    auto bType = op.b().getType().cast<MemRefType>();
    auto cType = op.c().getType().cast<MemRefType>();

    auto aIndices =
        transformed.indices().slice(cType.getRank(), aType.getRank());
    auto aPtr = getStridedElementPtr(op->getLoc(), aType, transformed.a(),
                                     aIndices, rewriter);

    auto bIndices = transformed.indices().slice(
        cType.getRank() + aType.getRank(), bType.getRank());
    auto bPtr = getStridedElementPtr(op->getLoc(), bType, transformed.b(),
                                     bIndices, rewriter);

    auto cIndices = transformed.indices().slice(0, cType.getRank());
    auto cPtr = getStridedElementPtr(op->getLoc(), cType, transformed.c(),
                                     cIndices, rewriter);

    auto func = getOrInsertFunc(op, rewriter);
    rewriter.create<LLVM::CallOp>(
        op->getLoc(), ArrayRef<Type>(), rewriter.getSymbolRefAttr(func),
        ArrayRef<Value>{transformed.ptr(), aPtr, bPtr, cPtr});
    rewriter.eraseOp(op);

    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kGemmInvokeF32 = "plaidml_rt_xsmm_gemm_invoke_f32";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kGemmInvokeF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int64Type = rewriter.getI64Type();
    auto floatPtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kGemmInvokeF32,
        LLVM::LLVMFunctionType::get(getVoidType(),
                                    ArrayRef<Type>{int64Type,     // funcPtr
                                                   floatPtrType,  // a
                                                   floatPtrType,  // b
                                                   floatPtrType}, // c
                                    /*isVarArg=*/false));
  }
};

struct XSMMBRGemmDispatchF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::BRGemmDispatchF32Op> {
  using ConvertOpToLLVMPattern<
      xsmm::BRGemmDispatchF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::BRGemmDispatchF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = getOrInsertFunc(op, rewriter);

    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    SmallVector<Value, 8> callOperands;

    // lda, ldb, ldc
    for (auto attr : op.tileld().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    // m, n, k
    for (auto attr : op.tile().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    /*
    // stride_a
    callOperands.push_back(
      rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type,
    dispatchOp.StrideA()));
      */

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, int64Type, rewriter.getSymbolRefAttr(func), callOperands);
    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kBRGemmDispatchF32 = "plaidml_rt_xsmm_brgemm_dispatch_f32";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kBRGemmDispatchF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kBRGemmDispatchF32,
        LLVM::LLVMFunctionType::get(int64Type,
                                    ArrayRef<Type>{int32Type,  // lda
                                                   int32Type,  // ldb
                                                   int32Type,  // ldc
                                                   int32Type,  // m
                                                   int32Type,  // n
                                                   int32Type}, // k
                                    /*isVarArg=*/false));
  }
};

struct XSMMBRGemmInvokeF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::BRGemmInvokeF32Op> {
  using ConvertOpToLLVMPattern<xsmm::BRGemmInvokeF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::BRGemmInvokeF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    xsmm::BRGemmInvokeF32Op::Adaptor transformed(operands);
    auto aType = op.a().getType().cast<MemRefType>();
    auto bType = op.b().getType().cast<MemRefType>();
    auto cType = op.c().getType().cast<MemRefType>();

    auto aIndices =
        transformed.indices().slice(cType.getRank(), aType.getRank());
    auto aPtr = getStridedElementPtr(op->getLoc(), aType, transformed.a(),
                                     aIndices, rewriter);

    auto bIndices = transformed.indices().slice(
        cType.getRank() + aType.getRank(), bType.getRank());
    auto bPtr = getStridedElementPtr(op->getLoc(), bType, transformed.b(),
                                     bIndices, rewriter);

    auto cIndices = transformed.indices().slice(0, cType.getRank());
    auto cPtr = getStridedElementPtr(op->getLoc(), cType, transformed.c(),
                                     cIndices, rewriter);

    IntegerType int64Type = rewriter.getI64Type();
    auto numBatches = rewriter.getI64IntegerAttr(op.numBatches());
    auto numBatchesValue =
        rewriter.create<LLVM::ConstantOp>(op->getLoc(), int64Type, numBatches);

    auto func = getOrInsertFunc(op, rewriter);
    rewriter.create<LLVM::CallOp>(
        op->getLoc(), ArrayRef<Type>(), rewriter.getSymbolRefAttr(func),
        ArrayRef<Value>{transformed.ptr(), aPtr, bPtr, cPtr, numBatchesValue});
    rewriter.eraseOp(op);

    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kBRGemmInvokeF32 = "plaidml_rt_xsmm_brgemm_invoke_f32";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kBRGemmInvokeF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int64Type = rewriter.getI64Type();
    auto floatPtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kBRGemmInvokeF32,
        LLVM::LLVMFunctionType::get(getVoidType(),
                                    ArrayRef<Type>{int64Type,    // funcPtr
                                                   floatPtrType, // a
                                                   floatPtrType, // b
                                                   floatPtrType, // c
                                                   int64Type},   // numBatches
                                    /*isVarArg=*/false));
  }
};

struct XSMMBRGemmOffsDispatchF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::BRGemmOffsDispatchF32Op> {
  using ConvertOpToLLVMPattern<
      xsmm::BRGemmOffsDispatchF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::BRGemmOffsDispatchF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = getOrInsertFunc(op, rewriter);

    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    SmallVector<Value, 8> callOperands;

    // lda, ldb, ldc
    for (auto attr : op.tileld().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    // m, n, k
    for (auto attr : op.tile().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, int64Type, rewriter.getSymbolRefAttr(func), callOperands);
    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kBRGemmOffsDispatchF32 =
        "plaidml_rt_xsmm_brgemm_offs_dispatch_f32";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kBRGemmOffsDispatchF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kBRGemmOffsDispatchF32,
        LLVM::LLVMFunctionType::get(int64Type,
                                    ArrayRef<Type>{int32Type,  // lda
                                                   int32Type,  // ldb
                                                   int32Type,  // ldc
                                                   int32Type,  // m
                                                   int32Type,  // n
                                                   int32Type}, // k
                                    /*isVarArg=*/false));
  }
};

struct XSMMBRGemmOffsInvokeF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::BRGemmOffsInvokeF32Op> {
  using ConvertOpToLLVMPattern<
      xsmm::BRGemmOffsInvokeF32Op>::ConvertOpToLLVMPattern;

  // static int aOffsetGlobalVarCount;
  // static int bOffsetGlobalVarCount;

  LogicalResult
  matchAndRewrite(xsmm::BRGemmOffsInvokeF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    xsmm::BRGemmOffsInvokeF32Op::Adaptor transformed(operands);
    auto aType = op.a().getType().cast<MemRefType>();
    auto bType = op.b().getType().cast<MemRefType>();
    auto cType = op.c().getType().cast<MemRefType>();

    auto aIndices =
        transformed.indices().slice(cType.getRank(), aType.getRank());
    auto aPtr = getStridedElementPtr(op->getLoc(), aType, transformed.a(),
                                     aIndices, rewriter);

    auto bIndices = transformed.indices().slice(
        cType.getRank() + aType.getRank(), bType.getRank());
    auto bPtr = getStridedElementPtr(op->getLoc(), bType, transformed.b(),
                                     bIndices, rewriter);

    auto cIndices = transformed.indices().slice(0, cType.getRank());
    auto cPtr = getStridedElementPtr(op->getLoc(), cType, transformed.c(),
                                     cIndices, rewriter);
    IntegerType int64Type = rewriter.getI64Type();

    auto numBatches = rewriter.getI64IntegerAttr(op.numBatches());

    auto numBatchesValue =
        rewriter.create<LLVM::ConstantOp>(op->getLoc(), int64Type, numBatches);

    auto module = op->getParentOfType<ModuleOp>();
    OpBuilder builder(module.getBodyRegion());

    auto aOffsetType = RankedTensorType::get({numBatches.getInt()}, int64Type);
    auto bOffsetType = RankedTensorType::get({numBatches.getInt()}, int64Type);
    LLVM::GlobalOp aOffsets;
    LLVM::GlobalOp bOffsets;
    std::string aGlobalVar =
        "brgemm_aoffsets" + std::to_string(aOffsetGlobalVarCount++);
    std::string bGlobalVar =
        "brgemm_boffsets" + std::to_string(bOffsetGlobalVarCount++);

    aOffsets = builder.create<LLVM::GlobalOp>(
        builder.getUnknownLoc(),
        LLVM::LLVMArrayType::get(int64Type, numBatches.getInt()),
        /*isConstant=*/true, LLVM::Linkage::Internal, StringRef(aGlobalVar),
        DenseElementsAttr::get(aOffsetType, op.aOffsets().getValue()));
    bOffsets = builder.create<LLVM::GlobalOp>(
        builder.getUnknownLoc(),
        LLVM::LLVMArrayType::get(int64Type, numBatches.getInt()),
        /*isConstant=*/true, LLVM::Linkage::Internal, StringRef(bGlobalVar),
        DenseElementsAttr::get(bOffsetType, op.bOffsets().getValue()));

    auto longPtrType = LLVM::LLVMPointerType::get(rewriter.getI64Type());

    auto aOffsetsBase =
        rewriter.create<LLVM::AddressOfOp>(op->getLoc(), aOffsets);

    auto bOffsetsBase =
        rewriter.create<LLVM::AddressOfOp>(op->getLoc(), bOffsets);

    SmallVector<Value, 4> aOffsetOperands = {aOffsetsBase};
    aOffsetOperands.insert(aOffsetOperands.end(), aOffsetType.getRank() + 1,
                           createIndexConstant(rewriter, op->getLoc(), 0));

    SmallVector<Value, 4> bOffsetOperands = {bOffsetsBase};
    bOffsetOperands.insert(bOffsetOperands.end(), bOffsetType.getRank() + 1,
                           createIndexConstant(rewriter, op->getLoc(), 0));

    auto aOffsetsPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), longPtrType,
                                                    aOffsetOperands);

    auto bOffsetsPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), longPtrType,
                                                    bOffsetOperands);

    auto func = getOrInsertFunc(op, rewriter);
    rewriter.create<LLVM::CallOp>(
        op->getLoc(), ArrayRef<Type>(), rewriter.getSymbolRefAttr(func),
        ArrayRef<Value>{transformed.ptr(), aPtr, bPtr, cPtr, numBatchesValue,
                        aOffsetsPtr, bOffsetsPtr});
    rewriter.eraseOp(op);

    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kBRGemmInvokeF32 = "plaidml_rt_xsmm_brgemm_offs_invoke_f32";
    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kBRGemmInvokeF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int64Type = rewriter.getI64Type();
    auto floatPtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
    auto longPtrType = LLVM::LLVMPointerType::get(rewriter.getI64Type());
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kBRGemmInvokeF32,
        LLVM::LLVMFunctionType::get(getVoidType(),
                                    ArrayRef<Type>{int64Type,    // funcPtr
                                                   floatPtrType, // a
                                                   floatPtrType, // b
                                                   floatPtrType, // c
                                                   int64Type,    // numBatches
                                                   longPtrType,  // aOffsetsPtr
                                                   longPtrType}, // bOffsetsPtr
                                    /*isVarArg=*/false));
  }
};

} // namespace

void populatePXAGemmToXSMMConversionPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *ctx) {
  patterns.insert<PxaGemmOpConversion>(ctx);
}

void populateXSMMToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<XSMMGemmDispatchF32Lowering, XSMMGemmInvokeF32Lowering,
                  XSMMBRGemmDispatchF32Lowering, XSMMBRGemmInvokeF32Lowering,
                  XSMMBRGemmOffsDispatchF32Lowering,
                  XSMMBRGemmOffsInvokeF32Lowering>(converter);
}
} // namespace pmlc::target::x86
