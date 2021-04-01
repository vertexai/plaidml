// Copyright 2020, Intel Corporation
#include "pmlc/conversion/comp_to_llvm/passes.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/comp_to_llvm/pass_detail.h"
#include "pmlc/conversion/comp_to_llvm/utils.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/dialect/comp/ir/dialect.h"
#include "pmlc/util/logging.h"

namespace pmlc::conversion::comp_to_llvm {

using namespace mlir; // NOLINT

namespace comp = pmlc::dialect::comp;

namespace {

template <typename Op>
class CompConversionBase : public ConvertOpToLLVMPattern<Op> {
public:
  CompConversionBase(const BinaryModulesMap &modulesMap, StringRef baseName,
                     LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<Op>(converter), modulesMap(modulesMap),
        baseName(baseName) {}

  Value convertOperand(Value val, Location loc, OpBuilder &builder) const {
    if (!val.getType().isa<LLVM::LLVMStructType>()) {
      return val;
    }
    MemRefDescriptor desc(val);
    auto ptr = desc.alignedPtr(builder, loc);
    auto llvmInt8Ptr =
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8));
    if (ptr.getType().cast<LLVM::LLVMPointerType>().getAddressSpace() != 0) {
      return builder.create<LLVM::AddrSpaceCastOp>(loc, llvmInt8Ptr, ptr);
    }
    return builder.create<LLVM::BitcastOp>(loc, llvmInt8Ptr, ptr);
  }

  FlatSymbolRefAttr getSym(Builder &builder, StringRef name) const {
    return builder.getSymbolRefAttr((baseName + name).str());
  }

protected:
  const BinaryModulesMap &modulesMap;
  std::string baseName;
};

// Conversion for comp allocations.  We actually make a proper memref descriptor
// for the alloced buffer.
class ConvertAlloc : public CompConversionBase<comp::Alloc> {
  using CompConversionBase<comp::Alloc>::CompConversionBase;

  LogicalResult
  matchAndRewrite(comp::Alloc op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Get some basic info about op
    auto memRefType = op->getResult(0).getType().cast<MemRefType>();
    auto loc = op->getLoc();

    // Get all of the sizes we need.  TODO: Latest upstream has improved this.
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;

    getMemRefDescriptorSizes(loc, memRefType, {}, rewriter, sizes, strides,
                             sizeBytes);

    // Do the actual allocation
    auto llvmInt8Ptr =
        LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 8));
    auto sym = getSym(rewriter, "alloc");
    Value ptr =
        rewriter
            .create<LLVM::CallOp>(loc, TypeRange(llvmInt8Ptr), sym,
                                  ArrayRef<Value>{operands[0], sizeBytes})
            .getResult(0);

    // Cast it the right result type
    Type elementPtrType = getElementPtrType(memRefType);
    Value typedPtr;
    if (elementPtrType.cast<LLVM::LLVMPointerType>().getAddressSpace() != 0) {
      typedPtr =
          rewriter.create<LLVM::AddrSpaceCastOp>(loc, elementPtrType, ptr);
    } else {
      typedPtr = rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, ptr);
    }

    // Create the MemRef descriptor.
    Value memRefDescriptor = createMemRefDescriptor(
        loc, memRefType, typedPtr, typedPtr, sizes, strides, rewriter);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }
};

// Special handling for creating kernels since need access to the binary mobdule
// mop to hook up the actual SPIRV
struct ConvertCreateKernel : CompConversionBase<comp::CreateKernel> {
  using CompConversionBase<comp::CreateKernel>::CompConversionBase;

  LogicalResult
  matchAndRewrite(comp::CreateKernel op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Gather some basic information
    Location loc = op.getLoc();
    std::string binaryName = op.kernelFuncAttr().getRootReference().str();
    std::string kernelName = op.kernelFuncAttr().getLeafReference().str();
    auto llvmKernelType =
        LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 8));

    // Create kernel from serialized binary.
    if (modulesMap.count(binaryName) == 0)
      return failure();
    if (modulesMap.at(binaryName).kernelsNameMap.count(kernelName) == 0)
      return failure();

    Value binaryPtr, binaryBytes;
    getPtrToBinaryModule(rewriter, loc, modulesMap.at(binaryName), binaryPtr,
                         binaryBytes);
    Value namePtr = getPtrToGlobalString(
        rewriter, loc, modulesMap.at(binaryName).kernelsNameMap.at(kernelName));

    // Call the actual function
    auto sym = getSym(rewriter, "create_kernel");
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, ArrayRef<Type>(llvmKernelType), sym,
        ArrayRef<Value>{operands[0], binaryPtr, binaryBytes, namePtr});
    return success();
  }
};

// Conversion for schedule compute.  This is special cased largely due to the
// need for 2 vararg regions as well as some index conversion
class ConvertScheduleCompute
    : public CompConversionBase<comp::ScheduleCompute> {
  using CompConversionBase<comp::ScheduleCompute>::CompConversionBase;

  LogicalResult
  matchAndRewrite(comp::ScheduleCompute op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Get an adaptor to view operands as an op, and some basic info.
    auto loc = op->getLoc();
    auto llvmInt8Ptr =
        LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 8));
    SmallVector<NamedAttribute, 8> sortedAttrs;
    DictionaryAttr::sort(op->getAttrs(), sortedAttrs);
    auto dict = DictionaryAttr::getWithSorted(op->getContext(), sortedAttrs);
    comp::ScheduleComputeAdaptor adaptor(operands, dict);

    // Make operand list
    SmallVector<Value, 4> newOperands;
    auto pushOperand = [&](Value val) {
      newOperands.push_back(convertOperand(val, loc, rewriter));
    };
    pushOperand(adaptor.execEnv());
    pushOperand(adaptor.kernel());
    pushOperand(adaptor.gridSizeX());
    pushOperand(adaptor.gridSizeY());
    pushOperand(adaptor.gridSizeZ());
    pushOperand(adaptor.blockSizeX());
    pushOperand(adaptor.blockSizeY());
    pushOperand(adaptor.blockSizeZ());
    Value bufferCount =
        createIndexConstant(rewriter, loc, adaptor.buffers().size());
    newOperands.push_back(bufferCount);
    Value eventCount =
        createIndexConstant(rewriter, loc, adaptor.depEvents().size());
    newOperands.push_back(eventCount);
    for (Value buf : adaptor.buffers()) {
      pushOperand(buf);
    }
    for (Value evt : adaptor.depEvents()) {
      pushOperand(evt);
    }

    // DO the actual call
    auto sym = getSym(rewriter, "schedule_compute");
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange(llvmInt8Ptr), sym,
                                              newOperands);
    return success();
  }
};

// Convert the op to a function in a basically 1-to-1 way.  We special case
// memrefs to turn into pointers, and we support variadic arguments via adding a
// count at the LLVM level.  This works for most of the comp dialect.
template <typename Op, bool varArg = false, size_t nonVarArgs = 0>
class ConvertToFuncCall : public CompConversionBase<Op> {
public:
  using CompConversionBase<Op>::CompConversionBase;

  LogicalResult
  matchAndRewrite(Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Convert result types
    SmallVector<Type, 1> resultTypes;
    for (Type type : op->getResultTypes()) {
      resultTypes.push_back(this->getTypeConverter()->convertType(type));
    }
    // Make a vector for operands + copy pre-vararg ops
    auto loc = op->getLoc();
    SmallVector<Value, 4> newOperands;
    for (unsigned i = 0; i < nonVarArgs; i++) {
      newOperands.push_back(this->convertOperand(operands[i], loc, rewriter));
    }
    // If it's vararg, add count of remaining ops
    if (varArg) {
      Value varArgCount = this->createIndexConstant(
          rewriter, loc, operands.size() - nonVarArgs);
      newOperands.push_back(varArgCount);
    }
    // Put in the remaining args
    for (unsigned i = nonVarArgs; i < operands.size(); i++) {
      newOperands.push_back(this->convertOperand(operands[i], loc, rewriter));
    }
    // Do thee actual call
    auto sym = this->getSym(rewriter, op->getName().stripDialect());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, resultTypes, sym,
                                              newOperands);
    return success();
  }
};

class ConvertCompToLLVMPass
    : public ConvertCompToLLVMBase<ConvertCompToLLVMPass> {
public:
  ConvertCompToLLVMPass() = default;
  explicit ConvertCompToLLVMPass(StringRef prefix) {
    this->prefix = prefix.str();
  }
  void addDeclarations(StringRef prefix, ModuleOp module) {
    Location loc = module.getLoc();
    OpBuilder builder(module.getBody()->getTerminator());
    MLIRContext *context = builder.getContext();
    auto llvmInt8Ptr = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmVoid = LLVM::LLVMVoidType::get(context);
    auto llvmInt64 = IntegerType::get(context, 64);

    auto addFunc = [&](const char *name, Type retType, ArrayRef<Type> argTypes,
                       bool isVarArg) {
      std::string newName = prefix.str() + name;
      if (!module.lookupSymbol(name)) {
        builder.create<LLVM::LLVMFuncOp>(
            loc, newName,
            LLVM::LLVMFunctionType::get(retType, argTypes, isVarArg));
      }
    };

    addFunc("create_execenv", llvmInt8Ptr, {llvmInt8Ptr}, /*isVarArg=*/false);
    addFunc("destroy_execenv", llvmVoid, {llvmInt8Ptr}, /*isVarArg=*/false);
    addFunc("create_kernel", llvmInt8Ptr,
            {llvmInt8Ptr, llvmInt8Ptr, llvmInt64, llvmInt8Ptr},
            /*isVarArg=*/false);
    addFunc("destroy_kernel", llvmVoid, {llvmInt8Ptr, llvmInt8Ptr},
            /*isVarArg=*/false);
    addFunc("schedule_compute", llvmInt8Ptr,
            {llvmInt8Ptr, llvmInt8Ptr, //
             llvmInt64, llvmInt64, llvmInt64, llvmInt64, llvmInt64,
             llvmInt64, //
             llvmInt64, llvmInt64},
            /*isVarArg=*/true);
    addFunc("submit", llvmVoid, {llvmInt8Ptr}, /*isVarArg=*/false);
    addFunc("alloc", llvmInt8Ptr, {llvmInt8Ptr, llvmInt64}, /*isVarArg=*/false);
    addFunc("dealloc", llvmVoid, {llvmInt8Ptr, llvmInt8Ptr},
            /*isVarArg=*/false);
    addFunc("schedule_write", llvmInt8Ptr,
            {llvmInt8Ptr, llvmInt8Ptr, llvmInt8Ptr, llvmInt64},
            /*isVarArg=*/true);
    addFunc("schedule_read", llvmInt8Ptr,
            {llvmInt8Ptr, llvmInt8Ptr, llvmInt8Ptr, llvmInt64},
            /*isVarArg=*/true);

    addFunc("wait", llvmVoid, {llvmInt64}, /*isVarArg=*/true);
    addFunc("schedule_barrier", llvmInt8Ptr, {llvmInt8Ptr, llvmInt64},
            /*isVarArg=*/true);
    addFunc("dump_profiling", llvmVoid, {llvmInt8Ptr}, /*isVarArg=*/false);
  }

  void runOnOperation() {
    // Get the main module
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    // Add the declarations
    addDeclarations(this->prefix, module);

    // Convert the SPIRV code to binary form
    BinaryModulesMap modulesMap;
    if (failed(serializeSpirvKernels(module, modulesMap))) {
      return signalPassFailure();
    }

    // Build the LLVM converter
    LowerToLLVMOptions options = {
        /*useBarePtrCallConv=*/false,
        /*emitCWrappers=*/true,
        /*indexBitwidth=*/64,
        /*useAlignedAlloc=*/false,
    };
    LLVMTypeConverter converter(context, options);

    // Make all the comp types convert to simple pointers
    auto llvmInt8Ptr = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    converter.addConversion(
        [=](comp::DeviceType deviceType) { return llvmInt8Ptr; });
    converter.addConversion(
        [=](comp::ExecEnvType execEnvType) { return llvmInt8Ptr; });
    converter.addConversion(
        [=](comp::EventType eventType) { return llvmInt8Ptr; });
    converter.addConversion(
        [=](comp::KernelType kernelType) { return llvmInt8Ptr; });

    // Add the conversion patterns
    OwningRewritePatternList patterns;
    populateExpandTanhPattern(patterns, context);
    populateStdToLLVMConversionPatterns(converter, patterns);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(converter,
                                                                   patterns);
    patterns.insert<                                      //
        ConvertToFuncCall<comp::CreateExecEnv>,           //
        ConvertToFuncCall<comp::DestroyExecEnv>,          //
        ConvertCreateKernel,                              //
        ConvertToFuncCall<comp::DestroyKernel>,           //
        ConvertScheduleCompute,                           //
        ConvertToFuncCall<comp::Submit>,                  //
        ConvertAlloc,                                     //
        ConvertToFuncCall<comp::Dealloc>,                 //
        ConvertToFuncCall<comp::DumpProfiling>,           //
        ConvertToFuncCall<comp::ScheduleWrite, true, 3>,  //
        ConvertToFuncCall<comp::ScheduleRead, true, 3>,   //
        ConvertToFuncCall<comp::Wait, true, 0>,           //
        ConvertToFuncCall<comp::ScheduleBarrier, true, 1> //
        >(modulesMap, this->prefix, converter);
    LLVMConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<ModuleTerminatorOp>();
    target.addIllegalDialect<comp::COMPDialect>();
    target.addLegalOp<gpu::GPUModuleOp>();
    target.markOpRecursivelyLegal<gpu::GPUModuleOp>();

    // Do the actual conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertCompToLLVMPass() {
  return std::make_unique<ConvertCompToLLVMPass>();
}

std::unique_ptr<Pass> createConvertCompToLLVMPass(StringRef prefix) {
  return std::make_unique<ConvertCompToLLVMPass>(prefix);
}

} // namespace pmlc::conversion::comp_to_llvm
