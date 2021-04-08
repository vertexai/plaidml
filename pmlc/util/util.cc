// Copyright 2019, Intel Corporation

#include "pmlc/util/util.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Process.h"

#include "mlir/IR/AffineMap.h"
#include "pmlc/util/layout.h"

using namespace mlir; // NOLINT

namespace pmlc::util {

uint64_t getByteSize(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 8> strides;
  if (failed(mlir::getStridesAndOffset(type, strides, offset))) {
    throw std::runtime_error("Could not retrieve strides");
  }
  auto sizes = type.getShape();
  uint64_t total = 0;
  for (unsigned i = 0; i < type.getRank(); i++) {
    if (!sizes[i]) {
      return 0;
    }
    if (strides[i] > 0) {
      total += (sizes[i] - 1) * strides[i];
    }
  }
  unsigned elem_bytes = llvm::divideCeil(type.getElementTypeBitWidth(), 8);
  return (total + 1) * elem_bytes;
}

// Check if all tags exist
bool hasAllTags(Operation *op, ArrayRef<StringRef> tags) {
  if (tags.empty()) {
    return true;
  }
  DictionaryAttr opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  if (!opTagsAttr) {
    return false;
  }
  for (StringRef tag : tags) {
    if (!opTagsAttr.get(tag)) {
      return false;
    }
  }
  return true;
}

bool hasTag(Operation *op, StringRef tag) {
  DictionaryAttr opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  if (!opTagsAttr) {
    return false;
  }
  return opTagsAttr.get(tag) != nullptr;
}

// Set tags in op
void setTags(Operation *op, ArrayRef<StringRef> tags) {
  if (tags.empty()) {
    return;
  }
  OpBuilder builder(op);
  DictionaryAttr opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  SmallVector<NamedAttribute, 4> newTags;
  if (opTagsAttr) {
    newTags.append(opTagsAttr.begin(), opTagsAttr.end());
  }
  for (StringRef tag : tags) {
    if (!opTagsAttr || !opTagsAttr.get(tag)) {
      newTags.emplace_back(builder.getNamedAttr(tag, builder.getUnitAttr()));
    }
  }
  op->setAttr(kTagAttribute, builder.getDictionaryAttr(newTags));
}

DiagnosticCounter::DiagnosticCounter() : counter(0), threshold(0) {
  auto env = llvm::sys::Process::GetEnv("PLAIDML_COUNTER");
  if (env) {
    threshold = std::atoi(env->c_str());
  }
}

DiagnosticCounter::Result DiagnosticCounter::next() {
  if (!threshold) {
    return Result::Continue;
  }
  ++counter;
  if (counter < threshold) {
    return Result::Continue;
  }
  if (counter > threshold) {
    return Result::Break;
  }
  return Result::Match;
}

void wrapFunctionAndPackArguments(llvm::Module *module, StringRef funcName,
                                  StringRef newName) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  auto *func = module->getFunction(funcName);
  if (!func) {
    throw std::runtime_error("Could not find function: " + funcName.str());
  }

  // Given a function `foo(<...>) -> T`, define the interface function
  // `mlir_foo(i8**) -> T`.
  auto newType = llvm::FunctionType::get(func->getReturnType(),
                                         builder.getInt8PtrTy()->getPointerTo(),
                                         /*isVarArg=*/false);
  auto funcCst = module->getOrInsertFunction(newName, newType);
  llvm::Function *interfaceFunc = cast<llvm::Function>(funcCst.getCallee());

  // Extract the arguments from the type-erased argument list and cast them to
  // the proper types.
  auto bb = llvm::BasicBlock::Create(ctx);
  bb->insertInto(interfaceFunc);
  builder.SetInsertPoint(bb);
  llvm::Value *argList = interfaceFunc->arg_begin();
  SmallVector<llvm::Value *, 8> args;
  args.reserve(llvm::size(func->args()));
  for (auto &indexedArg : llvm::enumerate(func->args())) {
    llvm::Value *argIndex = llvm::Constant::getIntegerValue(
        builder.getInt64Ty(), APInt(64, indexedArg.index()));
    llvm::Value *argPtrPtr = builder.CreateGEP(argList, argIndex);
    llvm::Value *argPtr = builder.CreateLoad(argPtrPtr);
    auto dstType = indexedArg.value().getType();
    llvm::Value *arg = dstType->isIntegerTy()
                           ? builder.CreatePtrToInt(argPtr, dstType)
                           : builder.CreateBitCast(argPtr, dstType);
    args.push_back(arg);
  }

  // Call the implementation function with the extracted arguments + return
  llvm::Value *val = builder.CreateCall(func, args);
  if (func->getReturnType()->isVoidTy()) {
    builder.CreateRetVoid();
  } else {
    builder.CreateRet(val);
  }
}

AffineMap updateMemRefWithLayoutMap(MLIRContext *context, int64_t rank,
                                    TensorLayout layout) {
  auto spatialNum = (rank == 4 || rank == 5) ? 2 : 0;

  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(rank);
  if (layout == TensorLayout::ncx) {
    dimExprs.push_back(mlir::getAffineDimExpr(0, context));
    for (auto spat = 0; spat < spatialNum; spat++) {
      dimExprs.push_back(
          mlir::getAffineDimExpr(rank - spatialNum + spat, context));
    }
    dimExprs.push_back(mlir::getAffineDimExpr(1, context));
  } else if (layout == TensorLayout::kcx) {
    for (auto spat = 0; spat < spatialNum; spat++) {
      dimExprs.push_back(
          mlir::getAffineDimExpr(rank - spatialNum + spat, context));
    }
    dimExprs.push_back(mlir::getAffineDimExpr(1, context));
    dimExprs.push_back(mlir::getAffineDimExpr(0, context));
  } else if (layout == TensorLayout::gkcx) {
    dimExprs.push_back(mlir::getAffineDimExpr(3, context));
    dimExprs.push_back(mlir::getAffineDimExpr(4, context));
    dimExprs.push_back(mlir::getAffineDimExpr(2, context));
    dimExprs.push_back(mlir::getAffineDimExpr(1, context));
    dimExprs.push_back(mlir::getAffineDimExpr(0, context));
  }
  return AffineMap::get(rank, 0, dimExprs, context);
}

} // namespace pmlc::util
