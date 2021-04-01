// Copyright 2020 Intel Corporation

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace {

struct TraceLinkingPass : public TraceLinkingBase<TraceLinkingPass> {
  void runOnOperation() override {
    getOperation().walk([](LLVM::LLVMFuncOp op) {
      if (!op->getAttrOfType<UnitAttr>("trace")) {
        return;
      }
      if (!op.empty()) {
        return;
      }
      auto loc = op.getLoc();
      auto *context = op.getContext();
      OpBuilder builder(context);
      auto module = op->getParentOfType<ModuleOp>();
      auto traceRef = getOrInsertTrace(loc, builder, module);
      auto block = op.addEntryBlock();
      builder.setInsertionPointToStart(block);
      auto id = op->getAttrOfType<IntegerAttr>("id").getValue().getZExtValue();
      auto msgStr = op->getAttrOfType<StringAttr>("msg").getValue().str();
      auto msg = StringRef(msgStr.c_str(), msgStr.size() + 1);
      auto msgSymbol = llvm::formatv("__trace_msg_{0}", id).str();
      auto msgValue =
          getOrCreateGlobalString(loc, builder, msgSymbol, msg, module);
      builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{}, traceRef,
                                   ArrayRef<Value>{msgValue});
      builder.create<LLVM::ReturnOp>(loc, ArrayRef<Value>{});
      op->removeAttr("id");
      op->removeAttr("msg");
      op->removeAttr("trace");
    });
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type =
          LLVM::LLVMArrayType::get(builder.getIntegerType(8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getIntegerType(8)), globalPtr,
        ArrayRef<Value>({cst0, cst0}));
  }

  static FlatSymbolRefAttr getOrInsertTrace(Location loc, OpBuilder &builder,
                                            ModuleOp module) {
    const char *symbol = "plaidml_rt_trace";
    auto *context = module.getContext();
    if (module.lookupSymbol(symbol)) {
      return SymbolRefAttr::get(context, symbol);
    }
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto voidTy = LLVM::LLVMVoidType::get(context);
    auto msgTy = LLVM::LLVMPointerType::get(builder.getIntegerType(8));
    auto funcType =
        LLVM::LLVMFunctionType::get(voidTy, {msgTy}, /*isVarArg=*/false);
    builder.create<LLVM::LLVMFuncOp>(loc, symbol, funcType);
    return SymbolRefAttr::get(context, symbol);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTraceLinkingPass() {
  return std::make_unique<TraceLinkingPass>();
}

} // namespace pmlc::target::x86
