// Copyright 2019, Intel Corporation

#include "pmlc/compiler/compiler.h"

#include <unordered_map>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/tile_to_pxa/tile_to_pxa.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]
using pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass;

namespace pmlc::compiler {

namespace {

using MemRefTypes = std::vector<MemRefType>;

class ArgumentCollectorPass : public FunctionPass<ArgumentCollectorPass> {
public:
  explicit ArgumentCollectorPass(MemRefTypes *into) : into(into) {}

  void runOnFunction() override {
    auto funcOp = getFunction();
    for (auto arg : funcOp.getArguments()) {
      into->emplace_back(arg.getType().cast<MemRefType>());
    }
  }

  static std::unique_ptr<Pass> create(MemRefTypes *into) {
    return std::make_unique<ArgumentCollectorPass>(into);
  }

private:
  MemRefTypes *into;
};

} // namespace

class MemRefDescriptor {
private:
  struct Base {
    void *basePtr;
    void *data;
    int64_t offset;
  };

public:
  MemRefDescriptor(void *data, MemRefType type) : memory(computeSize(type)) {
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto maybeStrides = getStridesAndOffset(type, strides, offset);
    if (failed(maybeStrides)) {
      throw std::runtime_error("unexpected non-strided memref");
    }
    auto base = reinterpret_cast<Base *>(memory.data());
    base->basePtr = data;
    base->data = data;
    base->offset = offset;
    auto var = reinterpret_cast<int64_t *>(memory.data() + sizeof(Base));
    auto rank = type.getRank();
    auto sizes = type.getShape();
    for (unsigned i = 0; i < rank; i++) {
      var[i] = sizes[i];
      var[i + rank] = strides[i];
    }
  }

  void *ptr() { return memory.data(); }

private:
  static unsigned computeSize(MemRefType type) {
    return sizeof(void *) +                   // allocatedPtr
           sizeof(void *) +                   // alignedPtr
           sizeof(int64_t) +                  // offset
           sizeof(int64_t) * type.getRank() + // sizes
           sizeof(int64_t) * type.getRank();  // strides
  }

  std::vector<char> memory;
};

void Executable::initialize() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  initializeLLVMPasses();
}

void Program::compile(StringRef target) {
  if (target.empty()) {
    return;
  }

  PassManager manager(module->getContext());

  auto shouldPrintBeforePass = [](auto pass, auto op) { return false; };
  auto shouldPrintAfterPass = [&](auto pass, auto op) { return VLOG_IS_ON(3); };
  manager.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, true,
                           false, llvm::errs());
  if (VLOG_IS_ON(1)) {
    manager.enableStatistics();
    manager.enableTiming();
  }

  manager.addPass(dialect::tile::createComputeBoundsPass());
  manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  manager.addNestedPass<FuncOp>(createCSEPass());

  manager.addPass(createLowerTileToPXAPass());
  manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  manager.addNestedPass<FuncOp>(createCSEPass());

  manager.addPass(ArgumentCollectorPass::create(&memRefTypes));

  auto pipelineBuilder = resolveTarget(target);
  pipelineBuilder(manager);

  if (failed(manager.run(*module))) {
    throw std::runtime_error("conversion to the LLVM IR dialect failed");
  }
}

Executable::Executable(const std::shared_ptr<Program> &program,
                       ArrayRef<void *> bufptrs)
    : program(program), args(bufptrs.size()), ptrs(bufptrs.size()) {
  if (program->memRefTypes.size() != bufptrs.size()) {
    throw std::runtime_error("memRefTypes and bufptrs size mismatch");
  }

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    throw std::runtime_error(
        "Failed to create a JITTargetMachineBuilder for the host");
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    throw std::runtime_error("Failed to create a TargetMachine for the host");
  }

  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/0, /*sizeLevel=*/0,
      /*targetMachine=*/tmOrError->get());

  if (VLOG_IS_ON(6)) {
    auto llvmModule = translateModuleToLLVMIR(*program->module);
    if (!llvmModule) {
      throw std::runtime_error("could not convert to LLVM IR");
    }
    llvmModule->print(llvm::errs(), nullptr);
  }

  auto maybeEngine = ExecutionEngine::create(*program->module, optPipeline);
  llvm::handleAllErrors(
      maybeEngine.takeError(), [](const llvm::ErrorInfoBase &err) {
        throw std::runtime_error("Failed to create ExecutionEngine: " +
                                 err.message());
      });
  engine = std::move(*maybeEngine);

  descriptors.reserve(args.size());
  for (unsigned i = 0; i < args.size(); i++) {
    descriptors.emplace_back(bufptrs[i], program->memRefTypes[i]);
    ptrs[i] = descriptors[i].ptr();
    args[i] = &ptrs[i];
  }
}

Executable::~Executable() = default;

void Executable::invoke() {
  auto result =
      engine->invoke(program->entry, llvm::MutableArrayRef<void *>(args));
  if (result) {
    throw std::runtime_error("JIT invocation failed");
  }
}

} // namespace pmlc::compiler
