// Copyright 2020 Intel Corporation

#include "pmlc/rt/jit_executable.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/rt/device_id.h"
#include "pmlc/rt/internal.h"
#include "pmlc/rt/runtime.h"
#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT[build/namespaces]

using pmlc::compiler::Program;

namespace pmlc::rt {

// Setup LLVM target triple from the current machine.
static void setupTargetTriple(llvm::Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    throw std::runtime_error("NO target: " + errorMessage);
  }

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);
}

static std::string makePackedFunctionName(StringRef name) {
  return "_mlir_" + name.str();
}

static std::string makeCWrapperFunctionName(StringRef name) {
  return "_mlir_ciface_" + name.str();
}

// Define an interface function that wraps all the arguments of the original
// function and all its results into an i8** pointer to provide a unified
// invocation interface.
static std::string packFunctionArguments(llvm::Module *module,
                                         StringRef entry) {
  auto funcName = makeCWrapperFunctionName(entry);
  auto newName = makePackedFunctionName(entry);
  util::wrapFunctionAndPackArguments(module, funcName, newName);
  return newName;
}

namespace {

constexpr const char *kInitName = "init";
constexpr const char *kFiniName = "fini";

class MemRefDescriptor {
private:
  struct Base {
    void *basePtr;
    void *data;
    int64_t offset;
    int64_t sizesAndStrides[];
  };

public:
  MemRefDescriptor(void *data, RankedTensorType type)
      : memory(computeSize(type)) {
    auto base = reinterpret_cast<Base *>(memory.data());
    base->basePtr = data;
    base->data = data;
    auto rank = type.getRank();
    auto shape = type.getShape();
    auto memRefType = MemRefType::get(shape, type.getElementType());
    SmallVector<int64_t, 8> strides;
    (void)getStridesAndOffset(memRefType, strides, base->offset);
    for (unsigned i = 0; i < rank; i++) {
      base->sizesAndStrides[i] = shape[i];
      base->sizesAndStrides[i + rank] = strides[i];
    }
  }

  void *ptr() { return memory.data(); }

private:
  static unsigned computeSize(RankedTensorType type) {
    return sizeof(void *) +                   // allocatedPtr
           sizeof(void *) +                   // alignedPtr
           sizeof(int64_t) +                  // offset
           sizeof(int64_t) * type.getRank() + // sizes
           sizeof(int64_t) * type.getRank();  // strides
  }

  std::vector<char> memory;
};

using Function = uint8_t *(*)(void **);

struct EngineImpl {
  virtual ~EngineImpl() = default;
  virtual void compile(std::unique_ptr<llvm::Module> module,
                       std::unique_ptr<llvm::LLVMContext> ctx) = 0;
  virtual Function getFunction(StringRef symbol) = 0;
};

static void *tryResolveSymbol(StringRef symbol) {
  if (auto ptr = resolveSymbol(symbol))
    return ptr;
  if (symbol[0] == '_') {
    if (auto ptr = resolveSymbol(symbol.drop_front()))
      return ptr;
  }

  auto ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(symbol.str());
  if (ptr)
    return ptr;

  if (symbol[0] == '_') {
    if (auto ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
            symbol.drop_front().str())) {
      return ptr;
    }
  }

  return nullptr;
}

struct MCJITEngineImpl : EngineImpl {
  struct Runtime : public llvm::LegacyJITSymbolResolver {
    llvm::JITSymbol findSymbol(const std::string &symbol) override {
      auto ptr = tryResolveSymbol(symbol);
      if (!ptr) {
        throw std::runtime_error(
            llvm::formatv("Could not find symbol: {0}", symbol));
      }
      auto addr = llvm::pointerToJITTargetAddress(ptr);
      return llvm::JITEvaluatedSymbol(addr, llvm::JITSymbolFlags::None);
    }

    llvm::JITSymbol findSymbolInLogicalDylib(const std::string &) override {
      return llvm::JITSymbol(nullptr);
    }
  };

  void compile(std::unique_ptr<llvm::Module> module,
               std::unique_ptr<llvm::LLVMContext> ctx) final {
    std::string error;
    std::unique_ptr<llvm::LegacyJITSymbolResolver> resolver(new Runtime);
    engine = std::unique_ptr<llvm::ExecutionEngine>(
        llvm::EngineBuilder(std::move(module))
            .setErrorStr(&error)
            .setOptLevel(llvm::CodeGenOpt::Aggressive)
            .setEngineKind(llvm::EngineKind::JIT)
            .setVerifyModules(true)
            .setSymbolResolver(std::move(resolver))
            .create());
    if (!engine) {
      throw std::runtime_error("Failed to create ExecutionEngine: " + error);
    }

    engine->finalizeObject();
  }

  Function getFunction(StringRef symbol) final {
    uint64_t addr = engine->getFunctionAddress(symbol.str());
    if (!addr) {
      throw std::runtime_error(
          llvm::formatv("Entry point not found: {0}", symbol.str()));
    }
    return reinterpret_cast<Function>(addr);
  }

  std::unique_ptr<llvm::ExecutionEngine> engine;
};

struct OrcJITEngineImpl : EngineImpl {
  void compile(std::unique_ptr<llvm::Module> module,
               std::unique_ptr<llvm::LLVMContext> ctx) final {
    using llvm::Expected;
    using llvm::orc::DynamicLibrarySearchGenerator;
    using llvm::orc::IRCompileLayer;
    using llvm::orc::JITTargetMachineBuilder;
    using llvm::orc::MangleAndInterner;
    using llvm::orc::SymbolMap;
    using llvm::orc::ThreadSafeModule;
    using llvm::orc::TMOwningSimpleCompiler;

    auto dataLayout = module->getDataLayout();

    // Callback to inspect the cache and recompile on demand. This follows
    // Lang's LLJITWithObjectCache example.
    auto compileFunctionCreator = [&](JITTargetMachineBuilder JTMB)
        -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
      JTMB.setCodeGenOptLevel(llvm::CodeGenOpt::Level::Aggressive);
      auto TM = JTMB.createTargetMachine();
      if (!TM)
        return TM.takeError();
      return std::make_unique<TMOwningSimpleCompiler>(std::move(*TM), nullptr);
    };

    jit = llvm::cantFail(llvm::orc::LLJITBuilder()
                             .setCompileFunctionCreator(compileFunctionCreator)
                             .create());

    // Add a ThreadSafeModule to the engine and return.
    ThreadSafeModule tsm(std::move(module), std::move(ctx));
    llvm::cantFail(jit->addIRModule(std::move(tsm)));

    SymbolMap symbols;
    auto &session = jit->getExecutionSession();

    auto addSymbol = [&](StringRef name, void *ptr) {
      auto addr = llvm::pointerToJITTargetAddress(ptr);
      auto symbol = llvm::JITEvaluatedSymbol(addr, llvm::JITSymbolFlags::None);
      symbols.insert(std::make_pair(session.intern(name), symbol));
    };

    for (const auto &kvp : SymbolRegistry::instance()->symbols) {
      addSymbol(kvp.first(), kvp.second);
#ifdef __APPLE__
      addSymbol(llvm::formatv("_{0}", kvp.first()).str(), kvp.second);
#endif
    }

    auto &mainJitDylib = jit->getMainJITDylib();
    cantFail(mainJitDylib.define(absoluteSymbols(symbols)));
    mainJitDylib.addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            dataLayout.getGlobalPrefix())));
  }

  Function getFunction(StringRef symbol) {
    // JIT lookup may return an Error referring to strings stored internally by
    // the JIT. If the Error outlives the ExecutionEngine, it would want have a
    // dangling reference, which is currently caught by an assertion inside JIT
    // thanks to hand-rolled reference counting. Rewrap the error message into a
    // string before returning. Alternatively, ORC JIT should consider copying
    // the string into the error message.
    auto expectedSymbol = jit->lookup(symbol);
    if (!expectedSymbol) {
      std::string errorMessage;
      llvm::raw_string_ostream os(errorMessage);
      llvm::handleAllErrors(expectedSymbol.takeError(),
                            [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
      throw std::runtime_error(os.str());
    }

    auto addr = expectedSymbol->getAddress();
    return reinterpret_cast<Function>(addr);
  }

  std::unique_ptr<llvm::orc::LLJIT> jit;
};

enum class EngineKind {
  MCJIT,
  OrcJIT,
};

struct StopWatch {
  using fp_milliseconds =
      std::chrono::duration<double, std::chrono::milliseconds::period>;

  void start() { startTime = std::chrono::steady_clock::now(); }

  void stop() { stopTime = std::chrono::steady_clock::now(); }

  double delta_ms() {
    return std::chrono::duration_cast<fp_milliseconds>(stopTime - startTime)
        .count();
  }

  std::chrono::steady_clock::time_point startTime;
  std::chrono::steady_clock::time_point stopTime;
};

class JitExecutable final : public Executable {
public:
  JitExecutable(const std::shared_ptr<Program> &program,
                std::shared_ptr<Device> device, ArrayRef<void *> preParams)
      : program(program), device(std::move(device)), preParams(preParams),
        jitInit(nullptr), jitFini(nullptr), initPack(nullptr) {
    static std::once_flag is_initialized;
    std::call_once(is_initialized, []() {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      initializeLLVMPasses();
      llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
    });
    registerLLVMDialectTranslation(*program->context);
    registerOpenMPDialectTranslation(*program->context);

    EngineKind kind = EngineKind::OrcJIT;
    auto jit = pmlc::util::getEnvVar("LLVM_JIT");
    if (jit == "ORC") {
      kind = EngineKind::OrcJIT;
    } else if (jit == "MCJIT") {
      kind = EngineKind::MCJIT;
    }

    switch (kind) {
    case EngineKind::MCJIT:
      impl = std::make_unique<MCJITEngineImpl>();
      break;
    case EngineKind::OrcJIT:
      impl = std::make_unique<OrcJITEngineImpl>();
      break;
    default:
      throw std::runtime_error("Invalid EngineKind");
    }

    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto llvmModule = translateModuleToLLVMIR(*program->module, *ctx);
    if (!llvmModule) {
      throw std::runtime_error("could not convert to LLVM IR");
    }

    setupTargetTriple(llvmModule.get());
    auto entryPoint = packFunctionArguments(llvmModule.get(), program->entry);
    std::string initPacked;
    std::string finiPacked;
    if (llvmModule->getFunction(kInitName) &&
        llvmModule->getFunction(kFiniName)) {
      initPacked = packFunctionArguments(llvmModule.get(), kInitName);
      finiPacked = packFunctionArguments(llvmModule.get(), kFiniName);
    }

    if (VLOG_IS_ON(6)) {
      llvmModule->print(llvm::errs(), nullptr);
    }

    impl->compile(std::move(llvmModule), std::move(ctx));
    jitMain = impl->getFunction(entryPoint);
    if (initPacked != "" && finiPacked != "") {
      jitInit = impl->getFunction(initPacked);
      jitFini = impl->getFunction(finiPacked);
    }

    if (!jitMain) {
      throw std::runtime_error("jitEntry function is null");
    }

    if (jitInit) {
      IVLOG(3, "Doing jit init");
      std::vector<void *> initPtrs;
      std::vector<MemRefDescriptor> initDescriptors;
      std::copy(preParams.begin(), preParams.end(),
                std::back_inserter(initPtrs));
      for (const compiler::ConstantArgument &arg : program->constants) {
        initDescriptors.emplace_back(arg.buffer->data(),
                                     arg.type.cast<RankedTensorType>());
        initPtrs.push_back(initDescriptors.back().ptr());
      }
      initPack = jitInit(initPtrs.data());
      IVLOG(3, "Jit init complete");
    }
  }
  ~JitExecutable() {
    if (jitFini) {
      IVLOG(3, "Doing jit fini");
      std::vector<void *> finiPtrs;
      finiPtrs.push_back(initPack);
      IVLOG(3, "Jit fini complete");
      free(initPack);
    }
  }

  double invoke(mlir::ArrayRef<util::BufferPtr> inputBuffers,
                mlir::ArrayRef<util::BufferPtr> outputBuffers) final {
    StopWatch stopWatch;
    if (VLOG_IS_ON(1)) {
      stopWatch.start();
    }
    bindArguments(inputBuffers, outputBuffers);
    jitMain(ptrs.data());
    if (VLOG_IS_ON(1)) {
      stopWatch.stop();
      IVLOG(1, "Execution time: " << stopWatch.delta_ms() << "ms");
    }
    return device->execTimeInMS;
  }

  void bindArguments(ArrayRef<util::BufferPtr> inputBuffers,
                     ArrayRef<util::BufferPtr> outputBuffers) {
    if (inputBuffers.size() != program->inputs.size()) {
      throw std::runtime_error("Program input arguments and buffers mismatch");
    }

    if (outputBuffers.size() != program->outputs.size()) {
      throw std::runtime_error(
          "Program outputs arguments and buffers mismatch");
    }

    descriptors.clear();
    ptrs.clear();

    if (jitInit) {
      ptrs.push_back(initPack);
    } else {
      std::copy(preParams.begin(), preParams.end(), std::back_inserter(ptrs));
    }

    for (auto [type, buffer] : llvm::zip(program->inputs, inputBuffers)) {
      descriptors.emplace_back(buffer->data(), type.cast<RankedTensorType>());
      ptrs.push_back(descriptors.back().ptr());
    }
    if (!jitInit) {
      for (const compiler::ConstantArgument &arg : program->constants) {
        descriptors.emplace_back(arg.buffer->data(),
                                 arg.type.cast<RankedTensorType>());
        ptrs.push_back(descriptors.back().ptr());
      }
    }
    for (auto [type, buffer] : llvm::zip(program->outputs, outputBuffers)) {
      descriptors.emplace_back(buffer->data(), type.cast<RankedTensorType>());
      ptrs.push_back(descriptors.back().ptr());
    }
  }

private:
  std::shared_ptr<Program> program;
  std::shared_ptr<Device> device;
  std::vector<void *> preParams;
  std::unique_ptr<EngineImpl> impl;
  std::vector<MemRefDescriptor> descriptors;
  std::vector<void *> ptrs;
  Function jitInit;
  Function jitMain;
  Function jitFini;
  uint8_t *initPack;
};

} // namespace

std::unique_ptr<Executable>
makeJitExecutable(const std::shared_ptr<Program> &program,
                  std::shared_ptr<Device> device, ArrayRef<void *> preParams) {
  return std::make_unique<JitExecutable>(program, std::move(device), preParams);
}

} // namespace pmlc::rt
