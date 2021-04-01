// This is a command line utility that executes an MLIR file on the OpenCL by
// translating MLIR GPU module to SPIR-V and host part to LLVM IR before
// JIT-compiling and executing the latter.
//
// Adapted from LLVM source - mlir-vulkan-runner.cpp for OpenCL by
// Intel Corporation.
// Original copyright:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"

#include "pmlc/all_dialects.h"
#include "pmlc/compiler/program.h"
#include "pmlc/conversion/comp_to_llvm/passes.h"
#include "pmlc/conversion/gpu/passes.h"
#include "pmlc/rt/executable.h"
#include "pmlc/rt/runtime_registry.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]
using pmlc::compiler::Program;
using pmlc::rt::Executable;
using pmlc::util::BufferPtr;
namespace comp = pmlc::dialect::comp;

static LogicalResult runMLIRPasses(ModuleOp module) {
  PassManager passManager(module.getContext());
  applyPassManagerCLOptions(passManager);

  passManager.addPass(pmlc::conversion::gpu::createGpuKernelOutliningPass(
      comp::ExecEnvRuntime::OpenCL, /*memorySpace=*/11));

  passManager.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  passManager.addPass(createConvertGPUToSPIRVPass());
  OpPassManager &modulePM = passManager.nest<spirv::ModuleOp>();
  modulePM.addPass(spirv::createLowerABIAttributesPass());
  modulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
  passManager.addPass(
      pmlc::conversion::comp_to_llvm::createConvertCompToLLVMPass());
  passManager.addPass(createLowerToLLVMPass(LowerToLLVMOptions{
      /*useBarePtrCallConv=*/false,
      /*emitCWrappers=*/true,
      /*indexBitwidth=*/kDeriveIndexBitwidthFromDataLayout,
      /*useAlignedAlloc=*/false,
  }));
  return passManager.run(module);
}

namespace {
/// This options struct prevents the need for global static initializers, and
/// is only initialized if the JITRunner is invoked.
struct Options {
  llvm::cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-")};
  llvm::cl::opt<std::string> mainFuncName{
      "e", llvm::cl::desc("The function to be called"),
      llvm::cl::value_desc("<function name>"), llvm::cl::init("main")};

  llvm::cl::opt<std::string> optDeviceID{
      "device", llvm::cl::desc("The device to use"),
      llvm::cl::value_desc("<device_id>"), llvm::cl::init("opencl.0")};
};
} // namespace

int JitRunnerMain(int argc, char **argv) {
  // Create the options struct containing the command line options for the
  // runner. This must come before the command line options are parsed.
  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "pmlc execution driver\n");

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(options.inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  auto context = std::make_unique<MLIRContext>(registry);
  auto program = std::make_shared<Program>(std::move(context), std::move(file));
  program->entry = options.mainFuncName.getValue();

  if (failed(runMLIRPasses(*program->module)))
    return EXIT_FAILURE;

  auto executable =
      Executable::fromProgram(program, options.optDeviceID.getValue());
  executable->invoke(ArrayRef<BufferPtr>{}, ArrayRef<BufferPtr>{});

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  auto verboseEnv = llvm::sys::Process::GetEnv("PLAIDML_VERBOSE");
  if (verboseEnv) {
    auto level = std::atoi(verboseEnv->c_str());
    if (level) {
      el::Loggers::setVerboseLevel(level);
    }
    IVLOG(level, "PLAIDML_VERBOSE=" << level);
  }

  llvm::llvm_shutdown_obj x;
  registerPassManagerCLOptions();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();
  pmlc::rt::initRuntimes();

  return JitRunnerMain(argc, argv);
}
