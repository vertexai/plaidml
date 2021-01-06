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

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"

#include "pmlc/all_dialects.h"
#include "pmlc/compiler/program.h"
#include "pmlc/conversion/comp_to_llvm/passes.h"
#include "pmlc/conversion/gpu/passes.h"
#include "pmlc/conversion/gpu_to_spirv/passes.h"
#include "pmlc/rt/executable.h"
#include "pmlc/rt/runtime_registry.h"
#include "pmlc/target/intel_gen/passes.h"
#include "pmlc/target/intel_level_zero/passes.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]
using pmlc::compiler::Program;
using pmlc::rt::Executable;
using pmlc::util::BufferPtr;
namespace comp = pmlc::dialect::comp;
namespace L0 = pmlc::target::intel_level_zero;

struct OclPipelineOptions : public PassPipelineOptions<OclPipelineOptions> {
  Option<bool> useBlockOps{*this, "use-block-ops",
                           llvm::cl::desc("Support for block operations"),
                           llvm::cl::initializer(true)};
  Option<unsigned> spirvVersion{*this, "spirv-version",
                                llvm::cl::desc("SPIR-V Version"),
                                llvm::cl::initializer(150)};
};

static LogicalResult runMLIRPasses(ModuleOp module) {
  PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);

  OclPipelineOptions oclPipelineOptions;

  // Lower mapped scf.parallel's to GPU
  pm.addPass(pmlc::target::intel_gen::createParallelLoopToGpuPass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(
      L0::createAddSpirvTargetPass(oclPipelineOptions.spirvVersion.getValue()));
  pm.addPass(pmlc::conversion::gpu::createGpuKernelOutliningPass(
      comp::ExecEnvRuntime::OpenCL, /*memorySpace=*/11));
  // pm.addPass(conversion::gpu::createGatherGpuLaunchFuncsPass());
  // pm.addPass(comp::createMinimizeBufferTransfersPass());
  // pm.addPass(comp::createExecEnvCoalescingPass());
  // pm.addPass(comp::createMinimizeAllocationsPass());
  // pm.addPass(comp::createRemoveRedundantRWPass());
  // pm.addPass(comp::createRecalculateEventDepsPass(/*safeDealloc=*/false));

  // GPU to SPIR-V.
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  bool nonUniformBroadcast = false;
  if (oclPipelineOptions.spirvVersion.getValue() >= 150) {
    nonUniformBroadcast = true;
  }
  pm.addPass(pmlc::conversion::gpu_to_spirv::createGPUToSPIRVCustomPass(
      nonUniformBroadcast));

  // SPIR-V passes for lowering attributes.
  pm.addPass(L0::createSetSubgroupSizePass());
  pm.addPass(L0::createSetAccessQualifiersPass());
  pm.addPass(L0::createLegalizeSpirvPass());
  pm.addPass(spirv::createLowerABIAttributesPass());
  pm.addPass(spirv::createUpdateVersionCapabilityExtensionPass());

  // Comp to LLVM - LevelZero function calls.
  pm.addPass(pmlc::conversion::comp_to_llvm::createConvertCompToLLVMPass(
      "level_zero_"));

  // Convert to LLVM code.
  pm.addPass(pmlc::target::intel_gen::createConvertStandardToLLVM());
  return pm.run(module);
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
      llvm::cl::value_desc("<device_id>"), llvm::cl::init("level_zero.0")};
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

  auto program = std::make_shared<Program>(std::move(file));
  program->entry = options.mainFuncName.getValue();

  runMLIRPasses(*program->module);

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

  mlir::enableGlobalDialectRegistry(true);
  registerAllDialects();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();
  pmlc::rt::registerRuntimes();
  pmlc::rt::initRuntimes();

  return JitRunnerMain(argc, argv);
}