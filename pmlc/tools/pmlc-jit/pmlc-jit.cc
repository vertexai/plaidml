//===- pmlc-jit.cpp - PMLC CPU Execution Driver----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by translating MLIR to LLVM IR before JIT-compiling and executing the
// latter.
//===----------------------------------------------------------------------===//

#include <memory>
#include <stdexcept>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "pmlc/all_dialects.h"
#include "pmlc/compiler/program.h"
#include "pmlc/rt/executable.h"
#include "pmlc/rt/runtime_registry.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT
using llvm::Error;
using pmlc::compiler::Program;
using pmlc::rt::Executable;
using pmlc::util::BufferPtr;

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

  llvm::cl::opt<bool> optMCJIT{"mcjit", llvm::cl::desc("Use MCJIT")};
  llvm::cl::opt<bool> optOrc{"orc", llvm::cl::desc("Use OrcJIT")};

  llvm::cl::opt<std::string> optDeviceID{
      "device", llvm::cl::desc("The device to use"),
      llvm::cl::value_desc("<device_id>"), llvm::cl::init("llvm_cpu.0")};
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

  DialectRegistry registry;
  registry.insert<LLVM::LLVMDialect, omp::OpenMPDialect>();

  auto context = std::make_unique<MLIRContext>(registry);
  auto program = std::make_shared<Program>(std::move(context), std::move(file));
  program->entry = options.mainFuncName.getValue();
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

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();
  pmlc::rt::registerRuntimes();
  pmlc::rt::initRuntimes();

  std::set_terminate([]() {
    auto eptr = std::current_exception();
    if (eptr) {
      try {
        std::rethrow_exception(eptr);
      } catch (const std::exception &ex) {
        llvm::outs() << "ERROR: " << ex.what() << "\n";
      } catch (...) {
        llvm::outs() << "ERROR: Unknown exception\n";
      }
    } else {
      llvm::outs() << "Abnormal termination\n";
    }
    llvm::outs().flush();
    std::exit(EXIT_SUCCESS);
  });

  int exitCode = EXIT_SUCCESS;
  try {
    exitCode = JitRunnerMain(argc, argv);
  } catch (const std::exception &ex) {
    llvm::outs() << "ERROR: " << ex.what() << "\n";
  } catch (...) {
    llvm::outs() << "ERROR: Unknown exception\n";
  }

  llvm::errs().flush();
  llvm::outs().flush();

  return exitCode;
}
