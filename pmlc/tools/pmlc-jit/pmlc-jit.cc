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
//
// The implementation also supports specifying an expected runtime_error being
// thrown and validates with no failure the expected string is correctly
// thrown.
//===----------------------------------------------------------------------===//

#include <stdexcept>

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Support/JitRunner.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "pmlc/util/all_dialects.h"
#include "pmlc/util/all_passes.h"

int main(int argc, char **argv) {
  registerAllDialects();
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();

  try {
    return mlir::JitRunnerMain(argc, argv, nullptr);
  } catch (const std::runtime_error &e) {
    llvm::outs() << e.what();
  }

  return 0;
}
