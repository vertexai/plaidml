// Copyright 2019 Intel Corporation

#include <unordered_set>

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Translation.h"

#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT(build/namespaces)
using namespace llvm; // NOLINT(build/namespaces)

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and "
                            "process each chunk independently"),
                   cl::init(false));

int main(int argc, char **argv) {
  START_EASYLOGGINGPP(argc, argv);
  auto level_str = pmlc::util::getEnvVar("PLAIDML_VERBOSE");
  if (level_str.size()) {
    auto level = std::atoi(level_str.c_str());
    if (level) {
      el::Loggers::setVerboseLevel(level);
    }
  }

  mlir::registerAllTranslations();

  // Add flags for all the registered translations.
  llvm::cl::opt<const TranslateFunction *, false, TranslationParser>
      translationRequested("", llvm::cl::desc("Translation to perform"),
                           llvm::cl::Required);

  cl::ParseCommandLineOptions(argc, argv, "pmlc MLIR translation driver\n");

  std::string error_message;
  auto output = openOutputFile(outputFilename, &error_message);
  if (!output) {
    errs() << error_message << "\n";
    return 1;
  }

  auto input = openInputFile(inputFilename, &error_message);
  if (!input) {
    errs() << error_message << "\n";
    return 1;
  }

  // Processes the memory buffer with a new MLIRContext.
  auto processBuffer = [&](std::unique_ptr<MemoryBuffer> ownedBuffer,
                           raw_ostream &os) {
    MLIRContext context;
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());

    // In the diagnostic verification flow, we ignore whether the translation
    // failed (in most cases, it is expected to fail). Instead, we check if the
    // diagnostics were produced as expected.
    mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr,
                                                              &context);
    if (failed((*translationRequested)(sourceMgr, os, &context)))
      return failure();
    return sourceMgrHandler.verify();
  };

  if (splitInputFile) {
    if (failed(splitAndProcessBuffer(std::move(input), processBuffer,
                                     output->os()))) {
      return 1;
    }
  } else {
    if (failed(processBuffer(std::move(input), output->os()))) {
      return 1;
    }
  }

  output->keep();
  return 0;
}
