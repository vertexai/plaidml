// Copyright 2019, Intel Corporation

#pragma once

#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"

#include "pmlc/util/layout.h"
#include "pmlc/util/logging.h"

namespace pmlc::util {

static constexpr const char *kTagAttribute = "tags";

uint64_t getByteSize(mlir::MemRefType type);

// Check if all tags exist in op and they are all true
bool hasAllTags(mlir::Operation *op, llvm::ArrayRef<llvm::StringRef> tags);

// Check if tag exists in op
bool hasTag(mlir::Operation *op, llvm::StringRef tag);

// Set tags in op
void setTags(mlir::Operation *op, llvm::ArrayRef<llvm::StringRef> tags);

// Pack function arguments to an i8** pointer
void wrapFunctionAndPackArguments(llvm::Module *module,
                                  llvm::StringRef funcName,
                                  llvm::StringRef newName);

// A diagnostic tool for searching for problematic transformations in passes.
// Example usage:
//   DiagnosticCounter counter;
//   for (auto op : func.getOps<SomeOp>()) {
//     auto result = counter.next();
//     if (result == DiagnosticCounter::Result::Break)
//       continue;
//     if (result == DiagnosticCounter::Result::Match)
//       IVLOG(0, "match: " << debugString(*fuseA));
//     // Do transformation as normal
//   }
// Use the PLAIDML_COUNTER environment variable to define the threshold where
// the counter will return Break. When the counter reaches the threshold
// excatly, Match is returned.
struct DiagnosticCounter {
  enum Result {
    Break,
    Continue,
    Match,
  };

  DiagnosticCounter();
  Result next();

  size_t counter;
  size_t threshold;
};

mlir::AffineMap updateMemRefWithLayoutMap(mlir::MLIRContext *context,
                                          int64_t rank, TensorLayout layout);

} // namespace pmlc::util

namespace llvm {

template <class T>
inline std::ostream &operator<<(std::ostream &os, const SmallVectorImpl<T> &x) {
  return stringify_collection(os, x.begin(), x.end());
}

} // namespace llvm
