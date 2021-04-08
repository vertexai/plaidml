// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

namespace pmlc {

bool hasTags(mlir::Operation *op);
void copyTags(mlir::Operation *dst, mlir::Operation *src);
void clearTags(mlir::Operation *op);

void clearTag(mlir::Operation *op, llvm::StringRef name);
void setUnitTag(mlir::Operation *op, llvm::StringRef name);
void setIntegerTag(mlir::Operation *op, llvm::StringRef name, int64_t val);
void setStringTag(mlir::Operation *op, llvm::StringRef name,
                  llvm::StringRef val);
bool hasUnitTag(mlir::Operation *op, llvm::StringRef name);
bool hasIntegerTag(mlir::Operation *op, llvm::StringRef name);
bool hasStringTag(mlir::Operation *op, llvm::StringRef name);
int64_t getIntegerTag(mlir::Operation *op, llvm::StringRef name,
                      int64_t defaultVal);
llvm::StringRef getStringTag(mlir::Operation *op, llvm::StringRef name,
                             llvm::StringRef defaultVal);

// Templated versions
template <typename T>
bool hasTags(T op) {
  return hasTags(op.getOperation());
}
template <typename T>
void copyTags(T dst, T src) {
  return copyTags(dst.getOperation(), src.getOperation());
}
template <typename T>
void clearTags(T op) {
  return clearTags(op.getOperation());
}

template <typename T>
void clearTag(T op, llvm::StringRef name) {
  clearTag(op.getOperation(), name);
}
template <typename T>
void setUnitTag(T op, llvm::StringRef name) {
  setUnitTag(op.getOperation(), name);
}
template <typename T>
void setIntegerTag(T op, llvm::StringRef name, int64_t val) {
  setIntegerTag(op.getOperation(), name, val);
}
template <typename T>
bool hasUnitTag(T op, llvm::StringRef name) {
  return hasUnitTag(op.getOperation(), name);
}
template <typename T>
bool hasIntegerTag(T op, llvm::StringRef name) {
  return hasIntegerTag(op.getOperation(), name);
}
template <typename T>
int64_t getIntegerTag(T op, llvm::StringRef name, int64_t defaultVal) {
  return getIntegerTag(op.getOperation(), name, defaultVal);
}

// List of 'known tags'.  These should probably be in their own headers.  But
// for now I just want to prevent a typo from compiling
inline llvm::StringRef subgroupSizeTag() { return "subgroupSize"; }
inline llvm::StringRef gpuThreadTag() { return "gpuThread"; }
inline llvm::StringRef gpuBlockTag() { return "gpuBlock"; }
inline llvm::StringRef cpuThreadTag() { return "cpuThread"; }

} // end namespace pmlc
