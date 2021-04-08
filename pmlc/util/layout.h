// Copyright 2020, Intel Corporation

#pragma once

// Copyright 2020 Intel Corporation
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "pmlc/util/enums.h"
#include "pmlc/util/tags.h"

namespace pmlc::util {

MLFramework getMLFramework(llvm::StringRef opName);

TensorLayout getLayoutType(MLFramework framework, llvm::StringRef opName,
                           bool isConst = false);

inline llvm::StringRef layoutTag() { return "layout"; }

inline void setLayoutTag(mlir::Operation *op, TensorLayout layout) {
  // setIntegerTag(op, layoutTag(), static_cast<int64_t>(layout));
  setStringTag(op, layoutTag(), stringifyTensorLayout(layout));
}

inline bool hasLayoutTag(mlir::Operation *op) {
  // return hasIntegerTag(op, layoutTag());
  return hasStringTag(op, layoutTag());
}

inline TensorLayout getLayoutTag(mlir::Operation *op) {
  // return static_cast<TensorLayout>(getIntegerTag(op, layoutTag(), 0));
  return symbolizeTensorLayout(
             getStringTag(op, layoutTag(),
                          stringifyTensorLayout(TensorLayout::nxc)))
      .getValue();
}

} // namespace pmlc::util
