// Copyright 2020 Intel Corporation
#include "pmlc/util/layout.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/IR/BuiltinTypes.h"

namespace pmlc::util {

using namespace llvm; // NOLINT
using namespace mlir; // NOLINT

MLFramework getMLFramework(StringRef opName) {
  StringRef nGraphPrefix = "ng.";
  if (opName.substr(0, nGraphPrefix.size()) == nGraphPrefix) {
    return MLFramework::ngraph;
  } else {
    return MLFramework::none;
  }
}

TensorLayout getLayoutType(MLFramework framework, StringRef opName,
                           bool isConst) {
  // TODO: Create a common list of primitives that would be used here and in the
  // plugins so we know they are consistent. For now use namings from OV plugin
  // Const are used to identify the primitives weights or parameters, rest is
  // considered as main flow data type
  if (framework == MLFramework::ngraph) {
    if ((opName.find("GroupConvolution") != StringRef::npos) && isConst) {
      return TensorLayout::gkcx;
    } else if ((opName.find("Convolution") != StringRef::npos) && isConst) {
      return TensorLayout::kcx;
    } else if (opName.find("Reshape") != StringRef::npos) {
      return TensorLayout::ncx;
    } else {
      return TensorLayout::ncx;
    }
  }

  return TensorLayout::nxc;
}

} // namespace pmlc::util
