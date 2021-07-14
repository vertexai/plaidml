// Copyright 2021, Intel Corporation

#pragma once

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"

namespace pmlc::conversion::linalg_to_tile {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/linalg_to_tile/passes.h.inc"

struct LinalgToTileTypeConverter : public mlir::TypeConverter {
  LinalgToTileTypeConverter();
};

void populateLinalgToTileSpecialPatterns(mlir::RewritePatternSet &patterns);

} // namespace pmlc::conversion::linalg_to_tile
