// Copyright 2021, Intel Corporation

#include "pmlc/conversion/linalg_to_tile/pass_detail.h"

namespace pmlc::conversion::linalg_to_tile {

namespace stdx = dialect::stdx;
namespace tile = dialect::tile;

using namespace mlir; // NOLINT

LinalgToTileTypeConverter::LinalgToTileTypeConverter() {}

void populateLinalgToTileSpecialPatterns(mlir::RewritePatternSet &patterns) {}

} // namespace pmlc::conversion::linalg_to_tile
