#ifndef __PMLC_CONVERSION_LINALG_TO_TILE_PASSES__
#define __PMLC_CONVERSION_LINALG_TO_TILE_PASSES__

include "mlir/Pass/PassBase.td"

def LowerLinalgToTile : Pass<"convert-linalg-to-tile", "mlir::ModuleOp"> {
  let summary = "Convert Linalg dialect to Tile dialect";
  let constructor = "pmlc::conversion::linalg_to_tile::createLowerLinalgToTilePass()";
  let dependentDialects = [
    "mlir::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "pmlc::dialect::layer::LayerDialect",
    "pmlc::dialect::stdx::StdXDialect",
  ];
}

#endif // __PMLC_CONVERSION_LINALG_TO_TILE_PASSES__
