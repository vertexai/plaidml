#ifndef __PMLC_DIALECT_TILE_PASSES__
#define __PMLC_DIALECT_TILE_PASSES__

include "mlir/Pass/PassBase.td"

def ComputeBounds : FunctionPass<"tile-compute-bounds"> {
  let summary = "Compute bounds for contractions";
  let constructor = "pmlc::dialect::tile::createComputeBoundsPass()";
}

def ExpandReshape : FunctionPass<"tile-expand-reshape"> {
  let summary = "Expand reshape and generate contraction";
  let constructor = "pmlc::dialect::tile::createExpandReshapePass()";
}

def Materialize : FunctionPass<"tile-materialize"> {
  let summary = "Materialize arbitrary-precision values based on type promotion rules";
  let constructor = "pmlc::dialect::tile::createMaterializePass()";
}

def PadRanges : FunctionPass<"tile-pad-ranges"> {
  let summary = "Pad ranges to make things more even (possibly adding constraints)";
  let constructor = "pmlc::dialect::tile::createPadRangesPass()";
  let options = [
    Option<"minPowerOfTwo", "min-po2", "unsigned", /*default=*/"8",
           "Minimum padding to consider, must be a power of 2">,
    Option<"maxPowerOfTwo", "max-po2", "unsigned", /*default=*/"16",
           "Maximum padding to consider, must be a power of 2">,
    Option<"maxIncrease", "max-increase", "float", /*default=*/"0.05",
           "Maximum allowed increase as a fraction of original size">
  ];
}

def PadConstraints : FunctionPass<"tile-pad-constraints"> {
  let summary = "Pad outputs to remove constraints on inputs";
  let constructor = "pmlc::dialect::tile::createPadConstraintsPass()";
}

def ScfForReplace : FunctionPass<"tile-scf-replaceArgument"> {
  let summary = "replace the alloc inside scf by function argument";
  let constructor = "pmlc::dialect::tile::createScfForReplacePass()";
}

def SplitMain :  Pass<"tile-split-main", "mlir::ModuleOp"> {
  let summary = "Split main into init/main/fini";
  let constructor = "pmlc::dialect::tile::createSplitMainPass()";
  let dependentDialects = [
    "pmlc::dialect::stdx::StdXDialect"
  ];
}

#endif // __PMLC_DIALECT_TILE_PASSES__
