// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::dialect::tile {

std::unique_ptr<mlir::Pass> createComputeBoundsPass();

std::unique_ptr<mlir::Pass> createExpandReshapePass();

std::unique_ptr<mlir::Pass> createMaterializePass();

std::unique_ptr<mlir::Pass> createPadRangesPass();

std::unique_ptr<mlir::Pass> createPadConstraintsPass();

std::unique_ptr<mlir::Pass> createReorderInputsPass();

std::unique_ptr<mlir::Pass> createSplitMainPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/tile/transforms/passes.h.inc"

} // namespace pmlc::dialect::tile
