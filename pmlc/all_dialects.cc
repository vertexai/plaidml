// Copyright 2020 Intel Corporation

#include "pmlc/all_dialects.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Dialect.h"

#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/xsmm/ir/ops.h"

using namespace mlir; // NOLINT [build/namespaces]

// Add all the MLIR dialects to the provided registry.
void registerAllDialects(DialectRegistry &registry) {
  registry.insert<AffineDialect,                      //
                  LLVM::LLVMDialect,                  //
                  linalg::LinalgDialect,              //
                  math::MathDialect,                  //
                  omp::OpenMPDialect,                 //
                  scf::SCFDialect,                    //
                  StandardOpsDialect,                 //
                  tensor::TensorDialect,              //
                  vector::VectorDialect,              //
                  pmlc::dialect::layer::LayerDialect, //
                  pmlc::dialect::pxa::PXADialect,     //
                  pmlc::dialect::stdx::StdXDialect,   //
                  pmlc::dialect::tile::TileDialect,   //
                  pmlc::dialect::xsmm::XSMMDialect>();
}
