// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "pmlc/dialect/pxa/ir/interfaces.h"
#include "pmlc/util/enums.h"

using mlir::AffineMapAccessInterface;

#define GET_OP_CLASSES
#include "pmlc/dialect/pxa/ir/ops.h.inc"

#include "pmlc/dialect/pxa/ir/dialect.h.inc"
