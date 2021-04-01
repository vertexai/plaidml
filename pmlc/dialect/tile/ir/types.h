// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Types.h"

namespace pmlc::dialect::tile {

class APFloatType : public mlir::Type::TypeBase<APFloatType, mlir::Type,
                                                mlir::DefaultTypeStorage> {
public:
  using Base::Base;
  using Base::getChecked;
};

class APSignedIntegerType
    : public mlir::Type::TypeBase<APSignedIntegerType, mlir::Type,
                                  mlir::DefaultTypeStorage> {
public:
  using Base::Base;
  using Base::getChecked;
};

class APUnsignedIntegerType
    : public mlir::Type::TypeBase<APUnsignedIntegerType, mlir::Type,
                                  mlir::DefaultTypeStorage> {
public:
  using Base::Base;
  using Base::getChecked;
};

} // namespace pmlc::dialect::tile
