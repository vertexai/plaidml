// Copyright 2019, Intel Corporation

#pragma once

#include <string>

#include "mlir/IR/Dialect.h"

namespace llvm {
class Regex;
}

namespace pmlc {
namespace dialect {
namespace stripe {

class TensorDim;

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx);

  static llvm::StringRef getDialectNamespace() { return "stripe"; }
  static std::string getDialectAttrName(llvm::StringRef name);
  static llvm::StringRef getStripeAttrsName() { return "stripe_attrs"; }
  static std::string getCanonicalOpName(llvm::StringRef name);

  mlir::Type parseTensor(llvm::StringRef tyData, mlir::Location loc) const;
  mlir::Type parseTensorRef(llvm::StringRef tyData, mlir::Location loc) const;
  mlir::LogicalResult parseTensorSize(llvm::StringRef sizeSpec, mlir::Location loc,    // NOLINT
                                      llvm::SmallVectorImpl<TensorDim>& odims) const;  // NOLINT

  mlir::Type parseType(mlir::DialectAsmParser& parser) const override;
  void printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const override;
};

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
