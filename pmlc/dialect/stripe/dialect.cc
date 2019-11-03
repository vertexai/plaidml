// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/dialect.h"

#include <utility>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"

#include "base/util/logging.h"
#include "pmlc/dialect/stripe/ops.h"

namespace pmlc {
namespace dialect {
namespace stripe {

struct OpAsmInterface : public mlir::OpAsmDialectInterface {
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. The desired
  /// name should be streamed into 'os'.
  void getOpResultName(Operation* op, llvm::raw_ostream& os) const final {
    if (auto attr = op->getAttrOfType<StringAttr>("name")) {
      os << attr.getValue();
    } else if (auto attr = op->getAttrOfType<StringAttr>("scalar_name")) {
      os << "s_" << attr.getValue().substr(1);
    } else if (auto poly_op = llvm::dyn_cast<AffinePolyOp>(op)) {
      if (poly_op.coeffs().size() == 0) {
        os << 'c' << poly_op.offset().getSExtValue();
      }
    }
  }

  void getBlockArgumentName(mlir::BlockArgument* arg, llvm::raw_ostream& os) const final {
    Operation* op = arg->getOwner()->getParentOp();
    if (auto vec = op->getAttrOfType<ArrayAttr>("idx_names")) {
      if (vec.size() > arg->getArgNumber()) {
        if (auto str_attr = vec.getValue()[arg->getArgNumber()].dyn_cast<StringAttr>()) {
          os << str_attr.getValue();
        }
      }
    }
  }

  void getTypeAliases(mlir::SmallVectorImpl<std::pair<Type, StringRef>>& aliases) const final {
    auto ctx = getDialect()->getContext();
    auto affineType = AffineType::get(ctx);
    aliases.push_back(std::make_pair(affineType, StringRef("aff")));
    for (const auto dataType : vertexai::tile::GetDataTypeSet()) {
      for (size_t rank = 0; rank < 9; rank++) {
        auto base = llvm::formatv("{0}_{1}", to_string(dataType), rank).str();
        auto scalarType = ScalarType::get(ctx, dataType);
        aliases.emplace_back(TensorRefType::get(scalarType, rank, false), mlir::Identifier::get(base, ctx));
        aliases.emplace_back(TensorRefType::get(scalarType, rank, true), mlir::Identifier::get(base + "_c", ctx));
      }
    }
  }
};

static mlir::DialectRegistration<Dialect> StripeOps;

Dialect::Dialect(mlir::MLIRContext* ctx) : mlir::Dialect(getDialectNamespace(), ctx) {
  addTypes<          //
      AffineType,    //
      ExecutorType,  //
      TensorType,    //
      TensorRefType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stripe/ops.cc.inc"
      >();
  addInterfaces<OpAsmInterface>();
}

std::string Dialect::getCanonicalOpName(llvm::StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

mlir::Type Dialect::parseTensor(llvm::StringRef tyData, mlir::Location loc) const {
  static llvm::Regex re{R"(([[:alnum:]_]+)\[([[:digit:]]+):([[:digit:]]+)\])"};
  bool is_const = tyData.consume_back("const");
  auto [typeSpec, sizeSpec] = tyData.trim().rsplit('(');
  auto type = mlir::parseType(typeSpec.trim(), getContext());
  if (!type) {
    emitError(loc, "invalid type specification: '") << typeSpec << "'";
    return Type();
  }
  if (!sizeSpec.consume_back(")")) {
    emitError(loc, "invalid tensor type, no ()'s on size spec");
    return Type();
  }
  auto odims = llvm::SmallVector<TensorDim, 8>();
  if (!sizeSpec.empty()) {
    auto dims = llvm::SmallVector<StringRef, 8>();
    sizeSpec.split(dims, ",");
    auto matches = llvm::SmallVector<StringRef, 4>();
    for (auto dim : dims) {
      if (!re.match(dim, &matches)) {
        emitError(loc, "invalid tensor dimension '") << dim << "'";
        return Type();
      }
      auto odim = TensorDim{0, 0, mlir::Identifier::get(matches[1], getContext())};
      matches[2].getAsInteger(10, odim.size);
      matches[3].getAsInteger(10, odim.stride);
      odims.emplace_back(std::move(odim));
    }
  }
  return TensorType::get(type, odims, OffsetsMap(), is_const);
}

mlir::Type Dialect::parseTensorRef(llvm::StringRef tyData, mlir::Location loc) const {
  bool is_const = tyData.consume_back("const");
  StringRef typeSpec, ndimSpec;
  std::tie(typeSpec, ndimSpec) = tyData.rsplit(':');
  auto type = mlir::parseType(typeSpec.trim(), getContext());
  if (!type) {
    emitError(loc, "invalid type specification: '") << typeSpec << "'";
    return Type();
  }
  size_t ndims;
  if (ndimSpec.trim().consumeInteger(0, ndims)) {
    emitError(loc, "invalid ndims'") << ndims << "'";
    return Type();
  }
  return TensorRefType::get(type, ndims, is_const);
}

std::string Dialect::getDialectAttrName(llvm::StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

mlir::Type Dialect::parseType(llvm::StringRef tyData, mlir::Location loc) const {
  if (tyData == "affine") {
    return AffineType::get(getContext());
  }
  if (tyData == "executor") {
    return ExecutorType::get(getContext());
  }
  if (tyData.consume_front("tensor ")) {
    return parseTensor(tyData, loc);
  }
  if (tyData.consume_front("tensor_ref ")) {
    return parseTensorRef(tyData, loc);
  }
  emitError(loc, "unknown stripe type: '" + tyData + "'");
  return Type();
}

static void print(AffineType type, llvm::raw_ostream& os) { os << "affine"; }

static void print(ExecutorType type, llvm::raw_ostream& os) { os << "executor"; }

static void print(TensorType type, llvm::raw_ostream& os) {
  os << "tensor ";
  os << type.getElementType() << "(";
  auto shape = type.getShape();
  for (int64_t i = 0; i < type.getRank(); i++) {
    const auto& dim = shape[i];
    if (i) {
      os << ", ";
    }
    os << dim.cls << '[' << dim.size << ":" << dim.stride << ']';
  }
  os << ")";
  if (type.is_const()) {
    os << " const";
  }
}

static void print(TensorRefType type, llvm::raw_ostream& os) {
  os << "tensor_ref " << type.getElementType() << ":" << std::to_string(type.getRank());
  if (type.is_const()) {
    os << " const";
  }
}

void Dialect::printType(mlir::Type type, llvm::raw_ostream& os) const {
  if (auto affineType = type.dyn_cast<AffineType>()) {
    print(affineType, os);
  } else if (auto executorType = type.dyn_cast<ExecutorType>()) {
    print(executorType, os);
  } else if (auto tensorType = type.dyn_cast<TensorType>()) {
    print(tensorType, os);
  } else if (auto tensorRefType = type.dyn_cast<TensorRefType>()) {
    print(tensorRefType, os);
  } else {
    llvm_unreachable("unhandled Plaid type");
  }
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
