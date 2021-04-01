// Copyright 2019, Intel Corporation

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/ir/types.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

struct OpAsmDialectInterfaceImpl : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  // Get a special name to use when printing the given operation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const final {
    llvm::SmallString<32> osbuf;
    llvm::raw_svector_ostream os(osbuf);
    if (auto cionOp = llvm::dyn_cast<ContractionOp>(op)) {
      if (cionOp.name().hasValue()) {
        os << *cionOp.name();
      }
    } else if (auto const_op = llvm::dyn_cast<tile::ConstantOp>(op)) {
      if (auto attr = const_op.value().dyn_cast<IntegerAttr>()) {
        os << 'c' << attr.getValue();
      } else {
        os << "cst";
      }
    }
    setNameFn(op->getResult(0), os.str());
  }
};

} // namespace

void TileDialect::initialize() {
  addTypes<APFloatType, APSignedIntegerType, APUnsignedIntegerType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/tile/ir/ops.cc.inc"
      >();
  addInterfaces<OpAsmDialectInterfaceImpl>();
}

std::string TileDialect::getDialectAttrName(StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

std::string TileDialect::getCanonicalOpName(StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

Operation *TileDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  IVLOG(3, "tile::TileDialect::materializeConstant> "
               << debugString(value) << " : " << debugString(type));
  auto rankedTensorType = getRankedTensorType(type);
  Type elementType = rankedTensorType.getElementType();
  if (elementType.isa<FloatType>()) {
    if (auto attr = value.dyn_cast<IntegerAttr>()) {
      return builder.create<tile::ConstantOp>(
          loc, elementType, static_cast<double>(attr.getInt()));
    }
  }
  if (elementType.isa<IntegerType>()) {
    if (auto attr = value.dyn_cast<FloatAttr>()) {
      return builder.create<tile::ConstantOp>(
          loc, elementType, static_cast<int64_t>(attr.getValueAsDouble()));
    }
  }
  return builder.create<tile::ConstantOp>(loc, type, value);
}

void TileDialect::printType(Type type, DialectAsmPrinter &printer) const {
  llvm::TypeSwitch<Type>(type)
      .Case<APFloatType>([&](auto deviceType) { printer << "fx"; })
      .Case<APSignedIntegerType>([&](auto eventType) { printer << "six"; })
      .Case<APUnsignedIntegerType>([&](auto eventType) { printer << "uix"; })
      .Default([](Type) { llvm_unreachable("Unsupported 'eltwise' type"); });
}

Type TileDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKeyword;
  if (failed(parser.parseKeyword(&typeKeyword)))
    return nullptr;

  MLIRContext *context = parser.getBuilder().getContext();
  return llvm::StringSwitch<function_ref<Type()>>(typeKeyword)
      .Case("fx", [&] { return parser.getChecked<APFloatType>(context); })
      .Case("six",
            [&] { return parser.getChecked<APSignedIntegerType>(context); })
      .Case("uix",
            [&] { return parser.getChecked<APUnsignedIntegerType>(context); })
      .Default([&] {
        parser.emitError(parser.getNameLoc(),
                         "Unsupported 'eltwise' type: " + typeKeyword);
        return Type();
      })();
}

} // namespace pmlc::dialect::tile
