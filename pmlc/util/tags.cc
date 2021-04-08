// Copyright 2020 Intel Corporation
#include "pmlc/util/tags.h"

#include "mlir/IR/BuiltinTypes.h"

namespace pmlc {

using namespace llvm; // NOLINT
using namespace mlir; // NOLINT

static StringRef tagDict() { return "tags"; }

bool hasTags(mlir::Operation *op) {
  return static_cast<bool>(op->getAttrOfType<DictionaryAttr>(tagDict()));
}

void copyTags(mlir::Operation *dst, mlir::Operation *src) {
  auto dict = src->getAttrOfType<DictionaryAttr>(tagDict());
  if (dict) {
    dst->setAttr(tagDict(), dict);
  }
}

void clearTags(mlir::Operation *op) { op->removeAttr(tagDict()); }

void clearTag(mlir::Operation *op, llvm::StringRef name) {
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  dict.erase(name);
  if (dict.empty()) {
    op->removeAttr(tagDict());
  } else {
    op->setAttr(tagDict(), dict.getDictionary(op->getContext()));
  }
}

void setUnitTag(mlir::Operation *op, llvm::StringRef name) {
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  dict.set(name, UnitAttr::get(op->getContext()));
  op->setAttr(tagDict(), dict.getDictionary(op->getContext()));
}

void setIntegerTag(mlir::Operation *op, llvm::StringRef name, int64_t val) {
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  auto type = IntegerType::get(op->getContext(), 64);
  dict.set(name, IntegerAttr::get(type, val));
  op->setAttr(tagDict(), dict.getDictionary(op->getContext()));
}

void setStringTag(mlir::Operation *op, llvm::StringRef name,
                  llvm::StringRef val) {
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  dict.set(name, StringAttr::get(val, op->getContext()));
  op->setAttr(tagDict(), dict.getDictionary(op->getContext()));
}

bool hasUnitTag(mlir::Operation *op, llvm::StringRef name) {
  auto dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  if (!dict) {
    return false;
  }
  auto attr = dict.get(name).dyn_cast_or_null<UnitAttr>();
  return static_cast<bool>(attr);
}

bool hasIntegerTag(mlir::Operation *op, llvm::StringRef name) {
  auto dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  if (!dict) {
    return false;
  }
  auto attr = dict.get(name).dyn_cast_or_null<IntegerAttr>();
  return static_cast<bool>(attr);
}

bool hasStringTag(mlir::Operation *op, llvm::StringRef name) {
  auto dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  if (!dict) {
    return false;
  }
  auto attr = dict.get(name).dyn_cast_or_null<StringAttr>();
  return static_cast<bool>(attr);
}

int64_t getIntegerTag(mlir::Operation *op, llvm::StringRef name,
                      int64_t defaultVal) {
  auto dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  if (!dict) {
    return defaultVal;
  }
  auto attr = dict.get(name).dyn_cast_or_null<IntegerAttr>();
  if (attr) {
    return attr.getInt();
  }
  return defaultVal;
}

llvm::StringRef getStringTag(mlir::Operation *op, llvm::StringRef name,
                             llvm::StringRef defaultVal) {
  auto dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  if (!dict) {
    return defaultVal;
  }
  auto attr = dict.get(name).dyn_cast_or_null<StringAttr>();
  if (attr) {
    return attr.getValue();
  }
  return defaultVal;
}

} // namespace pmlc
