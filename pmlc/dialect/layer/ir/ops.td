// Copyright 2020 Intel Corporation

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def LayerDialect : Dialect {
  let name = "layer";
  let cppNamespace = "pmlc::dialect::layer";
}

class LayerOp<string mnemonic, list<OpTrait> traits = []>
    : Op<LayerDialect, mnemonic, traits> {
}


def BoxOp : LayerOp<"box", [IsolatedFromAbove]> {
  let summary = "layer container operation";
  let arguments = (ins
    Variadic<AnyType>:$operands,
    StrAttr:$op,
    DictionaryAttr:$attrs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "mlir::StringRef":$op,
      "mlir::ArrayRef<mlir::Value>":$operands,
      "mlir::ArrayRef<mlir::Type>":$resultTypes,
      "mlir::DictionaryAttr":$attrs
    )>
  ];

  let extraClassDeclaration = [{
    mlir::Block *getBody();
  }];

  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def ReturnOp : LayerOp<"return", [
  NoSideEffect,
  ReturnLike,
  Terminator,
  HasParent<"BoxOp">
]> {
  let summary = "Return results from a layer";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
