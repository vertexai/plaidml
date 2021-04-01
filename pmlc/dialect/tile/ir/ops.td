// Copyright 2020 Intel Corporation

include "pmlc/dialect/tile/ir/base.td"
include "pmlc/dialect/tile/ir/eltwise.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"

#ifndef __PMLC_UTIL_ENUMS__
include "pmlc/util/enums.td"
#endif

// TODO: this should be defined upstream
def IntegerSetAttr : Attr<
    CPred<"$_self.isa<IntegerSetAttr>()">, "IntegerSet attribute"> {
  let storageType = [{ IntegerSetAttr }];
  let returnType = [{ IntegerSet }];
  let constBuilderCall = "IntegerSetAttr::get($0)";
}

def ContractionOp : TileOp<"contract", [
  NoSideEffect,
  MaterializeOperandsOpInterface
]> {
  let arguments = (ins
    EltwiseAny:$init,
    Variadic<EltwiseAny>:$operands,
    AggregationKind:$agg,
    CombinationKind:$combo,
    AffineMapAttr:$sink,
    AffineMapArrayAttr:$srcs,
    OptionalAttr<IntegerSetAttr>:$cons,
    OptionalAttr<AffineMapAttr>:$shape,
    OptionalAttr<AffineMapAttr>:$lowerBounds,
    OptionalAttr<AffineMapAttr>:$upperBounds,
    OptionalAttr<StrAttr>:$name
  );
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);

  let builders = [
    OpBuilder<(ins
      "mlir::Type":$resultType,
      "mlir::Value":$init,
      "mlir::ArrayRef<mlir::Value>":$tensors,
      "util::AggregationKind":$agg,
      "util::CombinationKind":$combo,
      "mlir::AffineMap":$sink,
      "mlir::ArrayRef<mlir::AffineMap>":$srcs,
      "mlir::IntegerSet":$cons,
      "mlir::StringRef":$name
    )>
  ];

  let extraClassDeclaration = [{
    static unsigned getNumTensors(CombinationKind combo);
    static mlir::StringRef getSinkAttrName() { return "sink"; }
    static mlir::StringRef getSourcesAttrName() { return "srcs"; }
    static mlir::StringRef getConstraintsAttrName() { return "cons"; }
    static mlir::StringRef getLowerBoundsAttrName() { return "lowerBounds"; }
    static mlir::StringRef getUpperBoundsAttrName() { return "upperBounds"; }

    unsigned getNumTensors();
    unsigned getNumSymbols();
    mlir::Value getTensor(unsigned i);
    mlir::Value getSymbol(unsigned i);
    mlir::AffineMap getSourceMap(unsigned i);

    void setLowerBounds(mlir::ArrayRef<int64_t> bounds);
    void setUpperBounds(mlir::ArrayRef<int64_t> bounds);
    void setSink(mlir::AffineMap sink);
    void setSources(mlir::ArrayRef<mlir::AffineMap> srcs);
    void setConstraints(mlir::IntegerSet cons);

    mlir::RankedTensorType getResultType() { return result().getType().cast<mlir::RankedTensorType>(); }

    mlir::LogicalResult materializeOperands(mlir::OpBuilder& builder);
  }];

  let printer = [{ print$cppClass(&p, *this); }];
  let parser = [{ return parse$cppClass(&parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

class SpecialOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, traits>;

class MaterializeOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
  Op<TileDialect, mnemonic, !listconcat(traits, [
    MaterializeOperandsOpInterface
  ])> {
  let extraClassDeclaration = [{
    mlir::LogicalResult materializeOperands(mlir::OpBuilder& builder);
  }];
}

def ArgSortOp : MaterializeOp<"argsort"> {
  let summary = "tensor sort index operation";
  let arguments = (ins
    RankedTensorOf<[AnyScalar]>:$tensor,
    IndexAttr:$axis,
    SortDirection:$direction);
  let results = (outs RankedTensorOf<[SI32]>:$result);
  let assemblyFormat = [{
    $direction $tensor `[` $axis `]` attr-dict `:` functional-type(operands, $result)
  }];
}

def GatherOp : MaterializeOp<"gather"> {
  let summary = "special gather operation";
  let arguments = (ins
    RankedTensorOf<[AnyScalar]>:$tensor,
    RankedTensorOf<[AnyScalar]>:$indices,
    DefaultValuedAttr<IndexAttr, "0">:$axis,
    DefaultValuedAttr<InterpolationMode, "InterpolationMode::linear">:$interpolationMode,
    DefaultValuedAttr<NearestMode, "NearestMode::round_prefer_floor">:$nearestMode,
    DefaultValuedAttr<F64Attr, "-0.75">:$cubeCoeff,
    DefaultValuedAttr<GatherMode, "GatherMode::normal">:$mode,
    DefaultValuedAttr<IndexAttr, "0">:$batchDims
  );
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
  let assemblyFormat = [{
    $tensor $indices attr-dict `:` functional-type(operands, $result)
  }];

  let builders = [
    OpBuilder<(ins
      "mlir::Type":$resultType,
      "mlir::ValueRange":$operands,
      "mlir::IntegerAttr":$axis,
      "mlir::IntegerAttr":$interpolationMode,
      "mlir::IntegerAttr":$nearestMode,
      "mlir::FloatAttr":$cubeCoeff,
      "mlir::IntegerAttr":$mode,
      "mlir::IntegerAttr":$batchDims
    )>
  ];
}

def IndexOp : SpecialOp<"index"> {
  let summary = "tensor index lookup operation";
  let arguments = (ins IndexAttr:$axis);
  let results = (outs EltwiseIndex:$result);
  let assemblyFormat = "$axis attr-dict `:` type($result)";
}

def PrngOp : SpecialOp<"prng"> {
  let summary = "pseudorandom number generator";
  let arguments = (ins RankedTensorOf<[UI32]>:$state);
  let results = (outs
    RankedTensorOf<[AnyFloat]>:$result,
    RankedTensorOf<[UI32]>:$new_state);
  let assemblyFormat = "$state attr-dict `:` functional-type($state, results)";
}

def ReshapeOp : MaterializeOp<"reshape">,
  HasFolder {
  let summary = "tensor reshape operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
  let assemblyFormat = [{
    $tensor attr-dict `:` functional-type($tensor, $result)
  }];
  let verifier = [{ return verify$cppClass(*this); }];
}

def ScatterOp : MaterializeOp<"scatter"> {
  let summary = "special scatter operation";
  let arguments = (ins
    RankedTensorOf<[AnyScalar]>:$data,
    RankedTensorOf<[AnyScalar]>:$indices,
    RankedTensorOf<[AnyScalar]>:$updates,
    DefaultValuedAttr<IndexAttr, "0">:$axis,
    DefaultValuedAttr<ScatterMode, "ScatterMode::normal">:$mode
  );
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
  let builders = [
    OpBuilder<(ins
      "mlir::Type":$resultType,
      "mlir::ValueRange":$operands,
      "mlir::IntegerAttr":$axis,
      "mlir::IntegerAttr":$mode
    )>
  ];

  let assemblyFormat = [{
    $data $indices $updates attr-dict `:` functional-type(operands, $result)
  }];
}

def ShapeOp : MaterializeOp<"shape"> {
  let summary = "tensor shape operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor);
  let results = (outs RankedTensorOf<[SI32]>:$result);
  let assemblyFormat = [{
    $tensor attr-dict `:` functional-type($tensor, $result)
  }];
}

def PragmaOp : MaterializeOp<"pragma", [
      AllTypesMatch<["tensor", "result"]>
    ]> {
  let summary = "pragma operation";
  let arguments = (ins AnyType:$tensor, StrAttr:$op, DictionaryAttr:$attrs);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins
      "mlir::Value":$tensor,
      "mlir::StringRef":$op,
      "mlir::DictionaryAttr":$attrs), [{
        $_state.addTypes(tensor.getType());
        $_state.addOperands(tensor);
        $_state.addAttribute("op", $_builder.getStringAttr(op));
        $_state.addAttribute("attrs", attrs);
      }]
    >
  ];

  let assemblyFormat = "$tensor $op $attrs attr-dict `:` type($tensor)";
}
