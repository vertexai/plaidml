// Copyright 2020 Intel Corporation

#ifndef __PML_TILE_BASE__
#define __PML_TILE_BASE__

include "mlir/Interfaces/SideEffectInterfaces.td"
include "pmlc/dialect/tile/ir/interfaces.td"

def TileDialect : Dialect {
  let name = "tile";
  let cppNamespace = "pmlc::dialect::tile";
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
    static std::string getDialectAttrName(llvm::StringRef name);
    static std::string getCanonicalOpName(llvm::StringRef name);
  }];
}

class TileOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, traits> {}

class HasFolder {
  bit hasFolder = 1;
}

def APFloatType : DialectType<TileDialect,
  CPred<"$_self.isa<tile::APFloatType>()">,"fx">;

def APSignedIntegerType : DialectType<TileDialect,
  CPred<"$_self.isa<tile::APSignedIntegerType>()">, "six">;

def APUnsignedIntegerType : DialectType<TileDialect,
  CPred<"$_self.isa<tile::APUnsignedIntegerType>()">, "uix">;

def AnyScalar : Type<Or<[
    Index.predicate,
    AnyFloat.predicate,
    I1.predicate,
    AnySignedInteger.predicate,
    AnyUnsignedInteger.predicate,
    APFloatType.predicate,
    APSignedIntegerType.predicate,
    APUnsignedIntegerType.predicate
  ]>, "scalar">;

def EltwiseAny : Type<RankedTensorOf<[AnyScalar]>.predicate,
  "eltwise-any">;

def EltwiseBool : Type<RankedTensorOf<[I1]>.predicate,
  "eltwise-bool">;

def EltwiseFloat : Type<RankedTensorOf<[AnyFloat, APFloatType]>.predicate,
  "eltwise-float">;

def EltwiseSigned : Type<
  RankedTensorOf<[AnySignedInteger, APSignedIntegerType]>.predicate,
  "eltwise-signed">;

def EltwiseUnsigned : Type<
  RankedTensorOf<[AnyUnsignedInteger, APUnsignedIntegerType]>.predicate,
  "eltwise-unsigned">;

def EltwiseInteger : Type<RankedTensorOf<[
    AnySignedInteger,
    AnyUnsignedInteger,
    APSignedIntegerType,
    APUnsignedIntegerType
  ]>.predicate,
  "eltwise-integer">;

def EltwiseIndex : Type<RankedTensorOf<[SI32]>.predicate, "eltwise-index">;

def SortDirection : I64EnumAttr<
    "SortDirection",
    "Direction for sorting operation",
    [
      I64EnumAttrCase<"asc", 0>,
      I64EnumAttrCase<"desc", 1>,
    ]> {
  let cppNamespace = "pmlc::dialect::tile";
}

#endif // __PML_TILE_BASE__
