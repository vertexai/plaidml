// Copyright 2019, Intel Corporation

#ifndef __PML_PXA_OPS__
#define __PML_PXA_OPS__

include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "pmlc/util/enums.td"
include "interfaces.td"

def PXA_Dialect : Dialect {
  let name = "pxa";
  let cppNamespace = "pmlc::dialect::pxa";
}

class PXA_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<PXA_Dialect, mnemonic, traits>;

class PXA_OpWithPP<string mnemonic, list<OpTrait> traits = []> :
    PXA_Op<mnemonic, traits> {
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def AnyStdScalar : AnyTypeOf<[AnyFloat, AnySignlessInteger]> {}

class PxaReduceOpBase<string mnemonic, list<OpTrait> traits = []> :
    PXA_OpWithPP<mnemonic, !listconcat(traits,
        [DeclareOpInterfaceMethods<PxaReduceOpInterface>,
        DeclareOpInterfaceMethods<AffineMapAccessInterface>,
        MemRefsNormalizable])> {

  code extraClassDeclarationBase = [{
    mlir::Value getMemRef() { return memref(); }
    unsigned getMemRefOperandIndex() { return 1; }
    void setMemRef(mlir::Value value) { setOperand(getMemRefOperandIndex(), value); }
    mlir::MemRefType getMemRefType() { return memref().getType().cast<mlir::MemRefType>(); }
    mlir::AffineMap getAffineMap() { return map(); }
    operand_range getMapOperands() { return idxs(); }
    static mlir::StringRef getMapAttrName() { return "map"; }
    mlir::AffineMapAttr getAffineMapAttr() {
      return (*this)->getAttr(getMapAttrName()).cast<mlir::AffineMapAttr>();
    }
    mlir::AtomicRMWKind getAgg() { return agg(); }
  }];
}

def PxaReduceOp : PxaReduceOpBase<"reduce"> {
  let summary = "affine reduction operation";
  let arguments = (ins
    AtomicRMWKindAttr:$agg,
    AnyType:$val,
    AnyMemRef:$memref,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins
      "mlir::AtomicRMWKind":$agg,
      "mlir::Value":$val,
      "mlir::Value":$mem,
      "mlir::AffineMap":$map,
      "mlir::ValueRange":$idxs
    ), [{
      build($_builder, $_state, mem.getType(), agg, val, mem, map, idxs);
    }]>
  ];

  let extraClassDeclaration =  extraClassDeclarationBase # [{
    mlir::Value getValueToStore() { return val(); }
  }];
}

def PxaGemmOp : PXA_OpWithPP<"gemm"> {
  let arguments = (ins
    Arg<AnyMemRef, "the C memref to store to", [MemWrite]>:$c,
    AffineMapAttr:$cAccessMap,
    AffineMapAttr:$cTileMap,
    Arg<AnyMemRef, "the A memref to load from", [MemRead]>:$a,
    AffineMapAttr:$aAccessMap,
    AffineMapAttr:$aTileMap,
    Arg<AnyMemRef, "the B memref to load from", [MemRead]>:$b,
    AffineMapAttr:$bAccessMap,
    AffineMapAttr:$bTileMap,
    I64ArrayAttr:$tile,
    I64ArrayAttr:$numBatches,
    Variadic<Index>:$mapOperands);
  let results = (outs AnyMemRef:$out);

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

def PxaVectorReduceOp : PxaReduceOpBase<"vector_reduce"> {
  let summary = "affine vector reduction operation";
  let arguments = (ins
    AtomicRMWKindAttr:$agg,
    AnyVector:$vector,
    AnyMemRef:$memref,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins
      "mlir::AtomicRMWKind":$agg,
      "mlir::Value":$vector,
      "mlir::Value":$mem,
      "mlir::AffineMap":$map,
      "mlir::ValueRange":$idxs
    ), [{
      build($_builder, $_state, mem.getType(), agg, vector, mem, map, idxs);
    }]>
  ];

  let extraClassDeclaration = extraClassDeclarationBase # [{
    mlir::Value getValueToStore() { return vector(); }
    mlir::VectorType getVectorType() {
      return vector().getType().cast<mlir::VectorType>();
    }
  }];
}

class PxaLoadOpBase<string mnemonic, list<OpTrait> traits = []> :
    PXA_OpWithPP<mnemonic, !listconcat(traits,
        [DeclareOpInterfaceMethods<PxaReadOpInterface>,
        DeclareOpInterfaceMethods<AffineMapAccessInterface>,
        MemRefsNormalizable])> {
  let arguments = (ins
    AnyMemRef:$memref,
    Variadic<Index>:$indices
  );

  code extraClassDeclarationBase = [{
    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    void setMemRef(mlir::Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// Returns the affine map used to index the memref for this operation.
    mlir::AffineMapAttr getAffineMapAttr() {
      return (*this)->getAttr(getMapAttrName()).cast<mlir::AffineMapAttr>();
    }

    static mlir::StringRef getMapAttrName() { return "map"; }
  }];
}

def PxaLoadOp : PxaLoadOpBase<"load", [NoSideEffect]> {
  let summary = "affine load operation";
  let description = [{
    The "affine.load" op reads an element from a memref, where the index
    for each memref dimension is an affine expression of loop induction
    variables and symbols. The output of 'affine.load' is a new value with the
    same type as the elements of the memref. An affine expression of loop IVs
    and symbols must be specified for each dimension of the memref. The keyword
    'symbol' can be used to indicate SSA identifiers which are symbolic.

    Example 1:

    ```mlir
    %1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
    ```

    Example 2: Uses 'symbol' keyword for symbols '%n' and '%m'.

    ```mlir
    %1 = affine.load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>
    ```
  }];

  let results = (outs AnyType:$result);

  let builders = [
    /// Builds an affine load op with the specified map and operands.
    OpBuilder<(ins
      "mlir::AffineMap":$map,
      "mlir::ValueRange":$operands
    )>,
    /// Builds an affine load op with an identity map and operands.
    OpBuilder<(ins
      "mlir::Value":$memref,
      CArg<"mlir::ValueRange", "{}">:$indices
    )>,
    /// Builds an affine load op with the specified map and its operands.
    OpBuilder<(ins
      "mlir::Value":$memref,
      "mlir::AffineMap":$map,
      "mlir::ValueRange":$mapOperands
    )>
  ];

  let extraClassDeclaration = extraClassDeclarationBase;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def PxaVectorLoadOp : PxaLoadOpBase<"vector_load"> {
  let summary = "affine vector load operation";
  let description = [{
    The "affine.vector_load" is the vector counterpart of
    [affine.load](#affineload-operation). It reads a slice from a
    [MemRef](../LangRef.md#memref-type), supplied as its first operand,
    into a [vector](../LangRef.md#vector-type) of the same base elemental type.
    The index for each memref dimension is an affine expression of loop induction
    variables and symbols. These indices determine the start position of the read
    within the memref. The shape of the return vector type determines the shape of
    the slice read from the memref. This slice is contiguous along the respective
    dimensions of the shape. Strided vector loads will be supported in the future.
    An affine expression of loop IVs and symbols must be specified for each
    dimension of the memref. The keyword 'symbol' can be used to indicate SSA
    identifiers which are symbolic.

    Example 1: 8-wide f32 vector load.

    ```mlir
    %1 = affine.vector_load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>, vector<8xf32>
    ```

    Example 2: 4-wide f32 vector load. Uses 'symbol' keyword for symbols '%n' and '%m'.

    ```mlir
    %1 = affine.vector_load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>, vector<4xf32>
    ```

    Example 3: 2-dim f32 vector load.

    ```mlir
    %1 = affine.vector_load %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
    ```

    TODOs:
    * Add support for strided vector loads.
    * Consider adding a permutation map to permute the slice that is read from memory
    (see [vector.transfer_read](../Vector/#vectortransfer_read-vectortransferreadop)).
  }];

  let results = (outs AnyVector:$result);

  let builders = [
    OpBuilder<(ins
      "mlir::VectorType":$type,
      "mlir::Value":$memref,
      "mlir::AffineMap":$map,
      "mlir::ValueRange":$mapOperands
    )>
  ];

  let extraClassDeclaration = extraClassDeclarationBase # [{
    mlir::VectorType getVectorType() {
      return result().getType().cast<mlir::VectorType>();
    }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def PxaStoreOp : PXA_OpWithPP<"store", [MemRefsNormalizable]> {
  let summary = "PXA non-affine store operation";

  let arguments = (ins
      AtomicRMWKindAttr:$agg,
      AnyTypeOf<[AnySignlessInteger, AnyFloat, AnyVector]>:$value,
      MemRefOf<[AnySignlessInteger, AnyFloat]>:$memref,
      Variadic<Index>:$indices);
  let results = (outs MemRefOf<[AnySignlessInteger, AnyFloat]>:$result);

  let assemblyFormat = [{
    $agg $value `,` $memref `[` $indices `]` attr-dict `:` `(` type($value) `,`
    type($memref) `)` `->` type($result)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins
      "mlir::AtomicRMWKind":$agg,
      "mlir::Value":$val,
      "mlir::Value":$mem,
      "mlir::ValueRange":$idxs
    ), [{
      build($_builder, $_state, mem.getType(), agg, val, mem, idxs);
    }]>
  ];
}

// def AffineGreyboxOp : PXA_OpWithPP<"greybox", [
//     AttrSizedOperandSegments,
//     IsolatedFromAbove]> {
//   let summary = "affine greybox operation";
//   let arguments = (ins
//     Variadic<AnyMemRef>:$reads,
//     Variadic<AnyMemRef>:$writes,
//     Variadic<AnyType>:$operands
//   );
//   let results = (outs Variadic<AnyMemRef>:$results);
// }

// TODO: replace PrngOp with an AffineGreyboxOp
def PrngOp : PXA_Op<"prng"> {
  let summary = "pseudorandom number generator operation";
  let arguments = (ins AnyMemRef:$state, AnyMemRef:$tensor, AnyMemRef:$new_state);
  let results = (outs AnyMemRef:$result_tensor, AnyMemRef:$result_state);
}

#endif // __PML_PXA_OPS__
