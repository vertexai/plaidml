// Copyright 2020 Intel Corporation

#ifndef __PML_STDX_OPS__
#define __PML_STDX_OPS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"

def StdX_Dialect : Dialect {
  let name = "stdx";
  let cppNamespace = "pmlc::dialect::stdx";
}

def Argpack : DialectType<StdX_Dialect, CPred<"$_self.isa<ArgpackType>()">, "Argpack type">,
        BuildableType<"$_builder.getType<ArgpackType>()">;

class StdX_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<StdX_Dialect, mnemonic, traits> {
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

def ReshapeOp : StdX_Op<"reshape",
  [DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "tensor reshape operation";
  let arguments = (ins AnyMemRef:$tensor);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    `(` $tensor `)` attr-dict `:` functional-type($tensor, results)
  }];
  let verifier = ?;
}

class StdX_UnaryFuncOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<StdX_Dialect, mnemonic, traits> {
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType);
  let assemblyFormat = [{
    `(` $value `)` attr-dict `:` functional-type(operands, results)
  }];
  let verifier = ?;
}

def ACosOp : StdX_UnaryFuncOp<"acos"> {
  let summary = "arccosine";
}

def ACosHOp : StdX_UnaryFuncOp<"acosh"> {
  let summary = "hyperbolic arccosine";
}

def ASinOp : StdX_UnaryFuncOp<"asin"> {
  let summary = "arcsine";
}

def ASinHOp : StdX_UnaryFuncOp<"asinh"> {
  let summary = "hyperbolic arcsine";
}

def ATanOp : StdX_UnaryFuncOp<"atan"> {
  let summary = "arctangent";
}

def ATanHOp : StdX_UnaryFuncOp<"atanh"> {
  let summary = "hyperbolic arctangent";
}

def CosHOp : StdX_UnaryFuncOp<"cosh"> {
  let summary = "hyperbolic cosine";
}

def ErfOp : StdX_UnaryFuncOp<"erf"> {
  let summary = "Gauss error function";
}

def FloorOp : StdX_UnaryFuncOp<"floor"> {
  let summary = "integer floor";
}

def PowOp : StdX_Op<"pow"> {
  let summary = "raise to power";
  let arguments = (ins AnyType:$base, AnyType:$exponent);
  let results = (outs AnyType);
  let assemblyFormat = [{
    `(` $base `,` $exponent `)` attr-dict `:` functional-type(operands, results)
  }];
  let verifier = ?;
}

def RoundOp : StdX_UnaryFuncOp<"round"> {
  let summary = "round to nearest integer";
}

def SinHOp : StdX_UnaryFuncOp<"sinh"> {
  let summary = "hyperbolic sine";
}

def TanOp : StdX_UnaryFuncOp<"tan"> {
  let summary = "tangent";
}

def AnyStdScalar : AnyTypeOf<[AnyFloat, AnySignlessInteger]> {}
def AnyVectorOrScalar : AnyTypeOf<[AnyStdScalar, AnyVector]> {}

def SubgroupBroadcastOp : StdX_Op<"subgroup_broadcast", [NoSideEffect, AllTypesMatch<["value", "result"]>]> {
  let summary = "broadcast to all elements in subgroup";
  let arguments = (ins AnyStdScalar:$value, Index:$localid);
  let results = (outs AnyStdScalar:$result);
  let assemblyFormat = [{
    `(` $value `,` $localid `)` attr-dict `:` type($value)
  }];
  let verifier = ?;
}

def SubgroupBlockReadINTELOp : StdX_Op<"subgroup_block_read_intel", [NoSideEffect]> {
  let summary = "See intel_subgroups extension.";
  let arguments = (ins Arg<AnyMemRef, "the reference to subgroup block read from",
                           [MemRead]>:$memref,
                           Variadic<Index>:$indices);
  let results = (outs AnyVectorOrScalar:$result);

  let builders = [
    OpBuilder<(ins
      "mlir::Value":$memref,
      CArg<"mlir::ValueRange", "{}">:$indices
    )>
  ];

  let extraClassDeclaration = [{
    mlir::MemRefType getMemRefType() {
      return memref().getType().cast<mlir::MemRefType>();
    }
  }];

  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref)`,` type($result)";
}

def SubgroupBlockWriteINTELOp : StdX_Op<"subgroup_block_write_intel", [NoSideEffect]> {
  let summary = "See intel_subgroups extension.";

  let arguments = (ins AnyVectorOrScalar:$value,
                       Arg<AnyMemRef, "the reference to subgroup block write to",
                       [MemWrite]>:$memref,
                       Variadic<Index>:$indices);

  let builders = [
    OpBuilder<(ins "mlir::Value":$valueToStore, "mlir::Value":$memref)>
  ];

  let extraClassDeclaration = [{
      mlir::MemRefType getMemRefType() {
        return memref().getType().cast<mlir::MemRefType>();
      }
  }];

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($value) `,` type($memref)
  }];
}

def PackOp : Op<StdX_Dialect, "pack"> {
  let summary = "Pack multiple values into a tuple";
  let arguments = (ins Variadic<AnyType>:$in);
  let results = (outs Argpack:$out);
  let assemblyFormat = [{ $in `:` type($in) attr-dict }];
}

def UnpackOp : Op<StdX_Dialect, "unpack"> {
  let summary = "Pack multiple values into a tuple";
  let arguments = (ins Argpack:$in);
  let results = (outs Variadic<AnyType>:$out);
  let assemblyFormat = [{ $in `:` type($out) attr-dict }];
}

#endif // __PML_STDX_OPS__
