// Copyright 2020 Intel Corporation

#ifndef __PML_XSMM_OPS__
#define __PML_XSMM_OPS__

include "mlir/Interfaces/SideEffectInterfaces.td"

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def XSMM_Dialect : Dialect {
  let name = "xsmm";
  let cppNamespace = "pmlc::dialect::xsmm";
}

class XSMM_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<XSMM_Dialect, mnemonic, traits> {
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def XSMM_GemmDispatchF32Op : XSMM_Op<"gemm.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let printer = ?;
  let parser = ?;
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_GemmInvokeF32Op : XSMM_Op<"gemm.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

def XSMM_BRGemmDispatchF32Op : XSMM_Op<"brgemm.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let printer = ?;
  let parser =  ?;
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_BRGemmInvokeF32Op : XSMM_Op<"brgemm.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    I64Attr:$numBatches,
    Variadic<Index>:$indices);

  let printer = ?;
  let parser = ?;

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

def XSMM_BRGemmOffsDispatchF32Op : XSMM_Op<"brgemm.offs.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let printer = ?;
  let parser =  ?;
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}


def XSMM_BRGemmOffsInvokeF32Op : XSMM_Op<"brgemm.offs.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    I64Attr:$numBatches,
    I64ArrayAttr:$aOffsets,
    I64ArrayAttr:$bOffsets,
    Variadic<Index>:$indices);
 
  let printer = ?;
 
  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

#endif // __PML_XSMM_OPS__
