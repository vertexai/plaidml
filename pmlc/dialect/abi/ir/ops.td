// Copyright 2020 Intel Corporation

#ifndef __PML_ABI_OPS__
#define __PML_ABI_OPS__

include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "pmlc/util/enums.td"

def ABI_Dialect : Dialect {
  let name = "abi";
  let cppNamespace = "pmlc::dialect::abi";
}

class ABI_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ABI_Dialect, mnemonic, traits>;

def ABI_LoopOp : ABI_Op<"loop", [
  IsolatedFromAbove, HasParent<"ModuleOp">, RecursiveSideEffects]> {
  let summary = "Repeat runs of a network";
  let description = [{
    The "abi.loop" operation represents repeated runs of a program,
    taking new inputs and generating new outputs each time.
    This allows loop-invariant code motion to move setup and teardown
    operations outside of the loop, so that their cost is ammortized over
    many runs of the program.
  }];
  let regions = (region AnyRegion:$initRegion, AnyRegion:$bodyRegion, AnyRegion:$finiRegion);
  let arguments = (ins TypeArrayAttr:$networkFieldTypes);
  let builders = [OpBuilder<(ins)>];
  let skipDefaultBuilders = 1;
  let assemblyFormat = [{
     `init` $initRegion `yield` $networkFieldTypes `body` $bodyRegion `fini` $finiRegion attr-dict
   }];
  let extraClassDeclaration = [{
    std::vector<mlir::Type> getNetworkFieldTypes();
    unsigned getNumNetworkFields();
    void setNetworkFieldTypes(mlir::TypeRange types);

    mlir::Block* getBodyEntryBlock();
    mlir::Block* getFiniEntryBlock();

    YieldOp getInitTerminator();
    TerminatorOp getFiniTerminator();
  }];
}

def ABI_TerminatorOp : ABI_Op<"terminator", [NoSideEffect, Terminator, HasParent<"LoopOp">]> {
  let summary = "Terminates an abi.loop operation body or fini region";
  let assemblyFormat = [{
    attr-dict
  }];
}

def ABI_YieldOp : ABI_Op<"yield", [Terminator, HasParent<"LoopOp">]> {
  let summary = "Terminates an abi.loop operation init region";
  let description = [{
    The "abi.yield" operation terminates the init region of an abi.loop.
    Its arguments are the values that are carried into the body region
    in order to actually run the network, and into the fini region to
    de-allocate resources previously allocated in the init region.
  }];
  let arguments = (ins Variadic<AnyType>);
  let assemblyFormat = [{
    operands attr-dict `:` type(operands)
  }];
}

#endif // __PML_ABI_OPS__
