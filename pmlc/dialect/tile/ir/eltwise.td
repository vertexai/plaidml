// Copyright 2019, Intel Corporation

#ifndef __PML_TILE_ELTWISE_OPS__
#define __PML_TILE_ELTWISE_OPS__

include "pmlc/dialect/tile/ir/base.td"

class EltwiseOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, !listconcat(traits, [
      MaterializeOperandsOpInterface
    ])>;

def ConstantOp :
    Op<TileDialect, "constant", [ConstantLike, NoSideEffect]>,
    HasFolder {
  let summary = "constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs EltwiseAny:$result);

  let builders = [
    OpBuilder<(ins "mlir::Type":$type, "double":$value), [{
      $_state.addAttribute("value", $_builder.getF64FloatAttr(value));
      $_state.addTypes(RankedTensorType::get({}, type));
    }]>,
    OpBuilder<(ins "mlir::Type":$type, "int64_t":$value), [{
      $_state.addAttribute("value", $_builder.getI64IntegerAttr(value));
      $_state.addTypes(RankedTensorType::get({}, type));
    }]>,
    OpBuilder<(ins "mlir::Type":$type, "uint64_t":$value), [{
      $_state.addAttribute("value", $_builder.getI64IntegerAttr(value));
      $_state.addTypes(RankedTensorType::get({}, type));
    }]>,
  ];

  let extraClassDeclaration = [{
    mlir::Attribute getValue() { return (*this)->getAttr("value"); }
  }];

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";
}

class EW_UnaryOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EltwiseOp<mnemonic, traits> {
  let arguments = (ins EltwiseAny:$operand);
  let results = (outs EltwiseAny:$result);
  let extraClassDeclaration = [{
    mlir::LogicalResult materializeOperands(mlir::OpBuilder& builder) {
      return tile::materializeOperands(builder, getOperation());
    }
  }];
  let assemblyFormat = [{
    $operand attr-dict `:` functional-type($operand, $result)
  }];
}

class EW_BinaryOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EltwiseOp<mnemonic, traits> {
  let arguments = (ins EltwiseAny:$lhs, EltwiseAny:$rhs);
  let results = (outs EltwiseAny:$result);
  let extraClassDeclaration = [{
    mlir::LogicalResult materializeOperands(mlir::OpBuilder& builder) {
      return tile::materializeOperands(builder, getOperation());
    }
  }];
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, $result)
  }];
}

def EW_CastOp : Op<TileDialect, "cast", [NoSideEffect]>, HasFolder {
  let summary = "cast operation";
  let arguments = (ins EltwiseAny:$tensor);
  let results = (outs EltwiseAny:$result);
  let assemblyFormat = [{
    $tensor attr-dict `:` functional-type($tensor, $result)
  }];
}

class EW_CompareOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EltwiseOp<mnemonic, traits> {
  let arguments = (ins EltwiseAny:$lhs, EltwiseAny:$rhs);
  let results = (outs EltwiseBool:$result);
  let extraClassDeclaration = [{
    mlir::LogicalResult materializeOperands(mlir::OpBuilder& builder) {
      return tile::materializeOperands(builder, getOperation());
    }
  }];
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, $result)
  }];
}

class EW_UnaryIntOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EW_UnaryOp<mnemonic, traits>,
    Arguments<(ins EltwiseInteger:$operand)>,
    Results<(outs EltwiseInteger:$result)>;

class EW_UnaryFloatOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EW_UnaryOp<mnemonic, traits>,
    Arguments<(ins EltwiseFloat:$operand)>,
    Results<(outs EltwiseFloat:$result)>;

class EW_BinaryIntOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EW_BinaryOp<mnemonic, traits>,
    Arguments<(ins EltwiseInteger:$lhs, EltwiseInteger:$rhs)>,
    Results<(outs EltwiseInteger:$result)>;

class EW_BinaryLogicOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EW_BinaryOp<mnemonic, traits>,
    Arguments<(ins EltwiseAny:$lhs, EltwiseAny:$rhs)>,
    Results<(outs EltwiseBool:$result)>;

class EW_UnaryLogicOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EW_UnaryOp<mnemonic, traits>,
    Arguments<(ins EltwiseAny:$operand)>,
    Results<(outs EltwiseBool:$result)>;


def EW_AbsOp : EW_UnaryOp<"abs">;
def EW_ACosOp : EW_UnaryFloatOp<"acos">;
def EW_ACosHOp : EW_UnaryFloatOp<"acosh">;
def EW_AddOp : EW_BinaryOp<"add", [Commutative, NoSideEffect]>, HasFolder;
def EW_ASinOp : EW_UnaryFloatOp<"asin">;
def EW_ASinHOp : EW_UnaryFloatOp<"asinh">;
def EW_AssignOp : EW_UnaryOp<"assign">;
def EW_ATanOp : EW_UnaryFloatOp<"atan">;
def EW_ATanHOp : EW_UnaryFloatOp<"atanh">;
def EW_CeilOp : EW_UnaryOp<"ceil">;
def EW_CmpEqOp : EW_CompareOp<"cmp_eq", [Commutative, NoSideEffect]>;
def EW_CmpGeOp : EW_CompareOp<"cmp_ge">;
def EW_CmpGtOp : EW_CompareOp<"cmp_gt">;
def EW_CmpLeOp : EW_CompareOp<"cmp_le">;
def EW_CmpLtOp : EW_CompareOp<"cmp_lt">;
def EW_CmpNeOp : EW_CompareOp<"cmp_ne", [Commutative, NoSideEffect]>;
def EW_CosHOp : EW_UnaryFloatOp<"cosh">;
def EW_CosOp : EW_UnaryFloatOp<"cos">;
def EW_DivOp : EW_BinaryOp<"div">, HasFolder;
def EW_ErfOp : EW_UnaryFloatOp<"erf">;
def EW_ExpOp : EW_UnaryFloatOp<"exp">;
def EW_FloorOp : EW_UnaryOp<"floor">;
// Prevent CSE from dropping 'ident' by avoiding the NoSideEffect trait
def EW_IdentOp : EW_UnaryOp<"ident", []>;
def EW_LogOp : EW_UnaryFloatOp<"log">;
def EW_MaxOp : EW_BinaryOp<"max", [Commutative, NoSideEffect]>;
def EW_MinOp : EW_BinaryOp<"min", [Commutative, NoSideEffect]>;
def EW_ModOp : EW_BinaryOp<"mod">;
def EW_MulOp : EW_BinaryOp<"mul", [Commutative, NoSideEffect]>, HasFolder;
def EW_NegOp : EW_UnaryOp<"neg">;
def EW_PowOp : EW_BinaryOp<"pow">;
def EW_ReluOp : EW_UnaryOp<"relu">;
def EW_RoundOp : EW_UnaryOp<"round">;
def EW_SignOp : EW_UnaryOp<"sign">;
def EW_SinHOp : EW_UnaryFloatOp<"sinh">;
def EW_SinOp : EW_UnaryFloatOp<"sin">;
def EW_SqrtOp : EW_UnaryFloatOp<"sqrt">;
def EW_SubOp : EW_BinaryOp<"sub">, HasFolder;
def EW_TanHOp : EW_UnaryFloatOp<"tanh">;
def EW_TanOp : EW_UnaryFloatOp<"tan">;

// bitwise operations
def EW_BitAndOp : EW_BinaryIntOp<"bit_and", [Commutative, NoSideEffect]>;
def EW_BitNotOp : EW_UnaryIntOp<"bit_not">;
def EW_BitOrOp : EW_BinaryIntOp<"bit_or", [Commutative, NoSideEffect]>;
def EW_BitXorOp : EW_BinaryIntOp<"bit_xor", [Commutative, NoSideEffect]>;
def EW_BitShlOp : EW_BinaryIntOp<"bit_shl">;
def EW_BitShrOp : EW_BinaryIntOp<"bit_shr">;

// logical operations
def EW_LogicalAndOp : EW_BinaryLogicOp<"logical_and", [Commutative, NoSideEffect]>;
def EW_LogicalNotOp : EW_UnaryLogicOp<"logical_not">;
def EW_LogicalOrOp : EW_BinaryLogicOp<"logical_or", [Commutative, NoSideEffect]>;
def EW_LogicalXorOp : EW_BinaryLogicOp<"logical_xor", [Commutative, NoSideEffect]>;

def EW_SelectOp : EltwiseOp<"select", [NoSideEffect]> {
  let summary = "conditional selection";
  let arguments = (ins EltwiseBool:$cond, EltwiseAny:$tcase, EltwiseAny:$fcase);
  let results = (outs EltwiseAny:$result);
  let extraClassDeclaration = [{
    mlir::LogicalResult materializeOperands(mlir::OpBuilder& builder);
  }];
  let assemblyFormat = [{
    $cond `,` $tcase `,` $fcase attr-dict `:` functional-type(operands, $result)
  }];
}

#endif // __PML_TILE_ELTWISE_OPS__
