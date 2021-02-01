// Copyright 2019 Intel Corporation.

#pragma once

#include "plaidml/core/ffi.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  PLAIDML_INT_OP_NEG,
  PLAIDML_INT_OP_ADD,
  PLAIDML_INT_OP_SUB,
  PLAIDML_INT_OP_MUL,
  PLAIDML_INT_OP_DIV,
  PLAIDML_INT_OP_MAX,
  PLAIDML_INT_OP_MIN,
} plaidml_int_op;

typedef enum {
  PLAIDML_VALUE_NONE,
  PLAIDML_VALUE_DIM,
  PLAIDML_VALUE_EXPR,
  PLAIDML_VALUE_FLOAT,
  PLAIDML_VALUE_INT,
  PLAIDML_VALUE_STR,
  PLAIDML_VALUE_TUPLE,
} plaidml_value_kind;

void plaidml_edsl_init(  //
    plaidml_error* err);

//
// plaidml_attr
//

typedef struct {
  const char* key;
  plaidml_value* value;
} plaidml_attr;

//
// plaidml_poly_expr
//

void plaidml_poly_expr_free(  //
    plaidml_error* err,       //
    plaidml_poly_expr* expr);

plaidml_string* plaidml_poly_expr_repr(  //
    plaidml_error* err,                  //
    plaidml_poly_expr* expr);

plaidml_poly_expr* plaidml_poly_expr_dim(  //
    plaidml_error* err,                    //
    plaidml_dim_expr* expr);

plaidml_poly_expr* plaidml_poly_expr_index(  //
    plaidml_error* err,                      //
    const char* name);

plaidml_poly_expr* plaidml_poly_expr_literal(  //
    plaidml_error* err,                        //
    int64_t value);

plaidml_poly_expr* plaidml_poly_expr_op(  //
    plaidml_error* err,                   //
    plaidml_int_op op,                    //
    size_t nargs,                         //
    plaidml_poly_expr** args);

//
// plaidml_dim_expr
//

void plaidml_dim_expr_free(  //
    plaidml_error* err,      //
    plaidml_dim_expr* expr);

plaidml_string* plaidml_dim_expr_repr(  //
    plaidml_error* err,                 //
    plaidml_dim_expr* expr);

plaidml_dim_expr* plaidml_dim_expr_none(  //
    plaidml_error* err                    //
);

plaidml_dim_expr* plaidml_dim_expr_int(  //
    plaidml_error* err,                  //
    int64_t value);

plaidml_dim_expr* plaidml_dim_expr_op(  //
    plaidml_error* err,                 //
    plaidml_int_op op,                  //
    size_t nargs,                       //
    plaidml_dim_expr** args);

//
// plaidml_expr
//

void plaidml_expr_free(  //
    plaidml_error* err,  //
    plaidml_expr* expr);

void* plaidml_expr_ptr(  //
    plaidml_error* err,  //
    plaidml_expr* expr);

plaidml_datatype plaidml_expr_get_dtype(  //
    plaidml_error* err,                   //
    plaidml_expr* expr);

size_t plaidml_expr_get_rank(  //
    plaidml_error* err,        //
    plaidml_expr* expr);

plaidml_shape* plaidml_expr_get_shape(  //
    plaidml_error* err,                 //
    plaidml_expr* expr);

void plaidml_expr_bind_dims(  //
    plaidml_error* err,       //
    plaidml_expr* expr,       //
    size_t rank,              //
    plaidml_dim_expr** dims);

plaidml_string* plaidml_expr_repr(  //
    plaidml_error* err,             //
    plaidml_expr* expr);

plaidml_expr* plaidml_expr_clone(  //
    plaidml_error* err,            //
    plaidml_expr* expr);

plaidml_dim_expr* plaidml_expr_get_dim(  //
    plaidml_error* err,                  //
    plaidml_expr* expr);

plaidml_expr* plaidml_expr_dim(  //
    plaidml_error* err,          //
    plaidml_dim_expr* expr);

plaidml_expr* plaidml_expr_input(  //
    plaidml_error* err,            //
    plaidml_shape* shape,          //
    const char* name);

plaidml_expr* plaidml_expr_constant(  //
    plaidml_error* err,               //
    plaidml_buffer* buffer,           //
    const char* name);

plaidml_expr* plaidml_expr_uint(  //
    plaidml_error* err,           //
    uint64_t value);

plaidml_expr* plaidml_expr_int(  //
    plaidml_error* err,          //
    int64_t value);

plaidml_expr* plaidml_expr_float(  //
    plaidml_error* err,            //
    double value);

plaidml_expr* plaidml_expr_element(  //
    plaidml_error* err,              //
    plaidml_expr* expr,              //
    size_t ordinal);

plaidml_expr* plaidml_expr_intrinsic(  //
    plaidml_error* err,                //
    const char* fn,                    //
    size_t nargs,                      //
    plaidml_expr** args);

plaidml_expr* plaidml_expr_cast(  //
    plaidml_error* err,           //
    plaidml_expr* tensor,         //
    plaidml_datatype dtype);

plaidml_expr* plaidml_expr_pragma(  //
    plaidml_error* err,             //
    plaidml_expr* tensor,           //
    const char* op,                 //
    size_t nattrs,                  //
    plaidml_attr** attrs);

typedef struct {
  size_t size;
  plaidml_expr** elts;
} plaidml_exprs;

void plaidml_exprs_free(  //
    plaidml_error* err,   //
    plaidml_exprs* exprs);

plaidml_expr* plaidml_expr_layer_begin(  //
    plaidml_error* err,                  //
    const char* op,                      //
    size_t ninputs,                      //
    plaidml_expr** inputs,               //
    size_t nattrs,                       //
    plaidml_attr** attrs);

plaidml_exprs* plaidml_expr_layer_end(  //
    plaidml_error* err,                 //
    plaidml_expr* expr,                 //
    size_t noutputs,                    //
    plaidml_expr** outputs);

plaidml_expr* plaidml_expr_loop(  //
    plaidml_error* err,            //
    const char* op,                //
    size_t nindex,                 //
    plaidml_expr** indexs,         //
    size_t ninputs,                //
    plaidml_expr** inputs,         //
    size_t noutputs,               //
    plaidml_expr** outputs);

//
// plaidml_contraction
//

typedef enum {
  PLAIDML_AGG_OP_NONE,
  PLAIDML_AGG_OP_SUM,
  PLAIDML_AGG_OP_MAX,
  PLAIDML_AGG_OP_MIN,
  PLAIDML_AGG_OP_PROD,
  PLAIDML_AGG_OP_ASSIGN
} plaidml_agg_op;

typedef enum {
  PLAIDML_COMBO_OP_NONE,
  PLAIDML_COMBO_OP_MUL,
  PLAIDML_COMBO_OP_ADD,
  PLAIDML_COMBO_OP_EQ,
  PLAIDML_COMBO_OP_COND,
} plaidml_combo_op;

plaidml_expr* plaidml_expr_contraction(  //
    plaidml_error* err,                  //
    plaidml_agg_op agg_op,               //
    plaidml_combo_op combo_op,           //
    size_t rank,                         //
    plaidml_poly_expr** idxs,            //
    plaidml_dim_expr** dims,             //
    plaidml_expr* init,                  //
    const char* name);

void plaidml_contraction_add_operand(  //
    plaidml_error* err,                //
    plaidml_expr* expr,                //
    plaidml_expr* tensor,              //
    size_t rank,                       //
    plaidml_poly_expr** idxs);

void plaidml_contraction_add_constraint(  //
    plaidml_error* err,                   //
    plaidml_expr* expr,                   //
    plaidml_poly_expr* lhs,               //
    plaidml_dim_expr* rhs);

void plaidml_contraction_build(  //
    plaidml_error* err,          //
    plaidml_expr* expr);

//
// plaidml_value
//

typedef struct {
  size_t size;
  plaidml_value** elts;
} plaidml_tuple;

void plaidml_tuple_free(  //
    plaidml_error* err,   //
    plaidml_tuple* tuple);

void plaidml_value_free(  //
    plaidml_error* err,   //
    plaidml_value* value);

plaidml_value* plaidml_value_clone(  //
    plaidml_error* err,              //
    plaidml_value* value);

plaidml_value_kind plaidml_value_get_kind(  //
    plaidml_error* err,                     //
    plaidml_value* value);

plaidml_value* plaidml_value_none(  //
    plaidml_error* err              //
);

plaidml_value* plaidml_value_dim(  //
    plaidml_error* err,            //
    plaidml_dim_expr* expr);

plaidml_value* plaidml_value_expr(  //
    plaidml_error* err,             //
    plaidml_expr* expr);

plaidml_value* plaidml_value_float(  //
    plaidml_error* err,              //
    double value);

plaidml_value* plaidml_value_int(  //
    plaidml_error* err,            //
    int64_t value);

plaidml_value* plaidml_value_str(  //
    plaidml_error* err,            //
    const char* value);

plaidml_value* plaidml_value_tuple(  //
    plaidml_error* err,              //
    size_t nelts,                    //
    plaidml_value** elts);

plaidml_dim_expr* plaidml_value_dim_get(  //
    plaidml_error* err,                   //
    plaidml_value* value);

plaidml_expr* plaidml_value_expr_get(  //
    plaidml_error* err,                //
    plaidml_value* value);

double plaidml_value_float_get(  //
    plaidml_error* err,          //
    plaidml_value* value);

int64_t plaidml_value_int_get(  //
    plaidml_error* err,         //
    plaidml_value* value);

plaidml_string* plaidml_value_str_get(  //
    plaidml_error* err,                 //
    plaidml_value* value);

plaidml_tuple* plaidml_value_tuple_get(  //
    plaidml_error* err,                  //
    plaidml_value* value);

plaidml_string* plaidml_value_repr(  //
    plaidml_error* err,              //
    plaidml_value* value);

//
// plaidml_targets
//

plaidml_strings* plaidml_targets_get(  //
    plaidml_error* err);

//
// plaidml_build
//

plaidml_program* plaidml_build(  //
    plaidml_error* err,          //
    const char* name,            //
    size_t ninputs,              //
    plaidml_expr** inputs,       //
    plaidml_shape** shapes,      //
    size_t noutputs,             //
    plaidml_expr** outputs);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
