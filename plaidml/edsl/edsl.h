// Copyright 2019 Intel Corporation.

#pragma once

#include <algorithm>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "plaidml/core/core.h"
#include "plaidml/edsl/ffi.h"

namespace plaidml {
namespace edsl {

struct Constraint;
class Contraction;
class IndexedTensor;
class Tensor;
class TensorDim;
class TensorIndex;
class Value;

namespace details {

struct Deleter {
  void operator()(plaidml_dim_expr* ptr) { ffi::call_void(plaidml_dim_expr_free, ptr); }
  void operator()(plaidml_expr* ptr) { ffi::call_void(plaidml_expr_free, ptr); }
  void operator()(plaidml_exprs* ptr) { ffi::call_void(plaidml_exprs_free, ptr); }
  void operator()(plaidml_integers* ptr) { ffi::call_void(plaidml_integers_free, ptr); }
  void operator()(plaidml_poly_expr* ptr) { ffi::call_void(plaidml_poly_expr_free, ptr); }
  void operator()(plaidml_strings* ptr) { ffi::call_void(plaidml_strings_free, ptr); }
  void operator()(plaidml_tuple* ptr) { ffi::call_void(plaidml_tuple_free, ptr); }
  void operator()(plaidml_value* ptr) { ffi::call_void(plaidml_value_free, ptr); }
};

template <typename T>
inline std::shared_ptr<T> make_ptr(T* ptr) {
  return std::shared_ptr<T>(ptr, Deleter{});
}

template <typename T>
void into_vector(std::vector<T>*) {}

template <typename T, typename Head, typename... Tail>
void into_vector(std::vector<T>* into, Head&& head, Tail&&... tail) {
  into->emplace_back(std::forward<Head>(head));
  into_vector(into, std::forward<Tail>(tail)...);
}

}  // namespace details

///
/// Initializes the PlaidML EDSL API.
///
inline void init() { plaidml::init(); }

///
/// Lists the available targets.
///
inline std::vector<std::string> list_targets() {
  auto strs = details::make_ptr(ffi::call<plaidml_strings*>(plaidml_targets_get));
  std::vector<std::string> ret(strs->size);
  for (size_t i = 0; i < ret.size(); i++) {
    ret[i] = ffi::str(strs->elts[i]);
  }
  return ret;
}

///
/// \defgroup edsl_objects Objects
///

///
/// \ingroup edsl_objects
/// \class TensorDim
/// A symbolic object used to specify the dimensions of a Tensor
///
class TensorDim {
 public:
  ///
  /// TensorDim constructor
  ///
  TensorDim(edsl_source_location loc = edsl_source_location::current())  // NOLINT
      : ptr_(details::make_ptr(ffi::call<plaidml_dim_expr*>(loc, plaidml_dim_expr_none))) {}

  ///
  /// TensorDim constructor
  ///
  explicit TensorDim(const std::shared_ptr<plaidml_dim_expr>& ptr) : ptr_(ptr) {}

  ///
  /// TensorDim constructor
  ///
  explicit TensorDim(int64_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_dim_expr*>(loc, plaidml_dim_expr_int, value))) {}

  TensorDim(plaidml_int_op op, const std::vector<TensorDim>& args,
            edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(makeOp(op, args, loc))) {}

  ///
  /// Represents a subtraction operator overload.
  ///
  TensorDim operator-() const;

  ///
  /// Returns the TensorDim as a string.
  ///
  std::string str(edsl_source_location loc = edsl_source_location::current()) const {
    return ffi::str(ffi::call<plaidml_string*>(loc, plaidml_dim_expr_repr, as_ptr()));
  }

  plaidml_dim_expr* as_ptr() const { return ptr_.get(); }

 private:
  static plaidml_dim_expr* makeOp(plaidml_int_op op, const std::vector<TensorDim>& args,
                                  edsl_source_location loc = edsl_source_location::current()) {
    std::vector<plaidml_dim_expr*> operands;
    for (const auto& arg : args) {
      operands.push_back(arg.as_ptr());
    }
    return ffi::call<plaidml_dim_expr*>(loc, plaidml_dim_expr_op, op, operands.size(), operands.data());
  }

 private:
  std::shared_ptr<plaidml_dim_expr> ptr_;
};

///
/// \ingroup edsl_objects
/// \class TensorIndex
/// A symbolic object used to directly index a Tensor or to compute a Tensor's index as part of a formula.
///
class TensorIndex {
 public:
  ///
  /// TensorIndex constructor
  ///
  explicit TensorIndex(int64_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_poly_expr*>(loc, plaidml_poly_expr_literal, value))) {}

  ///
  /// TensorIndex constructor
  ///
  explicit TensorIndex(const std::string& name = "", edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_poly_expr*>(loc, plaidml_poly_expr_index, name.c_str()))) {}

  TensorIndex(plaidml_int_op op, const std::vector<TensorIndex>& args,
              edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(makePolyOp(op, args, loc))) {}

  TensorIndex(plaidml_int_op op, const TensorIndex& idx, const TensorDim& dim, bool lhs_first,
              edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(makeDimPolyOp(op, idx, dim, lhs_first, loc))) {}

  ///
  /// Represents an subtraction operator overload on a TensorIndex
  ///
  TensorIndex operator-() const;

  ///
  /// TODO
  ///
  Constraint operator<(int64_t rhs) const;

  ///
  /// TODO
  ///
  Constraint operator<(const TensorDim& rhs) const;

  ///
  /// Returns the TensorIndex as a string.
  ///
  std::string str(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::str(ffi::call<plaidml_string*>(loc, plaidml_poly_expr_repr, as_ptr()));
  }

  plaidml_poly_expr* as_ptr() const { return ptr_.get(); }

 private:
  static plaidml_poly_expr* makePolyOp(plaidml_int_op op, const std::vector<TensorIndex>& args,
                                       edsl_source_location loc = edsl_source_location::current()) {
    std::vector<plaidml_poly_expr*> operands;
    for (const auto& arg : args) {
      operands.push_back(arg.as_ptr());
    }
    return ffi::call<plaidml_poly_expr*>(loc, plaidml_poly_expr_op, op, operands.size(), operands.data());
  }

  static plaidml_poly_expr* makeDimPolyOp(plaidml_int_op op, const TensorIndex& idx, const TensorDim& dim,
                                          bool lhs_first, edsl_source_location loc = edsl_source_location::current()) {
    std::vector<plaidml_poly_expr*> operands;
    auto* dim_ptr = ffi::call<plaidml_poly_expr*>(loc, plaidml_poly_expr_dim, dim.as_ptr());
    if (lhs_first) {
      operands.emplace_back(idx.as_ptr());
      operands.emplace_back(dim_ptr);
    } else {
      operands.emplace_back(dim_ptr);
      operands.emplace_back(idx.as_ptr());
    }
    return ffi::call<plaidml_poly_expr*>(loc, plaidml_poly_expr_op, op, operands.size(), operands.data());
  }

 private:
  std::shared_ptr<plaidml_poly_expr> ptr_;
};

///
/// \ingroup edsl_objects
/// \struct Constraint
/// This is a constraint.
///
struct Constraint {
  ///
  /// lhs
  ///
  TensorIndex lhs;

  ///
  /// rhs
  ///
  TensorDim rhs;
};

class TensorLens {
 public:
  TensorLens() = default;

  TensorLens(const std::string& source, const std::string& target,
             edsl_source_location loc = edsl_source_location::current());

  template <typename T>
  std::vector<T> apply(const std::vector<T>& dims, edsl_source_location loc = edsl_source_location::current()) const;

 private:
  std::vector<size_t> map;
};

///
/// \ingroup edsl_objects
/// \class Tensor
/// A multidimensional array of a fixed shape.
///
class Tensor {
 public:
  ///
  /// Tensor constructor
  ///
  Tensor() = default;

  explicit Tensor(plaidml_expr* ptr) : ptr_(details::make_ptr(ptr)) {}

  ///
  /// Tensor constructor
  /// \param value int
  /// \return Tensor
  ///
  explicit Tensor(int value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(loc, plaidml_expr_int, value))) {}

  ///
  /// Tensor constructor
  /// \param value unsigned int
  /// \return Tensor
  ///
  explicit Tensor(unsigned value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(loc, plaidml_expr_int, value))) {}

  ///
  /// Tensor constructor
  /// \param value uint64_t
  /// \return Tensor
  ///
  explicit Tensor(uint64_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(loc, plaidml_expr_uint, value))) {}

  ///
  /// Tensor constructor
  /// \param value int64_t
  /// \return Tensor
  ///
  explicit Tensor(int64_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(loc, plaidml_expr_int, value))) {}

  ///
  /// Tensor constructor
  /// \param value double
  /// \return Tensor
  ///
  explicit Tensor(double value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(loc, plaidml_expr_float, value))) {}

  ///
  /// Tensor constructor
  ///
  explicit Tensor(const TensorDim& dim, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(loc, plaidml_expr_dim, dim.as_ptr()))) {}

  template <typename... Ts>
  IndexedTensor operator()(Ts... idxs) const;
  IndexedTensor operator()(const std::vector<TensorIndex>& idxs,
                           edsl_source_location loc = edsl_source_location::current()) const;

  ///
  /// Represents an eltwise negation
  ///
  Tensor operator-() const;

  ///
  /// Represents an eltwise bit_not
  ///
  Tensor operator~() const;

  ///
  /// Elementwise logical not
  ///
  Tensor operator!() const;

  ///
  /// TODO
  ///
  std::string str(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::str(ffi::call<plaidml_string*>(loc, plaidml_expr_repr, as_ptr()));
  }

  ///
  /// Return the tensor's shape
  ///
  TensorShape compute_shape(edsl_source_location loc = edsl_source_location::current()) const {
    return TensorShape(ffi::call<plaidml_shape*>(loc, plaidml_expr_get_shape, as_ptr()));
  }

  DType dtype(edsl_source_location loc = edsl_source_location::current()) const {
    return static_cast<DType>(ffi::call<plaidml_datatype>(loc, plaidml_expr_get_dtype, as_ptr()));
  }

  size_t rank(edsl_source_location loc = edsl_source_location::current()) const {
    return ffi::call<size_t>(loc, plaidml_expr_get_rank, as_ptr());
  }

  ///
  /// Verify that the specified dims match the dims of this tensor.
  ///
  void bind_dims(const std::vector<TensorDim>& dims, edsl_source_location loc = edsl_source_location::current()) const {
    std::vector<plaidml_dim_expr*> raw_dims(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      raw_dims[i] = dims[i].as_ptr();
    }
    raw_dims = lens_.apply(raw_dims, loc);
    ffi::call_void(loc, plaidml_expr_bind_dims, as_ptr(), raw_dims.size(), raw_dims.data());
  }

  ///
  /// TODO
  ///
  template <typename... Ts>
  void bind_dims(Ts... dims) const {
    std::vector<TensorDim> vec;
    details::into_vector(&vec, std::forward<Ts>(dims)...);
    bind_dims(vec);
  }

  ///
  /// Get an element of an operation that returns a tuple (i.e. multiple results).
  ///
  Tensor element(size_t ordinal, edsl_source_location loc = edsl_source_location::current()) const;

  Tensor use(const TensorLens& lens) const { return Tensor(*this, lens); }

  plaidml_expr* as_ptr() const { return ptr_.get(); }

  void* raw_ptr(edsl_source_location loc = edsl_source_location::current()) const {
    return ffi::call<void*>(loc, plaidml_expr_ptr, as_ptr());
  }

 private:
  Tensor(const Tensor& rhs, const TensorLens& lens) : ptr_(rhs.ptr_), lens_(lens) {}

 private:
  std::shared_ptr<plaidml_expr> ptr_;
  TensorLens lens_;
};

using TensorVec = std::vector<Tensor>;

///
/// \ingroup edsl_objects
/// \class IndexedTensor
/// This is an IndexedTensor
///
class IndexedTensor {
  friend class Tensor;
  friend class Contraction;

 public:
  IndexedTensor() : op_(PLAIDML_COMBO_OP_NONE) {}

  IndexedTensor(const Tensor& ref, const std::vector<TensorIndex>& idxs)
      : op_(PLAIDML_COMBO_OP_NONE), ref_(ref), idxs_(idxs) {}

  IndexedTensor(plaidml_combo_op op, const std::vector<IndexedTensor>& operands) : op_(op), operands_(operands) {}

  ///
  /// Performs an addition combination within a contraction.
  ///
  IndexedTensor operator+(const IndexedTensor& rhs) const { return IndexedTensor(PLAIDML_COMBO_OP_ADD, {*this, rhs}); }

  ///
  /// Performs a multiplication combination within a contraction.
  ///
  IndexedTensor operator*(const IndexedTensor& rhs) const { return IndexedTensor(PLAIDML_COMBO_OP_MUL, {*this, rhs}); }

  ///
  /// Performs an equality comparison combination within a contraction.
  ///
  IndexedTensor operator==(const IndexedTensor& rhs) const { return IndexedTensor(PLAIDML_COMBO_OP_EQ, {*this, rhs}); }

 private:
  plaidml_combo_op op_;
  std::vector<IndexedTensor> operands_;
  Tensor ref_;
  std::vector<TensorIndex> idxs_;
};

// Performs a conditional combination within a contraction.
inline IndexedTensor cond(const IndexedTensor& lhs, const IndexedTensor& rhs, const IndexedTensor& true_case) {
  return IndexedTensor(PLAIDML_COMBO_OP_COND, {lhs, rhs, true_case});
}

class Contraction {
 public:
  explicit Contraction(const TensorLens& lens, const std::string& name = "") : name_(name), lens_(lens) {}

  explicit Contraction(const std::string& name = "") : name_(name) {}

  Contraction(const std::vector<TensorDim>& dims, const std::vector<TensorIndex>& idxs, const std::string& name = "")
      : name_(name) {
    outShape(dims);
    outAccess(idxs);
  }

  template <typename... Ts>
  Contraction& outShape(Ts... idxs);
  Contraction& outShape(const std::vector<TensorDim>& dims);
  Contraction& outShape(const std::vector<int64_t>& dims);

  template <typename... Ts>
  Contraction& outAccess(Ts... idxs);
  Contraction& outAccess(const std::vector<TensorIndex>& idxs);

  ///
  /// Performs an assignment contraction.
  ///
  Contraction& assign(const IndexedTensor& tensor);

  ///
  /// Performs an maximize contraction.
  ///
  Contraction& max(const IndexedTensor& tensor);

  ///
  /// Performs an minimize contraction.
  ///
  Contraction& min(const IndexedTensor& tensor);

  ///
  /// Performs a product contraction.
  ///
  Contraction& product(const IndexedTensor& tensor);

  ///
  /// Performs a summation contraction.
  ///
  Contraction& sum(const IndexedTensor& tensor);

  ///
  /// Set the initializer for a contraction.
  ///
  Contraction& init(const Tensor& rhs);

  ///
  /// Add a constraint to a contraction.
  ///
  Contraction& add_constraint(const Constraint& constraint);

  ///
  /// Add constraints to a contraction.
  ///
  Contraction& add_constraints(const std::vector<Constraint>& constraints);

  ///
  /// Construct a contraction.
  ///

  Tensor build(edsl_source_location loc = edsl_source_location::current());

  operator Tensor() { return build(); }

 private:
  std::string name_;
  std::vector<TensorDim> outDims_;
  std::vector<TensorIndex> outIdxs_;
  std::vector<Constraint> constraints_;
  IndexedTensor rhs_;
  plaidml_agg_op agg_op_;
  Tensor init_;
  TensorLens lens_;
};

template <typename... Ts>
inline Contraction& Contraction::outShape(Ts... dims) {
  details::into_vector(&outDims_, std::forward<Ts>(dims)...);
  return *this;
}

inline Contraction& Contraction::outShape(const std::vector<TensorDim>& dims) {
  outDims_ = dims;
  return *this;
}

inline Contraction& Contraction::outShape(const std::vector<int64_t>& dims) {
  for (int64_t dim : dims) {
    outDims_.emplace_back(dim);
  }
  return *this;
}

template <typename... Ts>
inline Contraction& Contraction::outAccess(Ts... idxs) {
  details::into_vector(&outIdxs_, std::forward<Ts>(idxs)...);
  return *this;
}

inline Contraction& Contraction::outAccess(const std::vector<TensorIndex>& idxs) {
  outIdxs_ = idxs;
  return *this;
}

inline Contraction& Contraction::init(const Tensor& rhs) {
  init_ = rhs;
  return *this;
}

inline Contraction& Contraction::assign(const IndexedTensor& rhs) {
  agg_op_ = PLAIDML_AGG_OP_ASSIGN;
  rhs_ = rhs;
  return *this;
}

inline Contraction& Contraction::max(const IndexedTensor& rhs) {
  agg_op_ = PLAIDML_AGG_OP_MAX;
  rhs_ = rhs;
  return *this;
}

inline Contraction& Contraction::min(const IndexedTensor& rhs) {
  agg_op_ = PLAIDML_AGG_OP_MIN;
  rhs_ = rhs;
  return *this;
}

inline Contraction& Contraction::product(const IndexedTensor& rhs) {
  agg_op_ = PLAIDML_AGG_OP_PROD;
  rhs_ = rhs;
  return *this;
}

inline Contraction& Contraction::sum(const IndexedTensor& rhs) {
  agg_op_ = PLAIDML_AGG_OP_SUM;
  rhs_ = rhs;
  return *this;
}

inline Contraction& Contraction::add_constraint(const Constraint& constraint) {
  constraints_.push_back(constraint);
  return *this;
}

inline Contraction& Contraction::add_constraints(const std::vector<Constraint>& constraints) {
  for (const auto& constraint : constraints) {
    add_constraint(constraint);
  }
  return *this;
}

inline Tensor Contraction::build(edsl_source_location loc) {
  size_t rank = outDims_.size();
  if (rank != outIdxs_.size()) {
    throw ffi_exception("Rank mismatch between outShape and outAccess", loc);
  }
  std::vector<plaidml_poly_expr*> idxs(rank);
  std::vector<plaidml_dim_expr*> dims(rank);
  for (size_t i = 0; i < rank; i++) {
    idxs[i] = outIdxs_[i].as_ptr();
    dims[i] = outDims_[i].as_ptr();
  }

  idxs = lens_.apply(idxs, loc);
  dims = lens_.apply(dims, loc);

  auto* ptr = ffi::call<plaidml_expr*>(  //
      loc,                               //
      plaidml_expr_contraction,          //
      agg_op_,                           //
      rhs_.op_,                          //
      rank,                              //
      idxs.data(),                       //
      dims.data(),                       //
      init_.as_ptr(),                    //
      name_.c_str());

  std::vector<IndexedTensor> operands;
  if (rhs_.op_ == PLAIDML_COMBO_OP_NONE) {
    operands.push_back(rhs_);
  } else {
    operands = rhs_.operands_;
  }

  for (const IndexedTensor& operand : operands) {
    size_t rank = operand.idxs_.size();
    std::vector<plaidml_poly_expr*> idxs(rank);
    for (size_t i = 0; i < rank; i++) {
      idxs[i] = operand.idxs_[i].as_ptr();
    }
    ffi::call_void(                       //
        loc,                              //
        plaidml_contraction_add_operand,  //
        ptr,                              //
        operand.ref_.as_ptr(),            //
        idxs.size(),                      //
        idxs.data());
  }

  for (const Constraint& constraint : constraints_) {
    ffi::call_void(                          //
        loc,                                 //
        plaidml_contraction_add_constraint,  //
        ptr,                                 //
        constraint.lhs.as_ptr(),             //
        constraint.rhs.as_ptr());
  }

  ffi::call_void(loc, plaidml_contraction_build, ptr);
  return Tensor(ptr);
}

inline TensorLens::TensorLens(const std::string& source, const std::string& target, edsl_source_location loc)
    : map(source.size()) {
  if (source.size() != target.size()) {
    std::stringstream ss;
    ss << "source and target rank mismatch: " << source << " != " << target;
    throw ffi_exception(ss.str(), loc);
  }
  for (unsigned i = 0; i < source.size(); i++) {
    auto pos = target.find(source[i]);
    if (pos == std::string::npos) {
      std::stringstream ss;
      ss << "source and target dims mismatch: " << source << " != " << target;
      throw ffi_exception(ss.str(), loc);
    }
    map[i] = pos;
  }
}

template <typename T>
inline std::vector<T> TensorLens::apply(const std::vector<T>& dims, edsl_source_location loc) const {
  if (map.empty()) {
    return dims;
  }
  if (dims.size() != map.size()) {
    throw ffi_exception("rank mismatch in TensorLens apply", loc);
  }
  std::vector<T> ret(dims.size());
  for (unsigned i = 0; i < dims.size(); i++) {
    ret[i] = dims[map[i]];
  }
  return ret;
}

template <typename... Ts>
inline IndexedTensor Tensor::operator()(Ts... idxs) const {
  std::vector<TensorIndex> vec;
  details::into_vector(&vec, std::forward<Ts>(idxs)...);
  return IndexedTensor(*this, lens_.apply(vec));
}

inline IndexedTensor Tensor::operator()(const std::vector<TensorIndex>& idxs, edsl_source_location loc) const {
  return IndexedTensor(*this, lens_.apply(idxs, loc));
}

inline Tensor Tensor::element(size_t ordinal, edsl_source_location loc) const {
  return Tensor(ffi::call<plaidml_expr*>(loc, plaidml_expr_element, as_ptr(), ordinal));
}

inline Tensor Constant(       //
    const Buffer& buffer,     //
    const std::string& name,  //
    edsl_source_location loc = edsl_source_location::current()) {
  auto* ptr = ffi::call<plaidml_expr*>(  //
      loc,                               //
      plaidml_expr_constant,             //
      buffer.as_ptr(),                   //
      name.c_str());
  return Tensor(ptr);
}

inline Tensor Constant(int value, edsl_source_location loc = edsl_source_location::current()) {
  return Tensor(value, loc);
}

inline Tensor Constant(int64_t value, edsl_source_location loc = edsl_source_location::current()) {
  return Tensor(value, loc);
}

inline Tensor Constant(uint64_t value, edsl_source_location loc = edsl_source_location::current()) {
  return Tensor(value, loc);
}

inline Tensor Constant(double value, edsl_source_location loc = edsl_source_location::current()) {
  return Tensor(value, loc);
}

inline Tensor Placeholder(         //
    const TensorShape& shape,      //
    const std::string& name = "",  //
    edsl_source_location loc = edsl_source_location::current()) {
  auto* ptr = ffi::call<plaidml_expr*>(  //
      loc,                               //
      plaidml_expr_input,                //
      shape.as_ptr(),                    //
      name.c_str());
  return Tensor(ptr);
}

inline Tensor Placeholder(             //
    DType dtype,                       //
    const std::vector<int64_t>& dims,  //
    const std::string& name = "",      //
    edsl_source_location loc = edsl_source_location::current()) {
  TensorShape shape(dtype, dims, loc);
  return Placeholder(shape, name, loc);
}

inline Tensor Zero(edsl_source_location loc = edsl_source_location::current()) { return Tensor(0, loc); }

Tensor intrinsicCall(const std::string& fn, const TensorVec& args);

Tensor intrinsicCall(edsl_source_location loc, const std::string& fn, const TensorVec& args);

template <typename... Ts>
Tensor intrinsic(const std::string& fn, Ts... args) {
  TensorVec vec;
  details::into_vector(&vec, std::forward<Ts>(args)...);
  return intrinsicCall(fn, vec);
}

template <typename... Ts>
Tensor intrinsic(edsl_source_location loc, const std::string& fn, Ts... args) {
  TensorVec vec;
  details::into_vector(&vec, std::forward<Ts>(args)...);
  return intrinsicCall(loc, fn, vec);
}

///
/// \defgroup edsl_intrinsics EDSL Intrinsics
///

/// \addtogroup edsl_intrinsics
/// @{

///
/// Computes the elementwise absolute value of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor abs(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "abs", x);
}

///
/// Computes the elementwise arccosine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor acos(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "acos", x);
}

enum class SortDirection {
  ASC,
  DESC,
};

///
/// Returns the indices of `x` which give its sorted order along `axis` in
/// the specified `direction`, where -1 represents the last axis.
/// \param x Tensor
/// \param axis int
/// \param direction SortDirection
/// \return Tensor
///
inline Tensor argsort(const Tensor& x, int axis = -1, SortDirection direction = SortDirection::ASC) {
  return intrinsic("argsort", x, static_cast<int64_t>(axis), static_cast<int64_t>(direction));
}

///
/// Computes the elementwise inverse hyperbolic cosine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor acosh(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "acosh", x);
}

///
/// Computes the elementwise arcsine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor asin(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "asin", x);
}

///
/// Computes the elementwise inverse hyperbolic sine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor asinh(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "asinh", x);
}

///
/// Computes the elementwise arctangent of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor atan(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "atan", x);
}

///
/// Computes the elementwise inverse hyperbolic tangent of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor atanh(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "atanh", x);
}

///
/// Casts the element type of a tensor `x` to the type specified by `dtype`.
/// \param x Tensor
/// \param dtype DType
/// \return Tensor
///
inline Tensor cast(const Tensor& x, DType dtype, edsl_source_location loc = edsl_source_location::current()) {
  return Tensor{ffi::call<plaidml_expr*>(loc, plaidml_expr_cast, x.as_ptr(), static_cast<plaidml_datatype>(dtype))};
}

///
/// Computes the elementwise ceiling of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor ceil(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "ceil", x);
}

///
/// Computes the elementwise cosine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor cos(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "cos", x);
}

///
/// Computes the elementwise hyperbolic cosine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor cosh(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "cosh", x);
}

///
/// Computes the elementwise Gauss error function of `x`
/// \param x Tensor
/// \return Tensor
///
inline Tensor erf(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "erf", x);
}

///
/// Computes the elementwise natural exponential function of `x`: _e_<sup>x</sup>.
/// \param x Tensor
/// \return Tensor
///
inline Tensor exp(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "exp", x);
}

///
/// Computes the elementwise floor of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor floor(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "floor", x);
}

enum class InterpolationMode : uint64_t {
  NEAREST,
  LINEAR,
  CUBIC,
};

enum class NearestMode : uint64_t {
  ROUND_PREFER_FLOOR,
  ROUND_PREFER_CEIL,
  FLOOR,
  CEIL,
  SIMPLE,
};

///
/// Gather takes an input tensor (`x`) and a set of indices to gather over (`y`), and computes an output tensor that
/// gathers the input tensor from the indices specified.
///
class gather {
 public:
  explicit gather(const Tensor& x, const Tensor& y) : x_(x), y_(y) {}

  ///
  /// Set the axis for gather.
  ///
  gather& axis(int64_t axis) {
    if (axis < 0) {
      axis += x_.rank();
    }
    axis_ = Tensor(axis);
    return *this;
  }

  ///
  /// Set the interpolation mode for gather.
  ///
  gather& interpolationMode(InterpolationMode mode) {
    interpolation_mode_ = Tensor(static_cast<uint64_t>(mode));
    return *this;
  }

  ///
  /// Set the nearest mode for gather.
  ///
  gather& nearestMode(NearestMode mode) {
    nearest_mode_ = Tensor(static_cast<uint64_t>(mode));
    return *this;
  }

  ///
  /// Set the coefficient that controls cubic interpolation for gather.
  ///
  gather& cubeCoeff(float cube_coeff) {
    cube_coeff_ = Tensor(cube_coeff);
    return *this;
  }

  ///
  /// Construct gather.
  ///
  Tensor build(edsl_source_location loc = edsl_source_location::current()) const {
    std::vector<Tensor> args = {x_, y_, axis_, interpolation_mode_, nearest_mode_, cube_coeff_};
    return intrinsicCall(loc, "gather", args);
  }

  operator Tensor() { return build(); }

 private:
  Tensor x_;
  Tensor y_;

  ///
  /// axis_ is a dimension index to gather data from
  /// interpolation_mode_ specifies type of interpolation
  /// nearest_mode_ specifies type of  nearest interpolation
  /// cube_coeff_ controls the cubic interpolation
  ///
  Tensor axis_ = Tensor(0);
  Tensor interpolation_mode_ = Tensor(static_cast<uint64_t>(InterpolationMode::LINEAR));
  Tensor nearest_mode_ = Tensor(static_cast<uint64_t>(NearestMode::ROUND_PREFER_FLOOR));
  Tensor cube_coeff_ = Tensor(-0.75);
};

///
/// Returns the identity of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor ident(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "ident", x);
}

///
/// Returns a tensor populated with the index value of the shape and axis specified.
/// \param dims std::vector<TensorDim>
/// \param axis size_t
/// \return Tensor
///
inline Tensor index(const std::vector<TensorDim>& dims, size_t axis,
                    edsl_source_location loc = edsl_source_location::current()) {
  TensorVec args = {Tensor{static_cast<int64_t>(axis)}};
  for (const auto& dim : dims) {
    args.emplace_back(dim);
  }
  return intrinsicCall(loc, "index", args);
}

///
/// Computes the elementwise natural logarithm of `x`: _ln_(`x`).
/// \param x Tensor
/// \return Tensor
///
inline Tensor log(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "log", x);
}

///
/// Computes the elementwise `y`th power of `x`.
/// \param x Tensor
/// \param y Tensor
/// \return Tensor
///
inline Tensor pow(const Tensor& x, const Tensor& y, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "pow", x, y);
}

///
/// Generates a Tensor of elementwise pseudorandom numbers using the seed values specified in `state`.
/// \param state Tensor
/// \param dims vector<int64_t>
/// \return Tensor
///
inline std::pair<Tensor, Tensor> prng(const Tensor& state, const std::vector<int64_t>& dims,
                                      edsl_source_location loc = edsl_source_location::current()) {
  TensorVec args = {state};
  for (int64_t dim : dims) {
    args.emplace_back(TensorDim(dim));
  }
  Tensor R = intrinsicCall(loc, "prng", args);
  return std::make_pair(R.element(0), R.element(1));
}

///
/// Takes an input tensor `x` and reshapes it according to `dims`.
/// \param x Tensor
/// \param dims vector<int64_t>
/// \return Tensor
///
inline Tensor reshape(const Tensor& x, const std::vector<int64_t>& dims,
                      edsl_source_location loc = edsl_source_location::current()) {
  TensorVec args = {x};
  for (int64_t dim : dims) {
    args.emplace_back(dim);
  }
  return intrinsicCall(loc, "reshape", args);
}

///
/// Takes an input tensor `x` and reshapes it according to `dims`.
/// \param x Tensor
/// \param dims vector<TensorDim>
/// \return Tensor
///
inline Tensor reshape(const Tensor& x, const std::vector<TensorDim>& dims,
                      edsl_source_location loc = edsl_source_location::current()) {
  TensorVec args = {x};
  for (const TensorDim& dim : dims) {
    args.emplace_back(dim);
  }
  return intrinsicCall(loc, "reshape", args);
}

///
/// Rounds `x` elementwise.
/// \param x Tensor
/// \return Tensor
///
inline Tensor round(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "round", x);
}

///
/// Takes an input tensor (`x`), a set of indices to scatter over (`y`), and the number of elements in the scattered
/// tensor (`z`), and returns an output tensor that scatters the input tensor across the number of elements specified.
/// \param x Tensor
/// \param y Tensor
/// \param z Tensor
/// \return Tensor
///

enum class ScatterMode : uint64_t { NORMAL, UPDATE_SLICE, UPDATE_ELT, UPDATE_ND };

class scatter {
 public:
  explicit scatter(const Tensor& x, const Tensor& y, const Tensor& z) : x_(x), y_(y), z_(z) {}

  scatter& axis(int64_t axis) {
    if (axis < 0) {
      axis += x_.rank();
    }
    axis_ = Tensor(axis);
    return *this;
  }

  scatter& mode(ScatterMode mode) {
    mode_ = Tensor(static_cast<uint64_t>(mode));
    return *this;
  }

  Tensor build(edsl_source_location loc = edsl_source_location::current()) const {
    std::vector<Tensor> args = {x_, y_, z_, axis_, mode_};
    return intrinsicCall(loc, "scatter", args);
  }

  operator Tensor() { return build(); }

 private:
  Tensor x_;
  Tensor y_;
  Tensor z_;
  Tensor axis_ = Tensor(0);
  Tensor mode_ = Tensor(static_cast<uint64_t>(ScatterMode::NORMAL));
};

///
/// Performs an elementwise conditional which returns the corresponding
/// element in `true_case` if the condition is evaluated to be true or the
/// corresponding element in `false_case` if the condition is evaluated to be
/// false.
/// \param cond Tensor
/// \param true_case Tensor
/// \param false_case Tensor
/// \return Tensor
///
inline Tensor select(const Tensor& cond, const Tensor& true_case, const Tensor& false_case,
                     edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "select", cond, true_case, false_case);
}

///
/// Returns the shape of `x` as a Tensor.
/// \param x Tensor
/// \return Tensor
///
inline Tensor shape(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "shape", x);
}

///
/// Computes the elementwise sine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor sin(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "sin", x);
}

///
/// Computes the elementwise hyperbolic sine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor sinh(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "sinh", x);
}

///
/// Computes the elementwise square root of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor sqrt(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "sqrt", x);
}

///
/// Computes the elementwise tangent of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor tan(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "tan", x);
}

///
/// Computes the elementwise hyperbolic tangent of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor tanh(const Tensor& x, edsl_source_location loc = edsl_source_location::current()) {
  return intrinsic(loc, "tanh", x);
}

/// @}

inline TensorDim TensorDim::operator-() const { return TensorDim(PLAIDML_INT_OP_NEG, {*this}); }

inline TensorIndex TensorIndex::operator-() const { return TensorIndex(PLAIDML_INT_OP_NEG, {*this}); }

inline Constraint TensorIndex::operator<(int64_t rhs) const { return Constraint{*this, TensorDim(rhs)}; }

inline Constraint TensorIndex::operator<(const TensorDim& rhs) const { return Constraint{*this, rhs}; }

#define PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(_op_, _int_op_, _fn_)           \
  inline TensorIndex operator _op_(const TensorIndex& lhs, const TensorIndex& rhs) { \
    return TensorIndex(_int_op_, {lhs, rhs});                                        \
  }                                                                                  \
  inline TensorIndex operator _op_(const TensorIndex& lhs, int64_t rhs) {            \
    return TensorIndex(_int_op_, {lhs, TensorIndex{rhs}});                           \
  }                                                                                  \
  inline TensorIndex operator _op_(int64_t lhs, const TensorIndex& rhs) {            \
    return TensorIndex(_int_op_, {TensorIndex{lhs}, rhs});                           \
  }                                                                                  \
  inline TensorIndex operator _op_(const TensorIndex& lhs, const TensorDim& rhs) {   \
    return TensorIndex(_int_op_, lhs, rhs, true);                                    \
  }                                                                                  \
  inline TensorIndex operator _op_(const TensorDim& lhs, const TensorIndex& rhs) {   \
    return TensorIndex(_int_op_, rhs, lhs, false);                                   \
  }                                                                                  \
  inline Tensor operator _op_(const Tensor& lhs, const TensorDim& rhs) { /**/        \
    return intrinsic(_fn_, lhs, Tensor(rhs));                                        \
  }                                                                                  \
  inline Tensor operator _op_(const TensorDim& lhs, const Tensor& rhs) { /**/        \
    return intrinsic(_fn_, Tensor(lhs), rhs);                                        \
  }                                                                                  \
  inline TensorDim operator _op_(const TensorDim& lhs, const TensorDim& rhs) {       \
    return TensorDim(_int_op_, {lhs, rhs});                                          \
  }                                                                                  \
  inline TensorDim operator _op_(int64_t lhs, const TensorDim& rhs) {                \
    return TensorDim(_int_op_, {TensorDim(lhs), rhs});                               \
  }                                                                                  \
  inline TensorDim operator _op_(const TensorDim& lhs, int64_t rhs) {                \
    return TensorDim(_int_op_, {lhs, TensorDim(rhs)});                               \
  }

PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(+, PLAIDML_INT_OP_ADD, "add");
PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(-, PLAIDML_INT_OP_SUB, "sub");
PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(*, PLAIDML_INT_OP_MUL, "mul");
PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(/, PLAIDML_INT_OP_DIV, "div");

#define PLAIDML_EDSL_DEFINE_TENSOR_DIM_BINARY_FN(_fn_, _int_op_)                                                  \
  inline TensorDim _fn_(const TensorDim& lhs, const TensorDim& rhs) { return TensorDim(_int_op_, {lhs, rhs}); }   \
  inline TensorDim _fn_(int64_t lhs, const TensorDim& rhs) { return TensorDim(_int_op_, {TensorDim(lhs), rhs}); } \
  inline TensorDim _fn_(const TensorDim& lhs, int64_t rhs) { return TensorDim(_int_op_, {lhs, TensorDim(rhs)}); }

PLAIDML_EDSL_DEFINE_TENSOR_DIM_BINARY_FN(max, PLAIDML_INT_OP_MAX);
PLAIDML_EDSL_DEFINE_TENSOR_DIM_BINARY_FN(min, PLAIDML_INT_OP_MIN);

inline Tensor Tensor::operator-() const { return intrinsicCall("neg", {*this}); }
inline Tensor Tensor::operator~() const { return intrinsicCall("bit_not", {*this}); }
inline Tensor Tensor::operator!() const { return intrinsicCall("logical_not", {*this}); }

template <typename T>
class has_tensor {
  typedef char one;
  typedef int two;

  template <typename C>
  static one test(decltype(&C::operator Tensor));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

struct LocatedTensor {
  Tensor tensor;
  edsl_source_location loc;

  template <typename T, typename = enable_if_t<has_tensor<T>::value>>
  LocatedTensor(T tensor, edsl_source_location loc = edsl_source_location::current())  // NOLINT
      : tensor(tensor), loc(loc) {}

  LocatedTensor(Tensor tensor, edsl_source_location loc = edsl_source_location::current())  // NOLINT
      : tensor(tensor), loc(loc) {}
};

#ifdef _MSC_VER
#define PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(_op_, _fn_)                                     \
  inline Tensor operator _op_(Tensor lhs, Tensor rhs) { return intrinsic(_fn_, lhs, rhs); }   \
  inline Tensor operator _op_(Tensor lhs, int rhs) { return intrinsic(_fn_, lhs, rhs); }      \
  inline Tensor operator _op_(Tensor lhs, int64_t rhs) { return intrinsic(_fn_, lhs, rhs); }  \
  inline Tensor operator _op_(Tensor lhs, uint64_t rhs) { return intrinsic(_fn_, lhs, rhs); } \
  inline Tensor operator _op_(Tensor lhs, double rhs) { return intrinsic(_fn_, lhs, rhs); }   \
  inline Tensor operator _op_(int lhs, Tensor rhs) { return intrinsic(_fn_, lhs, rhs); }      \
  inline Tensor operator _op_(int64_t lhs, Tensor rhs) { return intrinsic(_fn_, lhs, rhs); }  \
  inline Tensor operator _op_(uint64_t lhs, Tensor rhs) { return intrinsic(_fn_, lhs, rhs); } \
  inline Tensor operator _op_(double lhs, Tensor rhs) { return intrinsic(_fn_, lhs, rhs); }
#else
#define PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(_op_, _fn_)                                                            \
  inline Tensor operator _op_(LocatedTensor lhs, LocatedTensor rhs) {                                                \
    return intrinsic(lhs.loc, _fn_, lhs.tensor, rhs.tensor);                                                         \
  }                                                                                                                  \
  inline Tensor operator _op_(LocatedTensor lhs, int rhs) { return intrinsic(lhs.loc, _fn_, lhs.tensor, rhs); }      \
  inline Tensor operator _op_(LocatedTensor lhs, int64_t rhs) { return intrinsic(lhs.loc, _fn_, lhs.tensor, rhs); }  \
  inline Tensor operator _op_(LocatedTensor lhs, uint64_t rhs) { return intrinsic(lhs.loc, _fn_, lhs.tensor, rhs); } \
  inline Tensor operator _op_(LocatedTensor lhs, double rhs) { return intrinsic(lhs.loc, _fn_, lhs.tensor, rhs); }   \
  inline Tensor operator _op_(int lhs, LocatedTensor rhs) { return intrinsic(rhs.loc, _fn_, lhs, rhs.tensor); }      \
  inline Tensor operator _op_(int64_t lhs, LocatedTensor rhs) { return intrinsic(rhs.loc, _fn_, lhs, rhs.tensor); }  \
  inline Tensor operator _op_(uint64_t lhs, LocatedTensor rhs) { return intrinsic(rhs.loc, _fn_, lhs, rhs.tensor); } \
  inline Tensor operator _op_(double lhs, LocatedTensor rhs) { return intrinsic(rhs.loc, _fn_, lhs, rhs.tensor); }
#endif

PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(+, "add");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(-, "sub");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(*, "mul");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(/, "div");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(%, "mod");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(==, "cmp_eq");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(!=, "cmp_ne");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(<, "cmp_lt");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(>, "cmp_gt");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(<=, "cmp_le");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(>=, "cmp_ge");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(<<, "bit_shl");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(>>, "bit_shr");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(&, "bit_and");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(|, "bit_or");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(^, "bit_xor");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(&&, "logical_and");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(||, "logical_or");

inline Tensor intrinsicCall(const std::string& fn, const TensorVec& args) {
  std::vector<plaidml_expr*> ptrs(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    ptrs[i] = args[i].as_ptr();
  }
  return Tensor{ffi::call<plaidml_expr*>(plaidml_expr_intrinsic, fn.c_str(), ptrs.size(), ptrs.data())};
}

inline Tensor intrinsicCall(edsl_source_location loc, const std::string& fn, const TensorVec& args) {
  std::vector<plaidml_expr*> ptrs(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    ptrs[i] = args[i].as_ptr();
  }
  return Tensor{ffi::call<plaidml_expr*>(loc, plaidml_expr_intrinsic, fn.c_str(), ptrs.size(), ptrs.data())};
}

class Value {
 public:
  Value(edsl_source_location loc = edsl_source_location::current())  // NOLINT
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_none))) {}

  explicit Value(plaidml_value* ptr, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_clone, ptr))) {}

  explicit Value(int8_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_int, value))) {}

  explicit Value(int16_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_int, value))) {}

  explicit Value(int32_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_int, value))) {}

  explicit Value(int64_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_int, value))) {}

  explicit Value(uint8_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_int, value))) {}

  explicit Value(uint16_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_int, value))) {}

  explicit Value(uint32_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_int, value))) {}

  explicit Value(uint64_t value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_int, value))) {}

  explicit Value(double value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_float, value))) {}

  explicit Value(const std::string& value, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_str, value.c_str()))) {}

  explicit Value(const TensorDim& dim, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_dim, dim.as_ptr()))) {}

  explicit Value(const Tensor& tensor, edsl_source_location loc = edsl_source_location::current()) {
    if (auto* ptr = tensor.as_ptr()) {
      ptr_ = details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_expr, ptr));
    } else {
      ptr_ = details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_none));
    }
  }

  explicit Value(const std::vector<Value>& tuple, edsl_source_location loc = edsl_source_location::current()) {
    std::vector<plaidml_value*> args(tuple.size());
    for (size_t i = 0; i < args.size(); i++) {
      args[i] = tuple[i].as_ptr();
    }
    ptr_ = details::make_ptr(ffi::call<plaidml_value*>(loc, plaidml_value_tuple, args.size(), args.data()));
  }

  std::string str(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::str(ffi::call<plaidml_string*>(loc, plaidml_value_repr, as_ptr()));
  }

  bool is_none(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::call<plaidml_value_kind>(loc, plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_NONE;
  }

  bool is_int(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::call<plaidml_value_kind>(loc, plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_INT;
  }

  bool is_float(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::call<plaidml_value_kind>(loc, plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_FLOAT;
  }

  bool is_tensor(edsl_source_location loc = edsl_source_location::current()) const {
    return ffi::call<plaidml_value_kind>(loc, plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_EXPR;
  }

  bool is_tuple(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::call<plaidml_value_kind>(loc, plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_TUPLE;
  }

  bool is_str(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::call<plaidml_value_kind>(loc, plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_STR;
  }

  bool is_dim(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::call<plaidml_value_kind>(loc, plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_DIM;
  }

  bool as_bool(edsl_source_location loc = edsl_source_location::current()) const {
    // bools are integers under the hood, but we can still return a bool type
    return static_cast<bool>(ffi::call<int64_t>(loc, plaidml_value_int_get, as_ptr()));
  }

  int64_t as_int(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::call<int64_t>(loc, plaidml_value_int_get, as_ptr());
  }

  double as_float(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::call<double>(loc, plaidml_value_float_get, as_ptr());
  }

  std::string as_str(edsl_source_location loc = edsl_source_location::current()) const {  //
    return ffi::str(ffi::call<plaidml_string*>(loc, plaidml_value_str_get, as_ptr()));
  }

  Tensor as_tensor(edsl_source_location loc = edsl_source_location::current()) const {
    if (is_tensor()) {
      return Tensor(ffi::call<plaidml_expr*>(loc, plaidml_value_expr_get, as_ptr()));
    }
    if (is_dim(loc)) {
      return Tensor(as_dim(loc));
    }
    if (is_float(loc)) {
      return Tensor(as_float(loc));
    }
    if (is_int(loc)) {
      return Tensor(as_int(loc));
    }
    throw ffi_exception("Value cannot be coerced into Tensor", loc);
  }

  TensorDim as_dim(edsl_source_location loc = edsl_source_location::current()) const {  //
    return TensorDim(details::make_ptr(ffi::call<plaidml_dim_expr*>(loc, plaidml_value_dim_get, as_ptr())));
  }

  std::vector<Value> as_tuple(edsl_source_location loc = edsl_source_location::current()) const {
    auto tuple = details::make_ptr(ffi::call<plaidml_tuple*>(loc, plaidml_value_tuple_get, as_ptr()));
    std::vector<Value> ret(tuple->size);
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = Value{tuple->elts[i]};
    }
    return ret;
  }

  std::vector<int64_t> as_int_tuple(edsl_source_location loc = edsl_source_location::current()) const {
    auto tuple = details::make_ptr(ffi::call<plaidml_tuple*>(loc, plaidml_value_tuple_get, as_ptr()));
    std::vector<int64_t> ret(tuple->size);
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = Value{tuple->elts[i]}.as_int();
    }
    return ret;
  }

  std::vector<int64_t> as_int_tuple_or_empty(edsl_source_location loc = edsl_source_location::current()) const {  //
    return is_none(loc) ? std::vector<int64_t>{} : as_int_tuple(loc);
  }

  plaidml_value* as_ptr() const { return ptr_.get(); }

 private:
  std::shared_ptr<plaidml_value> ptr_;
};

template <typename... Ts>
Value make_tuple(Ts... elts) {
  std::vector<Value> vec;
  details::into_vector(&vec, std::forward<Ts>(elts)...);
  return Value{vec};
}

template <typename T>
Value make_tuple(const std::vector<T>& elts, edsl_source_location loc = edsl_source_location::current()) {
  std::vector<Value> vec(elts.size());
  for (size_t i = 0; i < vec.size(); i++) {
    vec[i] = Value{elts[i], loc};
  }
  return Value{vec, loc};
}

inline Value make_tuple(const std::vector<Value>& elts,
                        edsl_source_location loc = edsl_source_location::current()) {  //
  return Value{elts, loc};
}

inline Value None(edsl_source_location loc = edsl_source_location::current()) { return Value(loc); }

inline Program buildProgram(const std::string& name, const TensorVec& inputs, const TensorVec& outputs,
                            edsl_source_location loc = edsl_source_location::current()) {
  std::vector<plaidml_expr*> input_exprs(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    input_exprs[i] = inputs[i].as_ptr();
  }
  std::vector<plaidml_expr*> output_exprs(outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    output_exprs[i] = outputs[i].as_ptr();
  }
  auto* ptr = ffi::call<plaidml_program*>(  //
      loc,                                  //
      plaidml_build,                        //
      name.c_str(),                         //
      inputs.size(),                        //
      input_exprs.data(),                   //
      /*shapes=*/nullptr,                   //
      outputs.size(),                       //
      output_exprs.data());
  return Program(ptr);
}

inline std::ostream& operator<<(std::ostream& os, const Tensor& x) {
  os << x.str();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorDim& x) {
  os << x.str();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorIndex& x) {
  os << x.str();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Value& x) {
  os << x.str();
  return os;
}

using Dictionary = std::unordered_map<std::string, Value>;

inline Tensor pragma(const Tensor& tensor, const std::string& op, const Dictionary& attrs,
                     edsl_source_location loc = edsl_source_location::current()) {
  std::vector<plaidml_attr> elts;
  std::vector<plaidml_attr*> ptrs;
  elts.reserve(attrs.size());
  ptrs.reserve(attrs.size());
  for (const auto& kvp : attrs) {
    plaidml_attr attr{kvp.first.c_str(), kvp.second.as_ptr()};
    elts.push_back(attr);
    ptrs.push_back(&elts.back());
  }
  return Tensor{
      ffi::call<plaidml_expr*>(loc, plaidml_expr_pragma, tensor.as_ptr(), op.c_str(), elts.size(), ptrs.data())};
}

///
/// Adds a tracepoint to the graph
///
inline Tensor trace(const Tensor& x, const std::string& msg,
                    edsl_source_location loc = edsl_source_location::current()) {
  return pragma(x, "trace", {{"msg", Value(msg)}}, loc);
}

using LayerBodySingleFn = std::function<Tensor()>;
using LayerBodyMultiFn = std::function<TensorVec()>;

class LayerBuilder {
 public:
  LayerBuilder(const std::string& op, const TensorVec& operands, const Dictionary& attrs,
               edsl_source_location loc = edsl_source_location::current()) {
    std::vector<plaidml_attr> elts;
    std::vector<plaidml_attr*> ptrs;
    elts.reserve(attrs.size());
    ptrs.reserve(attrs.size());
    for (const auto& kvp : attrs) {
      plaidml_attr attr{kvp.first.c_str(), kvp.second.as_ptr()};
      elts.push_back(attr);
      ptrs.push_back(&elts.back());
    }

    std::vector<plaidml_expr*> rawOperands;
    rawOperands.reserve(operands.size());
    for (Tensor operand : operands) {
      rawOperands.push_back(operand.as_ptr());
    }

    expr = details::make_ptr(          //
        ffi::call<plaidml_expr*>(      //
            loc,                       //
            plaidml_expr_layer_begin,  //
            op.c_str(),                //
            rawOperands.size(),        //
            rawOperands.data(),        //
            ptrs.size(),               //
            ptrs.data()));
  }

  TensorVec build(const LayerBodyMultiFn& fn, edsl_source_location loc = edsl_source_location::current()) {
    TensorVec innerResults = fn();

    std::vector<plaidml_expr*> rawResults;
    rawResults.reserve(innerResults.size());
    for (Tensor result : innerResults) {
      rawResults.push_back(result.as_ptr());
    }

    outerExprs = details::make_ptr(  //
        ffi::call<plaidml_exprs*>(   //
            loc,                     //
            plaidml_expr_layer_end,  //
            expr.get(),              //
            rawResults.size(),       //
            rawResults.data()));

    TensorVec outerResults;
    outerResults.reserve(outerExprs->size);
    for (size_t i = 0; i < outerExprs->size; i++) {
      plaidml_expr* expr = outerExprs->elts[i];
      outerResults.push_back(Tensor{expr});
    }
    return outerResults;
  }

  ~LayerBuilder() {
    // We need to ensure that the plaidml_expr_layer_end is called even if an exception occurs.
    if (expr && !outerExprs) {
      details::make_ptr(               //
          ffi::call<plaidml_exprs*>(   //
              plaidml_expr_layer_end,  //
              expr.get(),              //
              0,                       //
              nullptr));
    }
  }

 private:
  std::shared_ptr<plaidml_expr> expr;
  std::shared_ptr<plaidml_exprs> outerExprs;
};

inline TensorVec layer(const std::string& op, const TensorVec& operands, const Dictionary& attrs,
                       const LayerBodyMultiFn& fn, edsl_source_location loc = edsl_source_location::current()) {
  return LayerBuilder(op, operands, attrs, loc).build(fn, loc);
}

inline Tensor layer(const std::string& op, const TensorVec& operands, const Dictionary& attrs,
                    const LayerBodySingleFn& fn, edsl_source_location loc = edsl_source_location::current()) {
  return layer(
      op, operands, attrs, [&]() { return TensorVec{fn()}; }, loc)[0];
}

inline Tensor layer(const std::string& op, const TensorVec& operands, const LayerBodySingleFn& fn,
                    edsl_source_location loc = edsl_source_location::current()) {
  return layer(op, operands, {}, fn, loc);
}

///
/// wrapper edsl graph into SCF loop.
/// \param TensorVec
/// \param loopCycle int64_t
/// \return Tensor
///

using loopSingefunc = std::function<Tensor()>;
using loopMultifunc = std::function<TensorVec()>;

inline Tensor loop(size_t lb, size_t hb, size_t step, const TensorVec& operands, const TensorVec& results) {
  if (operands.size() != results.size()) {
    throw ffi_exception("iter args don't equal to init args of scf", edsl_source_location::current());
  }
  TensorVec args = {Tensor{lb}, Tensor{static_cast<int64_t>(hb)}, Tensor{step}};
  args.insert(args.end(), operands.begin(), operands.end());
  args.insert(args.end(), results.begin(), results.end());
  return intrinsicCall("loop", args);
}

// inline Tensor loop(int64_t loopCycle, const TensorVec& operands, const loopMultifunc& fn){
//  return loop(loopCycle, fn());
//}

inline Tensor loop(size_t lb, size_t hb, size_t step, const TensorVec& operands, const loopSingefunc& fn) {
  return loop(lb, hb, step, operands, {fn()});
}

}  // namespace edsl
}  // namespace plaidml
