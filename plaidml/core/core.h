// Copyright 2019 Intel Corporation.

#pragma once

#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "plaidml/core/ffi.h"

#if __has_include(<source_location>)
#include <source_location>
using edsl_source_location = std::source_location;
#elif __cplusplus >= 201402 && __has_include(<experimental/source_location>)
#include <experimental/source_location>
using edsl_source_location = std::experimental::source_location;
#else
struct edsl_source_location {
  static edsl_source_location current() { return edsl_source_location{}; }
  constexpr uint_least32_t line() const noexcept { return 0; }
  constexpr uint_least32_t column() const noexcept { return 0; }
  constexpr const char* file_name() const noexcept { return "??"; }
  constexpr const char* function_name() const noexcept { return "??"; }
};
#endif

class ffi_exception : public std::exception {
 private:
  std::string msg_;

 public:
  explicit ffi_exception(const std::string& message, edsl_source_location loc) {
    std::stringstream ss;
    if (std::string(loc.file_name()).find("edsl/edsl.h") == std::string::npos) {
      ss << "Exception at " << loc.file_name() << ":" << std::to_string(loc.line());
    } else {
      ss << "Exception at ??:0";
    }
    ss << " with message: " << message;
    msg_ = ss.str();
  }
  virtual const char* what() const noexcept { return msg_.c_str(); }
};

namespace plaidml {

namespace ffi {

inline std::string str(plaidml_string* ptr) {
  std::string ret{plaidml_string_ptr(ptr)};
  plaidml_string_free(ptr);
  return ret;
}

template <typename T, typename F, typename... Args>
T call(edsl_source_location loc, F fn, Args... args) {
  plaidml_error err;
  auto ret = fn(&err, args...);
  if (err.code) {
    throw ffi_exception(str(err.msg), loc);
  }
  return ret;
}

template <typename T, typename F, typename... Args>
T call(F fn, Args... args) {
  plaidml_error err;
  auto ret = fn(&err, args...);
  if (err.code) {
    throw std::runtime_error(str(err.msg));
  }
  return ret;
}

template <typename F, typename... Args>
void call_void(edsl_source_location loc, F fn, Args... args) {
  plaidml_error err;
  fn(&err, args...);
  if (err.code) {
    throw ffi_exception(str(err.msg), loc);
  }
}

template <typename F, typename... Args>
void call_void(F fn, Args... args) {
  plaidml_error err;
  fn(&err, args...);
  if (err.code) {
    throw std::runtime_error(str(err.msg));
  }
}

}  // namespace ffi

namespace details {

struct Deleter {
  void operator()(plaidml_buffer* ptr) { ffi::call_void(plaidml_buffer_free, ptr); }
  void operator()(plaidml_integers* ptr) { ffi::call_void(plaidml_integers_free, ptr); }
  void operator()(plaidml_program* ptr) { ffi::call_void(plaidml_program_free, ptr); }
  void operator()(plaidml_shape* ptr) { ffi::call_void(plaidml_shape_free, ptr); }
  void operator()(plaidml_shapes* ptr) { ffi::call_void(plaidml_shapes_free, ptr); }
};

template <typename T>
inline std::shared_ptr<T> make_ptr(T* ptr) {
  return std::shared_ptr<T>(ptr, Deleter{});
}

}  // namespace details

class Program;

///
/// Initializes the PlaidML Core API.
///
inline void init(edsl_source_location loc = edsl_source_location::current()) {  //
  ffi::call_void(loc, plaidml_init);
}

///
/// \defgroup core_objects Objects
///

///
/// \ingroup core_objects
/// \enum DType
/// Defines the set of supported element types in a Tensor.
///
enum class DType {
  INVALID = PLAIDML_DATA_INVALID,
  INTX = PLAIDML_DATA_INTX,
  UINTX = PLAIDML_DATA_UINTX,
  FLOATX = PLAIDML_DATA_FLOATX,
  BOOLEAN = PLAIDML_DATA_BOOLEAN,
  INT8 = PLAIDML_DATA_INT8,
  UINT8 = PLAIDML_DATA_UINT8,
  INT16 = PLAIDML_DATA_INT16,
  UINT16 = PLAIDML_DATA_UINT16,
  INT32 = PLAIDML_DATA_INT32,
  UINT32 = PLAIDML_DATA_UINT32,
  INT64 = PLAIDML_DATA_INT64,
  UINT64 = PLAIDML_DATA_UINT64,
  FLOAT16 = PLAIDML_DATA_FLOAT16,
  FLOAT32 = PLAIDML_DATA_FLOAT32,
  FLOAT64 = PLAIDML_DATA_FLOAT64,
  _LAST,
};

inline const char* to_string(DType dtype) {
  switch (dtype) {
    case DType::BOOLEAN:
      return "bool";
    case DType::INT8:
      return "int8";
    case DType::UINT8:
      return "uint8";
    case DType::INT16:
      return "int16";
    case DType::UINT16:
      return "uint16";
    case DType::INT32:
      return "int32";
    case DType::UINT32:
      return "uint32";
    case DType::INT64:
      return "int64";
    case DType::UINT64:
      return "uint64";
    case DType::FLOAT16:
      return "float16";
    case DType::FLOAT32:
      return "float32";
    case DType::FLOAT64:
      return "float64";
    default:
      return "<invalid>";
  }
}

///
/// \ingroup core_objects
/// \class TensorShape
/// Represents the shape of a Tensor.
///
class TensorShape {
 public:
  ///
  /// The default constructor for a TensorShape.
  ///
  TensorShape() {}

  ///
  /// TensorShape constructor
  /// \param dtype DType
  ///
  TensorShape(DType dtype, const std::vector<int64_t>& sizes,
              edsl_source_location loc = edsl_source_location::current()) {
    ptr_ = details::make_ptr(ffi::call<plaidml_shape*>(  //
        loc,                                             //
        plaidml_shape_alloc,                             //
        static_cast<plaidml_datatype>(dtype),            //
        sizes.size(),                                    //
        sizes.data(),                                    //
        /*strides=*/nullptr));
  }

  ///
  /// Constructor for the TensorShape type.
  /// \param dtype DType
  /// \param sizes const vector<int64_t>
  /// \param strides const vector<int64_t>
  ///
  TensorShape(DType dtype,                        //
              const std::vector<int64_t>& sizes,  //
              const std::vector<int64_t>& strides, edsl_source_location loc = edsl_source_location::current()) {
    if (sizes.size() != strides.size()) {
      throw ffi_exception("Sizes and strides must have the same rank.", loc);
    }
    ptr_ = details::make_ptr(ffi::call<plaidml_shape*>(  //
        loc,                                             //
        plaidml_shape_alloc,                             //
        static_cast<plaidml_datatype>(dtype),            //
        sizes.size(),                                    //
        sizes.data(),                                    //
        strides.data()));
  }

  // This is an internal constructor.
  explicit TensorShape(plaidml_shape* ptr) : ptr_(details::make_ptr(ptr)) {}

  // This is an internal constructor.
  explicit TensorShape(const std::shared_ptr<plaidml_shape>& ptr) : ptr_(ptr) {}

  ///
  /// Returns the element DType of this TensorShape.
  ///
  DType dtype(edsl_source_location loc = edsl_source_location::current()) const {
    return static_cast<DType>(ffi::call<plaidml_datatype>(loc, plaidml_shape_get_dtype, as_ptr()));
  }

  ///
  /// Returns the number of dimensions in this TensorShape.
  ///
  size_t rank(edsl_source_location loc = edsl_source_location::current()) const {
    return ffi::call<size_t>(loc, plaidml_shape_get_rank, as_ptr());
  }

  ///
  /// Returns the sizes of this TensorShape.
  ///
  std::vector<int64_t> sizes(edsl_source_location loc = edsl_source_location::current()) const {
    auto dims = details::make_ptr(ffi::call<plaidml_integers*>(loc, plaidml_shape_get_sizes, as_ptr()));
    return std::vector<int64_t>(dims->elts, dims->elts + dims->size);
  }

  ///
  /// Returns the strides of this TensorShape.
  ///
  std::vector<int64_t> strides(edsl_source_location loc = edsl_source_location::current()) const {
    auto dims = details::make_ptr(ffi::call<plaidml_integers*>(loc, plaidml_shape_get_strides, as_ptr()));
    return std::vector<int64_t>(dims->elts, dims->elts + dims->size);
  }

  ///
  /// Returns the amount of memory that a Tensor with this TensorShape would
  /// occupy in bytes.
  ///
  uint64_t byte_size(edsl_source_location loc = edsl_source_location::current()) const {
    return ffi::call<uint64_t>(loc, plaidml_shape_get_nbytes, as_ptr());
  }

  ///
  /// Returns a string representation of this TensorShape.
  ///
  std::string str(edsl_source_location loc = edsl_source_location::current()) const {
    return ffi::str(ffi::call<plaidml_string*>(loc, plaidml_shape_repr, as_ptr()));
  }

  ///
  /// Equality comparison for TensorShape types.
  ///
  bool operator==(const TensorShape& rhs) const { return str() == rhs.str(); }

  // This is an internal method.
  plaidml_shape* as_ptr() const { return ptr_.get(); }

 private:
  std::shared_ptr<plaidml_shape> ptr_;
};

///
/// \ingroup core_objects
/// \class Buffer
///
class Buffer {
 public:
  ///
  /// Buffer constructor
  ///
  Buffer() = default;

  ///
  /// Buffer constructor
  /// \param shape TensorShape
  ///
  Buffer(char* data, size_t size, const TensorShape& shape, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_buffer*>(loc, plaidml_buffer_adopt, shape.as_ptr(), data, size))) {}

  template <typename T>
  Buffer(const std::vector<T>& vec, const TensorShape& shape,
         edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_buffer*>(        //
            loc,                                                  //
            plaidml_buffer_adopt,                                 //
            shape.as_ptr(),                                       //
            reinterpret_cast<char*>(const_cast<T*>(vec.data())),  //
            vec.size() * sizeof(T)))) {}

  ///
  /// Buffer constructor
  /// \param shape TensorShape
  ///
  explicit Buffer(const TensorShape& shape, edsl_source_location loc = edsl_source_location::current())
      : ptr_(details::make_ptr(ffi::call<plaidml_buffer*>(loc, plaidml_buffer_alloc, shape.as_ptr()))) {}

  ///
  /// Buffer constructor
  /// \param ptr plaidml_buffer*
  /// \param shape TensorShape
  explicit Buffer(plaidml_buffer* ptr) : ptr_(details::make_ptr(ptr)) {}

  ///
  /// data
  ///
  char* data(edsl_source_location loc = edsl_source_location::current()) {
    return ffi::call<char*>(loc, plaidml_buffer_data, as_ptr());
  }

  ///
  /// size
  ///
  size_t size(edsl_source_location loc = edsl_source_location::current()) {
    return ffi::call<size_t>(loc, plaidml_buffer_size, as_ptr());
  }

  ///
  /// shape
  ///

  TensorShape shape(edsl_source_location loc = edsl_source_location::current()) {
    return TensorShape{ffi::call<plaidml_shape*>(loc, plaidml_buffer_shape, as_ptr())};
  }

  plaidml_buffer* as_ptr() const { return ptr_.get(); }

  void copy_into(void* dst) { memcpy(dst, data(), size()); }

  void copy_from(const void* src) { memcpy(data(), src, size()); }

 private:
  std::shared_ptr<plaidml_buffer> ptr_;
};

///
/// \ingroup core_objects
/// \class Program
///
class Program {
 public:
  explicit Program(plaidml_program* ptr) : ptr_(details::make_ptr(ptr)) {}

  ///
  /// Returns a textual representation of the program.
  ///
  std::string str(edsl_source_location loc = edsl_source_location::current()) const {
    return ffi::str(ffi::call<plaidml_string*>(loc, plaidml_program_repr, as_ptr()));
  }

  ///
  /// Compiles a program using the specified target.
  ///
  void compile(const std::string& target = "", bool debug = false,
               edsl_source_location loc = edsl_source_location::current()) {
    ffi::call_void(loc, plaidml_program_compile, as_ptr(), debug, target.c_str());
  }

  std::vector<TensorShape> inputs(edsl_source_location loc = edsl_source_location::current()) const {
    auto shapes = details::make_ptr(ffi::call<plaidml_shapes*>(loc, plaidml_program_get_inputs, as_ptr()));
    std::vector<TensorShape> ret(shapes->size);
    for (size_t i = 0; i < shapes->size; i++) {
      ret[i] = TensorShape(shapes->elts[i]);
    }
    return ret;
  }

  std::vector<TensorShape> outputs(edsl_source_location loc = edsl_source_location::current()) const {
    auto shapes = details::make_ptr(ffi::call<plaidml_shapes*>(loc, plaidml_program_get_outputs, as_ptr()));
    std::vector<TensorShape> ret(shapes->size);
    for (size_t i = 0; i < shapes->size; i++) {
      ret[i] = TensorShape(shapes->elts[i]);
    }
    return ret;
  }

  Buffer save(const std::unordered_map<std::string, std::string>& config = {},
              edsl_source_location loc = edsl_source_location::current()) const {
    std::vector<const char*> keys;
    std::vector<const char*> values;
    for (const auto& kvp : config) {
      keys.push_back(kvp.first.c_str());
      values.push_back(kvp.second.c_str());
    }
    return Buffer(
        ffi::call<plaidml_buffer*>(loc, plaidml_program_save, as_ptr(), config.size(), keys.data(), values.data()));
  }

  plaidml_program* as_ptr() const { return ptr_.get(); }

 private:
  std::shared_ptr<plaidml_program> ptr_;
};

///
/// \defgroup core_settings Settings
///

///
/// \ingroup core_settings
/// \struct Settings
/// Settings represent global key/value pairs used for configuration purposes.
///
struct Settings {
  ///
  /// Returns the setting specified by `key`.
  /// \param key string
  /// \return string
  ///
  static std::string get(const std::string& key, edsl_source_location loc = edsl_source_location::current()) {
    return ffi::str(ffi::call<plaidml_string*>(loc, plaidml_settings_get, key.c_str()));
  }

  ///
  /// Sets the setting specified by `key` to the `value` specified.
  /// \param key string
  /// \param value string
  ///
  static void set(const std::string& key, const std::string& value,
                  edsl_source_location loc = edsl_source_location::current()) {
    ffi::call_void(loc, plaidml_settings_set, key.c_str(), value.c_str());
  }
};

inline std::ostream& operator<<(std::ostream& os, const Program& x) {
  os << x.str();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorShape& x) {
  os << x.str();
  return os;
}

}  // namespace plaidml
