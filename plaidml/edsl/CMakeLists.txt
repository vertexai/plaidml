add_subdirectory(tests)

set(_SRCS ffi.cc)
set(_HDRS
  edsl.h
  ffi.h
)

pml_cc_library(
  NAME edsl
  TYPE STATIC
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  DEPS
    plaidml::core
    pmlc::ast
    pmlc::util
    LLVMSupport
  COMPONENTS
    devkit
)

pml_cc_library(
  NAME obj
  TYPE OBJECT
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  DEPS
    plaidml::core::obj
    pmlc::ast
    pmlc::util
    LLVMSupport
)

pml_py_library(
  NAME py
  SRCS
    __init__.py
)
