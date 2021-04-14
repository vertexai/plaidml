add_subdirectory(tests)

set(_SRCS ffi.cc)
set(_HDRS
  exec.h
  ffi.h
)

pml_cc_library(
  NAME exec
  TYPE STATIC
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  DEPS
    plaidml::core
    pmlc::compiler
    pmlc::rt
    pmlc::util
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
    pmlc::compiler
    pmlc::rt
    pmlc::util
)

pml_py_library(
  NAME py
  SRCS
    __init__.py
)
