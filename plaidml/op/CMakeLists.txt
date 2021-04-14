add_subdirectory(lib)

set(_SRCS ffi.cc)
set(_HDRS
  ffi.h
  op.h
)

pml_cc_library(
  NAME op
  TYPE STATIC
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  DEPS
    plaidml::core
    plaidml::edsl
    plaidml::op::lib
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
    plaidml::edsl::obj
    plaidml::op::lib
    pmlc::util
)

pml_cc_test(
  NAME op_test
  SRCS op_test.cc
  DEPS
    plaidml::testenv
    pmlc::testing
  CHECKS
    core
    smoke
    test
)

pml_py_library(
  NAME py
  SRCS
    __init__.py
)
