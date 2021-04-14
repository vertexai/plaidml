set(_HDRS
  core.h
  ffi.h
)

set(_SRCS
  ffi.cc
  settings.cc
  version.cc
)

pml_cc_library(
  NAME core
  TYPE STATIC
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  DEPS
    pmlc::ast
    pmlc::compiler
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
    pmlc::ast
    pmlc::compiler
    pmlc::util
)

pml_py_library(
  NAME py
  SRCS
    __init__.py
    settings.py
)

configure_file(version.in.cc version.cc)
configure_file(_version.in.py _version.py)
