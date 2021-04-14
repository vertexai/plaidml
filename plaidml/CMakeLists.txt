add_subdirectory(bridge)
add_subdirectory(core)
add_subdirectory(edsl)
add_subdirectory(exec)
add_subdirectory(op)

include(symbols.cmake)

if (NOT WIN32)
  pml_cc_library(
    NAME plaidml
    TYPE SHARED
    LINKOPTS
      # TODO: use platform-specific flags
      -Wl,--version-script ${LINUX_LDS}
    PROPS
      LINK_DEPENDS ${LINUX_LDS}
      LINK_DEPENDS ${MACOS_LD}
      OUTPUT_NAME plaidml
      SOVERSION 1
    PRIVATE_DEPS
      plaidml::core::obj
      plaidml::edsl::obj
      plaidml::exec::obj
      plaidml::op::obj
      pmlc::rt::all_runtimes
      pmlc::target::all_obj
  )
else()
  pml_cc_library(
    NAME plaidml
    TYPE STATIC
    DEPS
      plaidml::core::obj
      plaidml::edsl::obj
      plaidml::exec::obj
      plaidml::op::obj
      pmlc::rt::all_runtimes
      pmlc::target::all_obj
  )
endif()

pml_cc_library(
  NAME testenv
  HDRS testenv.h
  SRCS testenv.cc
  DEPS
    plaidml::core
    plaidml::edsl
    plaidml::exec
    plaidml::op
    pmlc::rt::all_runtimes
    pmlc::target::all
    gflags
    gmock
    gtest
    half
  COMPONENTS
    devkit
)

pml_py_library(
  NAME py
  SRCS
    __init__.py
    ffi.py
    plaidml_setup.py
  DEPS
    ::plaidml
    plaidml__ffi
)

pml_py_cffi(
    NAME _ffi
    MODULE plaidml._ffi
    SRCS
      ${CMAKE_CURRENT_SOURCE_DIR}/core/ffi.h
      ${CMAKE_CURRENT_SOURCE_DIR}/edsl/ffi.h
      ${CMAKE_CURRENT_SOURCE_DIR}/exec/ffi.h
      ${CMAKE_CURRENT_SOURCE_DIR}/op/ffi.h
)

# TODO: support other platforms
set(_PLAIDML_BINARY "libplaidml.so")

add_custom_command(
  TARGET plaidml_plaidml
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_PLAIDML_BINARY}
    ${CMAKE_CURRENT_BINARY_DIR}/${_PLAIDML_BINARY}
)

pml_py_wheel(
  NAME wheel
  PKG_NAME "plaidml"
  VERSION ${PLAIDML_VERSION}
  PLATFORM "manylinux1_x86_64" # TODO: support other platforms
  DEPS ::plaidml
  PY_DEPS
    ::py
    plaidml::core::py
    plaidml::edsl::py
    plaidml::exec::py
    plaidml::op::py
)
