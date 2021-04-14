# Copyright 2021 Intel Corporation

# Heavily inspired by and with gratitude to the IREE project:
# https://github.com/google/iree/blob/main/build_tools/cmake/

include(CMakeParseArguments)
include(pml_installed_test)

# pml_py_library()
#
# Parameters:
# NAME: name of target
# SRCS: List of source files for the library
# DEPS: List of other targets the test python libraries require
function(pml_py_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS;GEN_SRCS"
    ${ARGN}
  )

  pml_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::pml::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_custom_target(${_NAME} ALL
    DEPENDS ${_RULE_DEPS} ${_RULE_SRCS}
  )

  # Copy each source file to the build directory.
  foreach(SRC_FILE ${_RULE_SRCS})
    add_custom_command(
      OUTPUT ${SRC_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}
        ${CMAKE_CURRENT_BINARY_DIR}/${SRC_FILE}
      DEPENDS ${SRC_FILE}
    )
    set_property(TARGET ${_NAME} APPEND PROPERTY PY_SRCS ${CMAKE_CURRENT_BINARY_DIR}/${SRC_FILE})
  endforeach()
  if (${_RULE_GEN_SRCS})
    set_property(TARGET ${_NAME} APPEND PROPERTY PY_SRCS ${_RULE_GEN_SRCS})
  endif()
endfunction()

# pml_py_test()
#
# Parameters:
# NAME: name of test
# SRC: Test source file
# ARGS: Command line arguments to the Python source file.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
function(pml_py_test)
  if(NOT PML_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC"
    "ARGS;LABELS;CHECKS"
    ${ARGN}
  )

  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  pml_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")

  pml_add_installed_test(
    TEST_NAME "${_NAME_PATH}"
    LABELS "${_RULE_LABELS}" "${_RULE_CHECKS}"
    ENVIRONMENT
      "PYTHONPATH=${PROJECT_BINARY_DIR}:$ENV{PYTHONPATH}"
    COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}
      ${_RULE_ARGS}
    INSTALLED_COMMAND
      python
      "${_PACKAGE_PATH}/${_RULE_SRC}"
  )

  install(FILES ${_RULE_SRC}
    DESTINATION "tests/${_PACKAGE_PATH}"
    COMPONENT Tests
  )

  add_custom_target(${_NAME} ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SRC}
  )

  add_custom_command(
    OUTPUT ${_RULE_SRC}
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}
      ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SRC}
  )
endfunction()


# pml_py_cffi()
#
# Rule to generate a cffi python module.
#
# Parameters:
# NAME: Name of target
# MODULE: Name of generated ffi module
# SRCS: List of files for input to cffi
function(pml_py_cffi)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;MODULE"
    "SRCS"
    ${ARGN}
  )

  pml_package_ns(_PACKAGE_NS)
  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  foreach(SRC ${_RULE_SRCS})
    list(APPEND _SRC_ARGS "--source" "${SRC}")
  endforeach()

  add_custom_target(${_NAME} ALL
    DEPENDS ${_RULE_NAME}.py
  )

  add_custom_command(
    OUTPUT ${_RULE_NAME}.py
    COMMAND
      ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/py_cffi/py_cffi.py
        --module ${_RULE_MODULE}
        --output ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.py
        ${_SRC_ARGS}
    DEPENDS
      ${PROJECT_SOURCE_DIR}/tools/py_cffi/py_cffi.py
      ${_RULE_SRCS}
  )
endfunction()


# pml_py_wheel()
#
# Create a python wheel
#
# Parameters:
# NAME: Name of target
# PKG_NAME:
# PLATFORM:
# DEPS:
# PY_DEPS:
# VERSION:
function(pml_py_wheel)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;PKG_NAME;PLATFORM;VERSION"
    "DEPS;PY_DEPS"
    ${ARGN}
  )

  if(NOT _RULE_PLATFORM)
    set(_RULE_PLATFORM "any")
  endif()

  pml_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::pml::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  list(TRANSFORM _RULE_PY_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  list(TRANSFORM _RULE_PY_DEPS REPLACE "::" "_")

  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  configure_file(setup.in.py setup.py)

  set(_ABI "none")
  set(_PY_VER "py3")
  set(_WHEEL_FILE ${_RULE_PKG_NAME}-${_RULE_VERSION}-${_PY_VER}-${_ABI}-${_RULE_PLATFORM}.whl)

  add_custom_target(${_NAME} ALL
    DEPENDS ${_WHEEL_FILE}
  )

  foreach(_PY_DEP ${_RULE_PY_DEPS})
    get_target_property(_PY_SRCS ${_PY_DEP} PY_SRCS)
    list(APPEND _PY_DEP_SRCS ${_PY_SRCS})
  endforeach()

  add_custom_command(
    OUTPUT ${_WHEEL_FILE}
    COMMAND ${PYTHON_EXECUTABLE} setup.py
      "--no-user-cfg"
      "bdist_wheel"
      "--dist-dir" ${CMAKE_CURRENT_BINARY_DIR}
      "--plat-name" ${_RULE_PLATFORM}
    DEPENDS ${_RULE_DEPS} ${CMAKE_CURRENT_BINARY_DIR}/setup.py ${_PY_DEP_SRCS}
  )

  # Copy wheels into top level directory.
  add_custom_command(
    TARGET ${_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_BINARY_DIR}/${_WHEEL_FILE}
      ${PROJECT_BINARY_DIR}/${_WHEEL_FILE}
    COMMENT "Copying ${_WHEEL_FILE} to ${PROJECT_BINARY_DIR}"
  )
endfunction()
