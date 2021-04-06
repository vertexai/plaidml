# Copyright 2020 Intel Corporation

#-------------------------------------------------------------------------------
# C++ used within PML
#-------------------------------------------------------------------------------

set(PML_CXX_STANDARD ${CMAKE_CXX_STANDARD})

set(PML_ROOT_DIR ${PROJECT_SOURCE_DIR})
list(APPEND PML_COMMON_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR}
)

set(PML_DEFAULT_COPTS)
set(PML_DEFAULT_LINKOPTS)
set(PML_TEST_COPTS)

#-------------------------------------------------------------------------------
# Sanitizer configurations
#-------------------------------------------------------------------------------

include(CheckCXXCompilerFlag)

if(${PML_ENABLE_ASAN})
  message("ASAN enabled")
  if(NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(FATAL_ERROR "PML_ENABLE_ASAN requires Debug build")
  endif()
  set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
  check_cxx_compiler_flag(-fsanitize=address COMPILER_SUPPORTS_ASAN)
  unset(CMAKE_REQUIRED_FLAGS)
  if(${COMPILER_SUPPORTS_ASAN})
    list(APPEND PML_C_FLAGS_DEBUG_LIST "-fsanitize=address")
    list(APPEND PML_CXX_FLAGS_DEBUG_LIST "-fsanitize=address")
  else()
    message(FATAL_ERROR "The compiler does not support address sanitizer "
                        "or is missing configuration for address sanitizer")
  endif()
endif()

if(${PML_ENABLE_MSAN})
  message("MSAN Enabled")
  if(NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(FATAL_ERROR "PML_ENABLE_MSAN requires Debug build")
  endif()
  set(CMAKE_REQUIRED_FLAGS "-fsanitize=memory")
  check_cxx_compiler_flag(-fsanitize=memory COMPILER_SUPPORTS_MSAN)
  unset(CMAKE_REQUIRED_FLAGS)
  if(${COMPILER_SUPPORTS_MSAN})
    list(APPEND PML_C_FLAGS_DEBUG_LIST "-fsanitize=memory")
    list(APPEND PML_CXX_FLAGS_DEBUG_LIST "-fsanitize=memory")
  else()
    message(FATAL_ERROR "The compiler does not support memory sanitizer "
                        "or is missing configuration for address sanitizer")
  endif()
endif()

if(${PML_ENABLE_TSAN})
  message("TSAN Enabled")
  if(NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(FATAL_ERROR "PML_ENABLE_TSAN requires Debug build")
  endif()
  set(CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
  check_cxx_compiler_flag(-fsanitize=thread COMPILER_SUPPORTS_TSAN)
  unset(CMAKE_REQUIRED_FLAGS)
  if(${COMPILER_SUPPORTS_TSAN})
    list(APPEND PML_C_FLAGS_DEBUG_LIST "-fsanitize=thread")
    list(APPEND PML_CXX_FLAGS_DEBUG_LIST "-fsanitize=thread")
  else()
    message(FATAL_ERROR "The compiler does not support thread sanitizer "
                        "or is missing configuration for address sanitizer")
  endif()
endif()

#-------------------------------------------------------------------------------
# Size-optimized build flags
#-------------------------------------------------------------------------------

  # TODO(#898): add a dedicated size-constrained configuration.
if(${PML_SIZE_OPTIMIZED})
  pml_select_compiler_opts(PML_SIZE_OPTIMIZED_DEFAULT_COPTS
    MSVC_OR_CLANG_CL
      "/GS-"
      "/GL"
      "/Gw"
      "/Gy"
      "/DNDEBUG"
      "/DPML_STATUS_MODE=0"
  )
  pml_select_compiler_opts(PML_SIZE_OPTIMIZED_DEFAULT_LINKOPTS
    MSVC_OR_CLANG_CL
      "/LTCG"
      "/opt:ref,icf"
  )
  # TODO(#898): make this only impact the runtime (PML_RUNTIME_DEFAULT_...).
  set(PML_DEFAULT_COPTS
      "${PML_DEFAULT_COPTS}"
      "${PML_SIZE_OPTIMIZED_DEFAULT_COPTS}")
  set(PML_DEFAULT_LINKOPTS
      "${PML_DEFAULT_LINKOPTS}"
      "${PML_SIZE_OPTIMIZED_DEFAULT_LINKOPTS}")
endif()
