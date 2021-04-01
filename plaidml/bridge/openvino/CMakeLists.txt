set(OUTPUT_ROOT ${CMAKE_BINARY_DIR})

if(NOT InferenceEngineDeveloperPackage_FOUND)
  find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

file(GLOB_RECURSE _OPS_SRCS CONFIGURE_DEPENDS "ops/*.cpp")

pml_cc_library(
  NAME openvino
  HDRS
    plaidml_builder.hpp
    plaidml_ops.hpp
    plaidml_util.hpp
  SRCS
    plaidml_builder.cpp
    plaidml_ops.cpp
    plaidml_util.cpp
    ${_OPS_SRCS}
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
  DEFINES
    HALF_ENABLE_CPP11_TYPE_TRAITS
  DEPS
    IE::inference_engine
    IE::ngraph
    half
)

add_subdirectory(plugin)
if(PML_BUILD_TESTS)
  add_subdirectory(tests/functional)
endif()

install(
  PROGRAMS
    ${OpenVINO_MAIN_SOURCE_DIR}/${BIN_FOLDER}/benchmark_app
  DESTINATION "testkit"
  COMPONENT testkit
)

install(
  DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/
  DESTINATION "testkit"
  COMPONENT testkit
  USE_SOURCE_PERMISSIONS
  PATTERN "*.a" EXCLUDE
)

install(
  PROGRAMS
    ${OUTPUT_ROOT}/bin/libplaidml.so.1
    ${OpenVINO_MAIN_SOURCE_DIR}/${BIN_FOLDER}/lib/libformat_reader.so
    ${IE_MAIN_SOURCE_DIR}/temp/tbb/lib/libtbb.so.2
    ${LLVM_BINARY_DIR}/lib/libmlir_runner_utils.so.13git
    ${LLVM_BINARY_DIR}/lib/libomp.so
  DESTINATION "testkit"
  COMPONENT testkit
)

install(
    DIRECTORY ${OpenCV_INSTALL_PATH}/lib/
    DESTINATION "testkit"
    COMPONENT testkit
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
      PATTERN "libopencv_core.so*"
      PATTERN "libopencv_img*.so*"
)
