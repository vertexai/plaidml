FetchContent_Declare(
  opencl_hpp_headers
  URL      https://github.com/KhronosGroup/OpenCL-CLHPP/archive/v2.0.12.zip
  URL_HASH SHA256=127936b3a5ef147f23b85fb043599d1480e9e57acabe2d2a67c5dac05aa4ad70
)
SET(BUILD_TESTS OFF CACHE BOOL "Build Unit Tests" FORCE)
FetchContent_MakeAvailable(opencl_hpp_headers)

add_library(opencl_hpp_headers INTERFACE)
target_include_directories(opencl_hpp_headers INTERFACE ${opencl_hpp_headers_SOURCE_DIR}/include)
