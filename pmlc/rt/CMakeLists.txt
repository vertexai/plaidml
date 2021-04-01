add_subdirectory(llvm)

if(PML_ENABLE_VULKAN)
  add_subdirectory(vulkan)
endif()

if(PML_ENABLE_OPENCL)
  add_subdirectory(opencl)
endif()

pml_cc_library(
  NAME rt
  HDRS
    device_id.h
    executable.h
    register.h
    runtime.h
    runtime_registry.h
    symbol_registry.h
  SRCS
    boundscheck.cc
    builtin.cc
    device.h
    device_id.cc
    executable.cc
    internal.h
    jit_executable.cc
    jit_executable.h
    prng.cc
    register.cc
    runtime_registry.cc
    symbol_registry.cc
    xsmm.cc
  DEPS
    pmlc::compiler
    pmlc::util
    half
    xsmm
    LLVMExecutionEngine
    LLVMMCJIT
    LLVMOrcJIT
    LLVMSupport
    LLVMX86AsmParser
    LLVMX86CodeGen
    MLIRExecutionEngine
    MLIROpenMPToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    MLIRTransforms
    mlir_runner_utils
)

set_property(SOURCE xsmm.cc
  PROPERTY COMPILE_DEFINITIONS
    __BLAS=0
    LIBXSMM_DEFAULT_CONFIG
)

set(_RT_DEFINES "")
set(_RT_DEPS 
  ::rt
  pmlc::rt::llvm
)

if(PML_ENABLE_OPENCL)
  list(APPEND _RT_DEFINES PML_ENABLE_OPENCL)
  list(APPEND _RT_DEPS pmlc::rt::opencl)
endif()

if(PML_ENABLE_VULKAN)
  list(APPEND _RT_DEFINES PML_ENABLE_VULKAN)
  list(APPEND _RT_DEPS pmlc::rt::vulkan)
endif()

pml_cc_library(
  NAME all_runtimes
  SRCS all_runtimes.cc
  DEFINES ${_RT_DEFINES}
  DEPS ${_RT_DEPS}
)
