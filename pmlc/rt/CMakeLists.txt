add_subdirectory(llvm)

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
    ${MLIR_SOURCE_DIR}/lib/ExecutionEngine/RunnerUtils.cpp
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

pml_cc_library(
  NAME all_runtimes
  SRCS all_runtimes.cc
  DEFINES ${_RT_DEFINES}
  DEPS ${_RT_DEPS}
)
