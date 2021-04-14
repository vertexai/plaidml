add_subdirectory(ast)
add_subdirectory(compiler)
add_subdirectory(conversion)
add_subdirectory(dialect)
add_subdirectory(rt)
add_subdirectory(target)
add_subdirectory(testing)
add_subdirectory(tools)
add_subdirectory(transforms)
add_subdirectory(util)

get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

pml_cc_library(
  NAME init_passes_and_dialects
  HDRS
    all_dialects.h
    all_passes.h
  SRCS
    all_dialects.cc
  DEPS
    ${MLIR_DIALECT_LIBS}
    ${MLIR_CONVERSION_LIBS}
    MLIRTransforms
    pmlc::conversion::pxa_to_affine
    pmlc::conversion::scf_to_omp
    pmlc::conversion::stdx_to_llvm
    pmlc::conversion::tile_to_pxa
    pmlc::dialect::affinex::transforms
    pmlc::dialect::layer::ir
    pmlc::dialect::layer::transforms
    pmlc::dialect::pxa::ir
    pmlc::dialect::pxa::transforms
    pmlc::dialect::stdx::ir
    pmlc::dialect::stdx::transforms
    pmlc::dialect::tile::ir
    pmlc::dialect::tile::transforms
    pmlc::dialect::xsmm::ir
    pmlc::rt::all_runtimes
    pmlc::target::all
    pmlc::transforms
)

pml_lit_test(
  NAME lit
  DEPS
    pmlc-jit
    pmlc-opt
  CHECKS
    core
    smoke
    test
)
