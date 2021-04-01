pml_cc_library(
  NAME x86
  HDRS
    passes.h
    pipeline.h
  SRCS
    heatmap.cc
    heatmap_impl.cc
    pipeline.cc
    register.cc
    trace_linking.cc
    xsmm_lowering.cc
    utils.cc
  DEPS
    ::passes-gen
    ::prng_lowering
    pmlc::compiler
    pmlc::conversion::pxa_to_affine
    pmlc::conversion::scf_to_omp
    pmlc::conversion::stdx_to_llvm
    pmlc::conversion::tile_to_pxa
    pmlc::dialect::layer::transforms
    pmlc::dialect::pxa::analysis
    pmlc::dialect::pxa::transforms
    pmlc::dialect::stdx::transforms
    pmlc::dialect::tile::transforms
    pmlc::dialect::xsmm::ir
    pmlc::transforms
    MLIRAffineToStandard
    MLIRAffineTransforms
    MLIRLLVMIRTransforms
    MLIRMathTransforms
    MLIROpenMPToLLVM
    MLIRSCFToStandard
    MLIRStandardOpsTransforms
    MLIRStandardToLLVM
    omp
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS -gen-pass-decls passes.h.inc
)

pml_cc_library(
  NAME prng_lowering
  HDRS passes.h
  SRCS prng_lowering.cc
  DEPS
    ::passes-gen
    pmlc::dialect::pxa::ir
)

add_custom_command(
  OUTPUT heatmap.cc
  COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tools/heatmap/heatmap.py
      ${CMAKE_CURRENT_SOURCE_DIR}/heatmap_skx_xeonplat_8180_1-7GHz_mblocked.csv.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/heatmap.tpl.cc
      ${CMAKE_CURRENT_BINARY_DIR}/heatmap.cc
  DEPENDS
    heatmap_skx_xeonplat_8180_1-7GHz_mblocked.csv.gz
    heatmap.tpl.cc
)
