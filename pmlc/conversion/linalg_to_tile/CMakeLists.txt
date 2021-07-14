pml_cc_library(
  NAME linalg_to_tile
  HDRS
    passes.h
  SRCS
    linalg_to_tile.cc
    utils.cc
  DEPS
    ::passes-gen
    pmlc::dialect::layer::ir
    pmlc::dialect::tile::ir
    pmlc::util
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
