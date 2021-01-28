pml_cc_library(
  NAME transforms
  HDRS
    contraction.h
    expand_reshape.h
    passes.h
  SRCS
    contraction.cc
    expand_reshape.cc
    materialize.cc
    pad_constraints.cc
    pad_ranges.cc
    padding.cc
    split_main.cc
    replace_scfFor_alloc.cc
  DEPS
    LLVMSupport
    MLIRIR
    pmlc::dialect::stdx::ir
    pmlc::dialect::tile::ir
    pmlc::util
    pmlc::util::bilp
    pmlc::util::math
    ::passes-gen
  PUBLIC
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
