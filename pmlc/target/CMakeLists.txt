add_subdirectory(x86)

pml_cc_library(
  NAME all
  TYPE STATIC
  SRCS all_targets.cc
  DEPS
    pmlc::compiler
    pmlc::target::x86
)

pml_cc_library(
  NAME all_obj
  TYPE OBJECT
  SRCS all_targets.cc
  DEPS
    pmlc::compiler
    pmlc::target::x86
)
