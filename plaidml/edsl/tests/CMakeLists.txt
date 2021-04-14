pml_cc_test(
  NAME cc_test
  SRCS edsl_test.cc
  DEPS
    plaidml::testenv
    pmlc::testing
  CHECKS
    core
    smoke
    test
)

pml_py_test(
  NAME py_test
  SRC edsl_test.py
  CHECKS
    core
    smoke
    test
)

pml_lit_test(
  NAME lit
  DATA ::cc_test
  CHECKS
    core
    smoke
    test
)
