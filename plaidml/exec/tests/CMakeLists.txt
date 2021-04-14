pml_cc_test(
  NAME cc_test
  SRCS list_devices_test.cc
  DEPS
    plaidml::testenv
    pmlc::testing
  CHECKS
    core
    smoke
    test
)
