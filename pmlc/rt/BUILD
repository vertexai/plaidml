# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "rt",
    srcs = [
        "boundscheck.cc",
        "builtin.cc",
        "device.h",
        "device_id.cc",
        "executable.cc",
        "internal.h",
        "jit_executable.cc",
        "jit_executable.h",
        "prng.cc",
        "register.cc",
        "runtime_registry.cc",
        "symbol_registry.cc",
        "xsmm.cc",
    ],
    hdrs = [
        "device_id.h",
        "executable.h",
        "register.h",
        "runtime.h",
        "runtime_registry.h",
        "symbol_registry.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/compiler",
        "//pmlc/util:logging",
        "@half",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:MCJIT",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:X86AsmParser",
        "@llvm-project//llvm:X86CodeGen",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:OpenMPToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:mlir_runner_utils",
        "@xsmm",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "all_runtimes",
    srcs = ["all_runtimes.cc"],
    defines = [
        "PML_ENABLE_OPENCL",
        "PML_ENABLE_VULKAN",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":rt",
        "//pmlc/rt/llvm",
        "//pmlc/rt/opencl",
        "//pmlc/rt/vulkan",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "device_id_test",
    size = "small",
    srcs = [
        "device_id_test.cc",
        "internal.h",
    ],
    deps = [
        ":all_runtimes",
        ":rt",
        "//pmlc/target:all_targets",
    ],
)
