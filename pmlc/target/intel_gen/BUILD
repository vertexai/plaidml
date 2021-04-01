# Copyright 2019 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:tblgen.bzl", "gentbl")

package(default_visibility = ["//visibility:public"])

gentbl(
    name = "gen-pass",
    tbl_outs = [
        (
            "-gen-pass-decls",
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    td_srcs = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

plaidml_cc_library(
    name = "intel_gen",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        ":gen-pass",
        "//pmlc/compiler",
        "//pmlc/conversion/comp_to_llvm",
        "//pmlc/conversion/gpu",
        "//pmlc/conversion/gpu_to_spirv",
        "//pmlc/conversion/pxa_to_affine",
        "//pmlc/conversion/stdx_to_llvm",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/dialect/affinex/transforms",
        "//pmlc/dialect/comp/transforms",
        "//pmlc/dialect/layer/transforms",
        "//pmlc/dialect/pxa/transforms",
        "//pmlc/dialect/stdx/transforms",
        "//pmlc/dialect/tile/transforms",
        "//pmlc/rt/vulkan",
        "//pmlc/target/x86:prng_lowering",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:GPUToSPIRV",
        "@llvm-project//mlir:GPUToVulkanTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:SCFToGPUPass",
        "@llvm-project//mlir:SPIRVTransforms",
        "@llvm-project//mlir:StandardOpsTransforms",
    ],
    alwayslink = 1,
)
