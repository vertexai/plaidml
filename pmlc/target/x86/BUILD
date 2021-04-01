# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//tools/heatmap:build_defs.bzl", "heatmap")
load("//vendor/mlir:tblgen.bzl", "gentbl")

package(default_visibility = ["//visibility:public"])

heatmap(
    name = "heatmap",
    out = "heatmap.cc",
    csv = "heatmap_skx_xeonplat_8180_1-7GHz_mblocked.csv.gz",
    template = "heatmap.tpl.cc",
)

gentbl(
    name = "gen-pass",
    tbl_outs = [("-gen-pass-decls", "passes.h.inc")],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    td_srcs = ["@llvm-project//mlir:PassBaseTdFiles"],
)

plaidml_cc_library(
    name = "lib",
    srcs = [
        "heatmap.h",
        "heatmap_impl.cc",
        "pass_detail.h",
        "pipeline.cc",
        "trace_linking.cc",
        "utils.cc",
        "utils.h",
        "xsmm_lowering.cc",
        ":heatmap",
    ],
    hdrs = [
        "passes.h",
        "pipeline.h",
    ],
    deps = [
        ":gen-pass",
        ":prng_lowering",
        "//pmlc/compiler",
        "//pmlc/conversion/pxa_to_affine",
        "//pmlc/conversion/scf_to_omp",
        "//pmlc/conversion/stdx_to_llvm",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/dialect/layer/transforms",
        "//pmlc/dialect/pxa/analysis",
        "//pmlc/dialect/pxa/transforms",
        "//pmlc/dialect/stdx/transforms",
        "//pmlc/dialect/tile/transforms",
        "//pmlc/dialect/xsmm",
        "//pmlc/rt",
        "//pmlc/transforms",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:OpenMPToLLVM",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//openmp",
    ],
)

plaidml_cc_library(
    name = "prng_lowering",
    srcs = ["prng_lowering.cc"],
    hdrs = ["passes.h"],
    deps = [
        ":gen-pass",
        "//pmlc/dialect/pxa/ir",
        "//pmlc/rt",
    ],
)

plaidml_cc_library(
    name = "x86",
    srcs = ["register.cc"],
    deps = [":lib"],
    alwayslink = 1,
)
