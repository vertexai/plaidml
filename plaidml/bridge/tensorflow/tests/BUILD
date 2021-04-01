load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library", "flatbuffer_py_library")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")

flatbuffer_cc_library(
    name = "archive_cc_fbs",
    srcs = ["archive.fbs"],
)

flatbuffer_py_library(
    name = "archive_py",
    srcs = ["archive.fbs"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [":archive_py"],
)

cc_library(
    name = "codegen_test",
    testonly = True,
    srcs = ["codegen_test.cc"],
    hdrs = ["codegen_test.h"],
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        "//plaidml:testenv",
        "//plaidml/bridge/tensorflow/service:compiler",
        "@org_tensorflow//tensorflow/compiler/xla/tests:test_utils",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "compare_op_test",
    srcs = ["compare_op_test.cc"],
    extra_copts = ["-fexceptions"],
    linkstatic = 1,
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        ":codegen_test",
        "//plaidml/bridge/tensorflow/service:compiler",
    ],
)

py_binary(
    name = "conv_op_test_gen",
    testonly = 1,
    srcs = ["conv_op_test_gen.py"],
    tags = [
        "manual",
        "skip_macos",
        "skip_windows",
    ],
)

genrule(
    name = "conv_op_test_genrule",
    testonly = 1,
    srcs = [],
    outs = ["conv_op_test.h.inc"],
    cmd = """XLA_FLAGS="--xla_dump_to=/tmp/hlo_module" TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=0" $(location :conv_op_test_gen) --output $@""",
    tags = [
        "manual",
        "skip_macos",
        "skip_windows",
    ],
    tools = [":conv_op_test_gen"],
)

tf_cc_test(
    name = "conv_op_test",
    srcs = [
        "conv_op_test.cc",
        # "conv_op_test.h.inc",
    ],
    # data = [":conv_op_test_genrule"],
    extra_copts = ["-fexceptions"],
    linkstatic = 1,
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        ":codegen_test",
        "//plaidml/bridge/tensorflow/service:compiler",
    ],
)

tf_cc_test(
    name = "dot_op_test",
    srcs = ["dot_op_test.cc"],
    extra_copts = ["-fexceptions"],
    linkstatic = 1,
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        ":codegen_test",
        "//plaidml/bridge/tensorflow/service:compiler",
    ],
)

tf_cc_test(
    name = "eltwise_op_test",
    srcs = ["eltwise_op_test.cc"],
    extra_copts = ["-fexceptions"],
    linkstatic = 1,
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        ":codegen_test",
        "//plaidml/bridge/tensorflow/service:compiler",
    ],
)

py_binary(
    name = "i3d_control",
    srcs = ["i3d_control.py"],
    data = ["@kinetics-i3d//:i3d-data"],
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    deps = ["@kinetics-i3d//:i3d"],
)

py_binary(
    name = "i3d_test_gen",
    testonly = 1,
    srcs = ["i3d_test_gen.py"],
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        ":util",
        "@flatbuffers//:runtime_py",
    ],
)

genrule(
    name = "i3d_test_genrule",
    testonly = 1,
    outs = ["i3d.pml"],
    cmd = "$(location :i3d_test_gen) $(location @tfhub_i3d_kinetics_400//:saved_model.pb) $@",
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    tools = [
        ":i3d_test_gen",
        "@tfhub_i3d_kinetics_400//:saved_model.pb",
    ],
)

tf_cc_test(
    name = "i3d_test",
    srcs = ["i3d_test.cc"],
    data = [":i3d_test_genrule"],
    extra_copts = ["-fexceptions"],
    linkstatic = 1,
    tags = [
        "manual",
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        ":archive_cc_fbs",
        ":codegen_test",
        "//plaidml/bridge/tensorflow/service:compiler",
        "@org_tensorflow//tensorflow/compiler/xla/tests:verified_hlo_module",
    ],
)

tf_cc_test(
    name = "logical_op_test",
    srcs = ["logical_op_test.cc"],
    extra_copts = ["-fexceptions"],
    linkstatic = 1,
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        ":codegen_test",
        "//plaidml/bridge/tensorflow/service:compiler",
    ],
)

py_binary(
    name = "shape_op_test_gen",
    testonly = 1,
    srcs = ["shape_op_test_gen.py"],
    tags = [
        "manual",
        "skip_macos",
        "skip_windows",
    ],
)

genrule(
    name = "shape_op_test_genrule",
    testonly = 1,
    srcs = [],
    outs = ["shape_op_test.h.inc"],
    cmd = """XLA_FLAGS="--xla_dump_to=/tmp/hlo_module" TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=0" $(location :shape_op_test_gen) --output $@""",
    tags = [
        "manual",
        "skip_macos",
        "skip_windows",
    ],
    tools = [":shape_op_test_gen"],
)

tf_cc_test(
    name = "shape_op_test",
    srcs = [
        "shape_op_test.cc",
        # "shape_op_test.h.inc",
    ],
    # data = [":shape_op_test_genrule"],
    extra_copts = ["-fexceptions"],
    linkstatic = 1,
    tags = [
        "skip_macos",
        "skip_windows",
    ],
    deps = [
        ":codegen_test",
        "//plaidml/bridge/tensorflow/service:compiler",
    ],
)
