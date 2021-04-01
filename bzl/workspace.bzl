# Copyright 2020 Intel Corporation

load("//vendor/bazel:repo.bzl", "http_archive")
load("//vendor/conda:repo.bzl", "conda_repo")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def plaidml_workspace():
    http_archive(
        name = "bazel_latex",
        sha256 = "5119802a5fbe2f27914af455c59b4ecdaaf57c0bc6c63da38098a30d94f48c9a",
        strip_prefix = "bazel-latex-b6375d9df2952548c3371c0c865710655e8b1cc1",
        url = "https://github.com/plaidml/bazel-latex/archive/b6375d9df2952548c3371c0c865710655e8b1cc1.zip",
    )

    http_archive(
        name = "boost",
        url = "https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz",
        sha256 = "96b34f7468f26a141f6020efb813f1a2f3dfb9797ecf76a7d7cbd843cc95f5bd",
        strip_prefix = "boost_1_71_0",
        build_file = clean_dep("//bzl:boost.BUILD"),
    )

    http_archive(
        name = "com_github_google_benchmark",
        url = "https://github.com/google/benchmark/archive/v1.5.0.tar.gz",
        sha256 = "3c6a165b6ecc948967a1ead710d4a181d7b0fbcaa183ef7ea84604994966221a",
        strip_prefix = "benchmark-1.5.0",
    )

    http_archive(
        name = "com_google_googletest",
        url = "https://github.com/google/googletest/archive/release-1.10.0.tar.gz",
        sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
        strip_prefix = "googletest-release-1.10.0",
    )

    conda_repo(
        name = "com_intel_plaidml_conda",
        env_unix = clean_dep("//conda:unix.yml"),
        build_file_unix = clean_dep("//conda:unix.BUILD"),
        env_windows = clean_dep("//conda:windows.yml"),
        build_file_windows = clean_dep("//conda:windows.BUILD"),
    )

    http_archive(
        name = "crosstool_ng_linux_x86_64_gcc_8.3.0",
        build_file = clean_dep("//toolchain:crosstool_ng/linux_x86_64.BUILD"),
        sha256 = "091f5732882a499c6b9fb5fcb895176d0c96e958236e16b61d1a9cafec4271ad",
        strip_prefix = "x86_64-unknown-linux-gnu",
        url = "https://github.com/plaidml/depot/raw/master/toolchain/gcc-8.3/x86_64-unknown-linux-gnu-20191010.tgz",
    )

    http_archive(
        name = "easylogging",
        url = "https://github.com/amrayn/easyloggingpp/archive/v9.96.7.tar.gz",
        sha256 = "237c80072b9b480a9f2942b903b4b0179f65e146e5dcc64864dc91792dedd722",
        build_file = clean_dep("//bzl:easylogging.BUILD"),
        strip_prefix = "easyloggingpp-9.96.7",
    )

    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

    http_archive(
        name = "half",
        url = "https://github.com/plaidml/depot/raw/master/half-1.11.0.zip",
        sha256 = "9e5ddb4b43abeafe190e780b5b606b081acb511e6edd4ef6fbe5de863a4affaf",
        strip_prefix = "half-1.11.0",
        build_file = clean_dep("//bzl:half.BUILD"),
    )

    http_archive(
        name = "io_bazel_rules_jsonnet",
        sha256 = "d05d719c4738e8aac5f13b32f745ff4832b9638ecc89ddcb6e36c379a1ada025",
        strip_prefix = "rules_jsonnet-0.1.0",
        url = "https://github.com/bazelbuild/rules_jsonnet/archive/0.1.0.zip",
    )

    http_archive(
        name = "jsonnet",
        url = "https://github.com/google/jsonnet/archive/v0.13.0.zip",
        sha256 = "e9f7095dd2a383001188aa622edaf82059732e11d74f8d0bfdfa84f2682dd547",
        strip_prefix = "jsonnet-0.13.0",
    )

    LLVM_COMMIT = "4f9da4c89c754f49cdd465f316d83d35094a18c2"
    LLVM_SHA256 = "57ebfa9121a785bf853d45186162f3f5d77f5223d68374dde7857938c0fd9e79"
    LLVM_URL = "https://github.com/plaidml/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)
    http_archive(
        name = "llvm-project",
        url = LLVM_URL,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        link_files = {
            clean_dep("//vendor/llvm:llvm.BUILD"): "llvm/BUILD.bazel",
            clean_dep("//vendor/mlir:mlir.BUILD"): "mlir/BUILD.bazel",
            clean_dep("//vendor/mlir:test.BUILD"): "mlir/test/BUILD.bazel",
            clean_dep("//vendor/openmp:openmp.BUILD"): "openmp/BUILD.bazel",
        },
        override = "PLAIDML_LLVM_REPO",
    )

    http_archive(
        name = "pybind11",
        url = "https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz",
        sha256 = "b69e83658513215b8d1443544d0549b7d231b9f201f6fc787a2b2218b408181e",
        strip_prefix = "pybind11-2.2.4",
        build_file = clean_dep("//bzl:pybind11.BUILD"),
    )

    http_archive(
        name = "rules_pkg",
        sha256 = "e46b4f5aa71d1037c7c8142e2fedb503127af4bbd9dbde4a742d119749f68a3f",
        strip_prefix = "rules_pkg-cb54c427343aa48c32e3c09ddcc8f6316cdbd5a6/pkg",
        url = "https://github.com/bazelbuild/rules_pkg/archive/cb54c427343aa48c32e3c09ddcc8f6316cdbd5a6.tar.gz",
    )

    http_archive(
        name = "rules_python",
        sha256 = "b5bab4c47e863e0fbb77df4a40c45ca85f98f5a2826939181585644c9f31b97b",
        strip_prefix = "rules_python-9d68f24659e8ce8b736590ba1e4418af06ec2552",
        url = "https://github.com/bazelbuild/rules_python/archive/9d68f24659e8ce8b736590ba1e4418af06ec2552.tar.gz",
    )

    http_archive(
        name = "tbb",
        url = "https://github.com/intel/tbb/archive/v2020.1.zip",
        sha256 = "1550c9cbf629435acd6699f9dd3d8841c1d5e0eaf0708f54d328c8cd020951c1",
        strip_prefix = "oneTBB-2020.1",
        build_file = clean_dep("//vendor/tbb:tbb.BUILD"),
    )

    http_archive(
        name = "volk",
        url = "https://github.com/zeux/volk/archive/2638ad1b2b40f1ad402a0a6ac55b60bc51a23058.zip",
        sha256 = "4a5fb828e05d8c86f696f8754e90302d6446b950236256bcb4857408357d2b60",
        strip_prefix = "volk-2638ad1b2b40f1ad402a0a6ac55b60bc51a23058",
        build_file = clean_dep("//vendor/volk:volk.BUILD"),
    )

    http_archive(
        name = "vulkan_headers",
        url = "https://github.com/KhronosGroup/Vulkan-Headers/archive/v1.2.132.zip",
        sha256 = "e6b5418e3d696ffc7c97991094ece7cafc4c279c8a88029cc60e587bc0c26068",
        strip_prefix = "Vulkan-Headers-1.2.132",
        build_file = clean_dep("//vendor/vulkan_headers:vulkan_headers.BUILD"),
    )

    http_archive(
        name = "xsmm",
        url = "https://github.com/hfp/libxsmm/archive/592d72f2902fa49fda51a4221e83f7f978936a9d.zip",
        sha256 = "2595c625e9e63099de8bea4876a7e5835f0c5f560049db6de31611da2539d448",
        strip_prefix = "libxsmm-592d72f2902fa49fda51a4221e83f7f978936a9d",
        build_file = clean_dep("//vendor/xsmm:xsmm.BUILD"),
    )

    http_archive(
        name = "zlib",
        url = "https://www.zlib.net/zlib-1.2.11.tar.gz",
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        build_file = clean_dep("//bzl:zlib.BUILD"),
    )

    http_archive(
        name = "opencl_headers",
        url = "https://github.com/KhronosGroup/OpenCL-Headers/archive/v2020.06.16.zip",
        sha256 = "518703d3c3a6333bcf8e4f80758e4e98f7af30fbd72a09fe8c2673da1628d80c",
        strip_prefix = "OpenCL-Headers-2020.06.16",
        build_file = clean_dep("//vendor/opencl_headers:opencl_headers.BUILD"),
    )

    http_archive(
        name = "opencl_hpp_headers",
        url = "https://github.com/KhronosGroup/OpenCL-CLHPP/archive/v2.0.12.zip",
        sha256 = "127936b3a5ef147f23b85fb043599d1480e9e57acabe2d2a67c5dac05aa4ad70",
        strip_prefix = "OpenCL-CLHPP-2.0.12",
        build_file = clean_dep("//vendor/opencl_hpp_headers:opencl_hpp_headers.BUILD"),
        # Patch hpp headers trying to use system OpenCL header on Apple.
        patches = [clean_dep("//vendor/opencl_hpp_headers:fix_apple_headers.patch")],
    )

    http_archive(
        name = "opencl_icd_loader",
        url = "https://github.com/KhronosGroup/OpenCL-ICD-Loader/archive/v2020.06.16.zip",
        sha256 = "e4c27a5adcef4dbc0fee98864af203dc78dfc967ca7287c9bad9add030e7516e",
        strip_prefix = "OpenCL-ICD-Loader-2020.06.16",
        build_file = clean_dep("//vendor/opencl_icd_loader:opencl_icd_loader.BUILD"),
    )
