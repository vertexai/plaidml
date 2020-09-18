# Copyright 2020 Intel Corporation

load("//vendor/bazel:repo.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def openvino_workspace():
    http_archive(
        name = "ade",
        url = "https://github.com/opencv/ade/archive/cbe2db61a659c2cc304c3837406f95c39dfa938e.zip",
        strip_prefix = "ade-cbe2db61a659c2cc304c3837406f95c39dfa938e",
        sha256 = "6660e1b66bd3d8005026155571a057765ace9b0fdd9899aaa5823eca12847896",
        build_file = clean_dep("//vendor/openvino:ade.BUILD"),
    )

    #new_git_repository(
    #    name = "openvino",
    #    # sha256 = "40652941587e579d45a190731960008827221d11575f7f2e6162285b6625b940",
    #    remote = "file:///home/tim/openvino/.git",
    #    commit = "ced877e4a8723161073eabd30beee92011c3ccad",
    #    init_submodules = 1,
    #    build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    #)

    new_git_repository(
        name = "openvino",
        # sha256 = "40652941587e579d45a190731960008827221d11575f7f2e6162285b6625b940",
        remote = "file:///home/nchoudhu/github_repo/openvino/.git",
        commit = "35768e452c3c97bc40a9a4921599ac478b8b1d12",
        init_submodules = 1,
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
