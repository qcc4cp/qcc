load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "qcall",
    srcs = [],
    deps = [
        "//src/lib:bell",
        "//src/lib:circuit",
        "//src/lib:helper",
        "//src/lib:ops",
        "//src/lib:optimizer",
        "//src/lib:state",
        "//src/lib:tensor",
        "@qcc_pip//absl_py",
        "@qcc_pip//numpy",
        "@qcc_pip//scipy",
    ],
)

py_binary(
    name = "larose_benchmark",
    srcs = ["larose_benchmark.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)

py_binary(
    name = "tensor_math",
    srcs = ["tensor_math.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)
