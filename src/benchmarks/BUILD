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
