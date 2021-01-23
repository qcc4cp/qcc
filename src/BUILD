py_library(
    name = "qcall",
    srcs = [],
    deps = [
        "//src/lib:helper",
        "//src/lib:ops",
        "//src/lib:state",
        "//src/lib:tensor",
    ],
)

py_binary(
    name = "phase_kick",
    srcs = ["phase_kick.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)