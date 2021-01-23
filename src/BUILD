py_library(
    name = "qcall",
    srcs = [],
    deps = [
        "//src/lib:circuit",
        "//src/lib:helper",     
        "//src/lib:ops",
        "//src/lib:state",
        "//src/lib:tensor",
    ],
)

py_binary(
    name = "arith_classic",
    srcs = ["arith_classic.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)


py_binary(
    name = "arith_quantum",
    srcs = ["arith_quantum.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)

py_binary(
    name = "bernstein",
    srcs = ["bernstein.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)

py_binary(
    name = "deutsch",
    srcs = ["deutsch.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)

py_binary(
    name = "deutsch_jozsa",
    srcs = ["deutsch_jozsa.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)

py_binary(
    name = "grover",
    srcs = ["grover.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)

py_binary(
    name = "order_finding",
    srcs = ["order_finding.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)

py_binary(
    name = "phase_estimation",
    srcs = ["phase_estimation.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
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

