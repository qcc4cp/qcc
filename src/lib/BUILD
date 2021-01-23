py_runtime(
    python_version="PY3",
    name = "python-3.7.3",
    files = [],
    interpreter = "python3",
)

py_library(
    name = "helper",
    visibility = ["//visibility:public"],
    srcs = [
        "helper.py",
    ],
    srcs_version = "PY3",
    deps = [
    ],
)

py_library(
    name = "state",
    visibility = ["//visibility:public"],
    srcs = [
        "state.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":helper",
        ":tensor",
    ],
)

py_library(
    name = "ops",
    visibility = ["//visibility:public"],
    srcs = [
        "ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":helper",
        ":state",
        ":tensor",
    ],
)

py_library(
    name = "tensor",
    visibility = ["//visibility:public"],
    srcs = [
        "tensor.py",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "tensor_test",
    size = "small",
    srcs = ["tensor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensor",
    ],
)

py_test(
    name = "state_test",
    size = "small",
    srcs = ["state_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":helper",
        ":state",
    ],
)

py_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        ":state",
     ],
)

py_test(
    name = "helper_test",
    size = "small",
    srcs = ["helper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensor",
	":state",
	":ops",

    ],
)
