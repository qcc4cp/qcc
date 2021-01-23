py_runtime(
    python_version="PY3",
    name = "python-3.7.3",
    files = [],
    interpreter = "python3",
)

py_library(
    name = "tensor",
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