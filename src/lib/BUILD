load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

cc_binary(
    name = "libxgates.so",
    linkshared = True,
    srcs = [
        "xgates.cc",
    ],
    copts = [
        "-O3",
        "-ffast-math",
        "-DNPY_NO_DEPRECATED_API",
    ],
    # Python symbols are resolved at load time from the embedding interpreter,
    # so we must not fail the link on unresolved Python/NumPy symbols. On macOS
    # this requires -undefined dynamic_lookup.
    linkopts = select({
        "@platforms//os:macos": ["-undefined", "dynamic_lookup"],
        "//conditions:default": [],
    }),
    deps = [
        "@third_party_numpy//:numpy",
        "@third_party_python//:python",
    ],
)

py_library(
    name = "tensor",
    visibility = ["//visibility:public"],
    srcs = [
        "tensor.py",
    ],
    srcs_version = "PY3",
    deps = [
        "@qcc_pip//numpy",
        "@qcc_pip//absl_py",
    ],
)

py_library(
    name = "helper",
    visibility = ["//visibility:public"],
    srcs = [
        "helper.py",
    ],
    srcs_version = "PY3",
    deps = [        "@qcc_pip//numpy",
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
        "@qcc_pip//numpy",
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
        "@qcc_pip//numpy",
    ],
)

py_library(
    name = "bell",
    visibility = ["//visibility:public"],
    srcs = [
        "bell.py",
    ],
    srcs_version = "PY3",
    deps = [        "@qcc_pip//numpy",
    ],
)

py_library(
    name = "ir",
    visibility = ["//visibility:public"],
    srcs = [
        "ir.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":ops",
        ":state",
        ":tensor",
    ],
)

py_library(
    name = "dumpers",
    visibility = ["//visibility:public"],
    srcs = [
        "dumpers.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":ir",
    ],
)

py_library(
    name = "optimizer",
    visibility = ["//visibility:public"],
    srcs = [
        "optimizer.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":ir",
    ],
)

py_library(
    name = "circuit",
    visibility = ["//visibility:public"],
    srcs = [
        "circuit.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":dumpers",
        ":ir",
        ":ops",
        ":state",
        ":tensor",
        "@qcc_pip//numpy",
        "@qcc_pip//scipy",
        "@qcc_pip//absl_py",
    ],
)

# Catch all libraries.
py_library(
    name = "qcall",
    deps = [
        ":bell",
        ":circuit",
        ":helper",
        ":ir",
        ":ops",
        ":state",
        ":tensor",
    ],
)

py_test(
    name = "bell_test",
    size = "small",
    srcs = ["bell_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensor",
        ":state",
        ":ops",
        ":bell",
        "@qcc_pip//numpy",
        "@qcc_pip//absl_py",
    ],
)

py_test(
    name = "tensor_test",
    size = "small",
    srcs = ["tensor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensor",
        "@qcc_pip//absl_py",
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
        "@qcc_pip//numpy",
        "@qcc_pip//absl_py",
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
        "@qcc_pip//numpy",
        "@qcc_pip//absl_py",
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
        "@qcc_pip//numpy",
        "@qcc_pip//absl_py",
    ],
)

py_test(
    name = "measure_test",
    size = "small",
    srcs = ["measure_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        ":state",
        "@qcc_pip//absl_py",
    ],
)

py_test(
    name = "equalities_test",
    size = "small",
    srcs = ["equalities_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bell",
        ":ops",
        ":state",
        "@qcc_pip//numpy",
        "@qcc_pip//absl_py",
    ],
)

py_test(
    name = "circuit_test",
    size = "small",
    srcs = ["circuit_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
        "@qcc_pip//numpy",
        "@qcc_pip//absl_py",
    ],
)
