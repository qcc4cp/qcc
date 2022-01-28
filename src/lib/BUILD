cc_library(
    name = "xgates",
    srcs = [
        "xgates.cc",
    ],
    copts = [
        "-O3",
        "-ffast-math",
        "-DNPY_NO_DEPRECATED_API",
        "-DNPY_1_7_API_VERSION",
    ],
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
    name = "bell",
    visibility = ["//visibility:public"],
    srcs = [
        "bell.py",
    ],
    srcs_version = "PY3",
    deps = [
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
        ":optimizer",
        ":state",
        ":tensor",
    ],
)

# Depending on the build environment, it can be difficult to build
# the target 'xgates'. If you fail to build it, you may simply delete
# the corresponding entry in this filegroup. Execution will be re-directed
# to a working but slow Python implementation.
filegroup(
  name = "all",
  srcs = [
    ":qcall",
    ":xgates",
  ]
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

py_test(
    name = "measure_test",
    size = "small",
    srcs = ["measure_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        ":state",
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
    ],
)


# This test depends on xgates. However, bazel doesn't allow dependencies
# to cc_libraries. This means this test should NOT be run with
#     bazel test circuit_test
#
# Instead, it has to be run manually, via
#     bazel run circuit_test
#
# If the target 'xgates' fails to build, you may still run all algorithms
# using a slower Python fallback implementation. In this case (which,
# hopefully, you will not encounter), simply ignore this particular test.
#
py_binary(
    name = "circuit_test",
    srcs = ["circuit_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":qcall",
    ],
)
