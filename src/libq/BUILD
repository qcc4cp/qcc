cc_library(
    name = "libq",
    srcs = [
        "apply.cc",
        "gates.cc",
        "qureg.cc",
    ],
    hdrs = ["libq.h"],
    copts = [
        "-O3",
        "-ffast-math",
        "-march=skylake",
        "-fstrict-aliasing",
        "-Wno-class-memaccess",
        "-Wno-alloc-size-larger-than",
    ],
)

cc_library(
    name = "libq_jit",
    srcs = [
        "apply.cc",
        "gates_jit.cc",
        "qureg.cc",
    ],
    hdrs = ["libq.h"],
    copts = [
        "-O3",
        "-ffast-math",
        "-march=skylake",
        "-fstrict-aliasing",
        "-Wno-class-memaccess",
        "-Wno-alloc-size-larger-than",
    ],
)

cc_test(
    name = "libq_test",
    srcs = ["libq_test.cc"],
    deps = [
        ":libq",
    ],
)

cc_test(
    name = "libq_arith_test",
    srcs = ["libq_arith_test.cc"],
    copts = [
        "-O3",
        "-ffast-math",
        "-march=skylake",
        "-fstrict-aliasing",
    ],
    deps = [
        ":libq",
    ],
)

cc_test(
    name = "libq_arith_jit_test",
    srcs = ["libq_arith_test.cc"],
    copts = [
        "-O3",
        "-ffast-math",
        "-march=skylake",
        "-fstrict-aliasing",
    ],
    deps = [
        ":libq_jit",
    ],
)

cc_test(
    name = "libq_order22_test",
    srcs = ["libq_order22_test.cc"],
    copts = [
        "-O3",
        "-ffast-math",
        "-march=skylake",
        "-fstrict-aliasing",
    ],
    deps = [
        ":libq",
    ],
)

cc_test(
    name = "libq_order22_jit_test",
    srcs = ["libq_order22_test.cc"],
    copts = [
        "-O3",
        "-ffast-math",
        "-march=skylake",
        "-fstrict-aliasing",
    ],
    deps = [
        ":libq_jit",
    ],
)
