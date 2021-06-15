These instructions may be helpful for MacOS.

The first problem you encounter may be that the header Python.h cannot be found.
It may be necessary to expand the file [`qcc/WORKSPACE`](WORKSPACE) and add an 'external
repository', pointing to your Python installation. For example:

```
[...]
new_local_repository(
    name = "third_party_python",
    path = "[system path]/Python/3.7/",
    build_file = __workspace_dir__ + "/python.BUILD",
)
```

With a corresponding file [`python.BUILD`](python.BUILD). You have to ensure that the paths
are set according to your setting:

```
package(
    default_visibility = ["//visibility:public"]
)

cc_library(
    name = "python",
    srcs = [
    ],
    hdrs = glob([
        "include/python3.7m/*.h",
    ]),
    includes = ["include/python3.7m"],
)
```

Finally, modify [`qcc/src/lib/BUILD`](src/lib/BUILD) to point and use this external
repository:

```
cc_library(
    name = "xgates",
    srcs = [
	"xgates.cc",
    ],
    copts = [
        "-O3",
        "-ffast-math",
        "-march=skylake",
        "-DNPY_NO_DEPRECATED_API",
        "-DNPY_1_7_API_VERSION",
    ],
    deps = [
	"@third_party_numpy//:numpy",
	"@third_party_python//:python",
    ],
)
```

Run these commands to verify things work as expected
(replace ... with the appropriate path in your system):

```
# This should build libqgates.so in .../qcc/bazel-bin/src/lib
bazel build xgates

# Set PYTHONPATH to point to this directory
export PYTHONPATH=.../qcc/bazel-bin/src/lib

# Test it
bazel run circuit_test

# Test all tests
bazel test ...

# Run all the algorithms
cd .../qcc/src
./runall.sh
