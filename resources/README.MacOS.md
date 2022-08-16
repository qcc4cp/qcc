These instructions may be helpful for MacOS.

The first problem you encounter may be that the header Python.h cannot be found.
It may be necessary to edit the file [`qcc/WORKSPACE`](WORKSPACE) and modify the 'external
repository', pointing to your Python installation. For example:

```
[...]
new_local_repository(
    name = "third_party_python",
    path = "[system path]/Python/3.7/include/python3.7m",
    build_file = __workspace_dir__ + "/python.BUILD",
)
```

With a corresponding file [`python.BUILD`](python.BUILD). You have to ensure that the paths
are set according to your machine setup:

```
package(
    default_visibility = ["//visibility:public"]
)

cc_library(
    name = "python",
    srcs = [
    ],
    hdrs = glob([
        "**/*.h",
    ]),
    includes = [""],
)
```

The BUILD file [`qcc/src/lib/BUILD`](src/lib/BUILD) should already point and use this external
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

On MacOS it appears to make a difference whether or not the command-line option `-c opt` is passed
to build targets. For example the file [`runall.sh`](src/runall.sh) uses this flag. 

Run these commands to verify things work as expected. Replace ... with the appropriate path in your system.

```
# This should build libqgates.so in .../qcc/bazel-bin/src/lib
# Some systems require 
#    bazel build -c opt [target]
# on all build/run targets.

cd .../qcc/src/lib
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
