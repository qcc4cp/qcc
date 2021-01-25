# Quantum Computing for Classical Programmers

Source code for the book project by Robert Hundt. In this project, we build infrastructure from the ground up, implement standard algorithms, such as Quantum Teleportation, Grover's Search, QFT, and Shor's integer factorization. We also implement high performance quantum simulation and a transpilation technique to compile our circuits to other infrastructures, such as Qiskit or Cirq. The book itself details this implementation, its motivation and the underlying math, in great detail. At this point, the book has not yet been published.

The code organization is fairly simple. 
*  `src` is the main source directory. All key algorithms are in this directory.
*  `src/lib` contains the library functions for tensors, states, operators, circuits, and so on, as well as their corresponding tests. The algorithms only depend on these library functions.
*  `src/libq` contains the implementation based on a sparse representation.
*  `src/benchmarks` contains just a few benchmarks, as they were mentioned in the book.

## Installation

To run the code we need a few tools:

*  The `bazel` build system. Install from [bazel's homepage](https://docs.bazel.build/versions/master/install.html)

*  We will need Python's `pip` tool to install packages. If it is not available, here is one way to do install it:
```
    sudo apt update
    sudo apt install python3-pip
```

*  Google's `absl` library. Install with 
```
   python3 -m pip install absl-py
```   
   
*  `numpy`. Install with 
```
   python3 -m pip install numpy
```    

*  `scipy`. This library is only used in phase estimation (and could be skipped). Install with 
```
   python3 -m pip install scipy
```
   
* Finally, to get these source onto your computer:
```
    git clone https://github.com/qcc4cp/qcc.git
```
    
## Build

Much of the code is in Python and will run out of the box.  There is
some C++ for the high performance simulation which requires
configuration.

The file `src/lib/BUILD` contains the build rule for the C++ xgates
extension module.  This module needs to be able to access the Python
header (`Python.h`), as well as some numpy headers. These files'
location might be different on your build machine. Please set the
command-line parameter `-I...` below appropriately to point to your
Python installation.

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
        # Configure:
        "-I/usr/include/python3.7m",
    ],
    deps = [
        "@third_party_numpy//:numpy",
    ],
)
```

There is a subtlety about `bazel`: All headers must be within the
source tree, or in `/usr/include/...` To work around this, we have to
point `bazel` to the installation directory for `numpy`.  The
specification for the external numpy installation is in the WORKSPACE
file. Point `path` to your numpy installation's header files,
excluding the final `include` part of the path. The `include` path is
specified in the co-located file `numpy.BUILD`.

```
new_local_repository(
    name = "third_party_numpy",
    path = "/usr/local/lib/python3.7/dist-packages/numpy/core/",
    build_file = __workspace_dir__ + "/numpy.BUILD", 
)
```

Once `xgates` builds successfully, it is imported into `circuit.py`. At the top of this
file is the import statement that might need to be adjusted:

```
# Configure: This line might have to change, depending on
#            the current build environment.
#
# Google internal:
# import xgates
#
# GitHub Linux:
import libxgates as xgates
```

Additionally, to enable Python to find the extension module, make sure
to include in `PYTHONPATH` the directory where the generated
`xgates.so` or `libxgates.so` is being generated. For example:

```
export PYTHONPATH=$PYTHONPATH:/home/usrname/qcc/blaze-bin/src/lib
```

`bazel` also attempts to use the Python interpreter `python`. On systems that
only have a `python3` installed, make sure a `python` is available in the PATH, eg.:

```
cat python
  python3 $@
```

## Run
To build the library and test for correct installation, go to `src/lib` and run:

```
    bazel build all
    bazel test ...
    
    # Make sure xgates was built properly:
    bazel run circuit_test
```
    
The main algorithms are all in `src`.
To run individual algorithms, run any of these command lines (note the missing `.py` extensions):

```
   bazel run arith_classic
   bazel run arith_quantum
   bazel run bernstein
   bazel run deutsch
   bazel run deutsch_jozsa
   bazel run grover
   bazel run order_finding
   bazel run phase_estimation
   bazel run phase_kick
   bazel run shor_classic
   bazel run superdense
   bazel run supremacy
   bazel run swap_test
   bazel run teleportation
   bazel run vqe_simple
```

To test aspects of the sparse implementation:

```
  cd src/libq
  bazel test ...
```

To run the benchmarks:

```
  cd src/benchmarks
  bazel run larose_benchmark
  bazel run tensor_math
```

## About

This code and book were written by Robert Hundt. At the time of this writing, Robert
was a Distinguished Enginer at Google. However, this is a private project, developed on
personal infrastructure and in private time. It is completely independent of Robert's work
at Google.

Reach Robert at
*  https://www.linkedin.com/in/robert-hundt-2000/
*  qcc4cp@gmail.com
