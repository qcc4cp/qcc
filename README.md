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
*  Google's `absl` library. Install with 
   `pip install absl-py`
*  `numpy`. Install with 
    `pip install numpy`
*  `scipy`. This library is only used in phase estimation (and could be skipped). Install with 
   `pip install scipy`.
    
## Build

Much of the code is in Python and will run out of the box. 
There is some C++ for the high performance simulation which requires configuration.

The file `src/lib/BUILD` contains the build rule for the C++ xgates extension module.
This module needs to be able to access the Python header (`Python.h`), 
as well as some numpy headers.
Their location might be different on your build machine. Please set the command-line
parameters `-I...` below appropriately to point to these directories.

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
        "-I/usr/include/python3.7m",
        "-I/usr/include/numpy",
    ],
)
```

There is a subtely about `bazel`: All headers must be within the source tree, or in `/usr/include/...` 
To work around this, you can add a symbolic link pointing to your numpy installation
as we've done for the BUILD file above. For example:

```
ln -s ./usr/local/lib/python3.7/dist-packages/numpy/core/include /usr/include/numpy
```

Once `xgates` builds successfully, it has to be imported into `circuit.py`. At the top of this
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

Additionally, to enable Python to find this file, make sure to include in `PYTHONPATH` the
directory where the generated `xgates.so` or `libxgates.so` is being generated. For
example:

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
To test for correct installation, go to `src/lib` and run:

```
    bazel test ...
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
personal infrastructure and time. It is completely independent of Robert's work
at Google.

Reach Robert at
*  https://www.linkedin.com/in/robert-hundt-2000/
*  qcc4cp@gmail.com
