# Quantum Computing for Programmers

In this project, we build vendor-independent infrastructure from the ground up and implement standard algorithms, such as Quantum Teleportation, Quantum Phase estimation (QPE), Grover's Search, Quantum counting, Quantum random walks, VQE, QAOA, Max-Cut, Subset-Sum, Quantum Fourier Transform (QFT), Shor's integer factorization, and Solovay-Kitaev. We also implement high performance quantum simulation and a transpilation technique to compile our circuits to other infrastructures, such as Qiskit or Cirq. 

This is the open-source repository for the book [Quantum Computing for Programmers](https://www.cambridge.org/us/academic/subjects/computer-science/algorithmics-complexity-computer-algebra-and-computational-g/quantum-computing-programmers?format=HB) by Robert Hundt, Cambridge University Press, estimated arrival April 2022. The book describes this implementation in great detail, including all the underlying math and derivations.

The code is organized as follows:
*  `src` is the main source directory. All algorithms are in this directory.
*  `src/lib` contains the library functions for tensors, states, operators, circuits, and so on, as well as their corresponding tests. All algorithms depend on these library functions.
*  `src/libq` contains the sparse implementation.
*  `src/benchmarks` contains a few benchmarks, as they are mentioned in the book.
*  `resources` contains additional text, sections and chapters.
*  `errata` contains the errata for the book - corrections and clarifications.   

## Installation

These instructions focus on Debian Linux. For MacOS, see [README.MacOS.md](README.MacOS.md). For Windows (partially supported), see [README.Windows.md](README.Windows.md). For interactive SageMath, see [README.SageMath.md](README.SageMath.md). 
CentOS is also supported (see [README.CentOS.md](README.CentOS.md)).
We may add other OS'es in the future.

To run the code a few tools are needed:

*  The `bazel` build system. Install from [bazel's homepage](https://docs.bazel.build/versions/master/install.html)

*  We will need Python's `pip` tool to install packages and `git` to manage the source.
  Here is one way to install them:
```
    sudo apt-get install python3-pip
    sudo apt-get install git
```

*  We need Google's `absl` library, as well as `numpy` and `scipy`. Install with
```
   sudo python3 -m pip install absl-py
   sudo python3 -m pip install numpy
   sudo python3 -m pip install scipy
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
header (`Python.h`), as well as certain `numpy` headers. These files'
location may be different on your build machine. The location
is controlled with the `numpy` and `python` dependencies, which we
explain in a second:

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

There is a subtlety about `bazel`: All headers must be within the
source tree, or in `/usr/include/...` To work around this, we have to
point `bazel` to the installation directories of `numpy` and `python`.  The
specification for the external installations is in the `WORKSPACE`
file. Point `path` to your installation's header files,
excluding the final `include` part of the path. The `include` path is
specified in the co-located files [`numpy.BUILD`](numpy.BUILD) and [`python.BUILD`](python.BUILD). Both
of these file should not require modification (in most cases).

```
new_local_repository(
    name = "third_party_numpy",
    build_file = __workspace_dir__ + "/numpy.BUILD",
    # Configure:
    path = "/usr/local/lib/python3.7/dist-packages/numpy/core/",
)

new_local_repository(
    name = "third_party_python",
    build_file = __workspace_dir__ + "/python.BUILD",
    # Configure:
    path = "/usr/include/python3.9",
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
export PYTHONPATH=$PYTHONPATH:/home/usrname/qcc/bazel-bin/src/lib
```

`bazel` also attempts to use the Python 2 interpreter `python`. If it
is not available on a system, install via:

```
sudo apt-get install python
```

## Run
To build the library and test for correct installation, go to `src/lib` and run:

```
    bazel build all
    bazel test ...

    # Make sure to set PYTHONPATH (once):
    export PYTHONPATH=$PYTHONPATH:/home/usrname/qcc/bazel-bin/src/lib

    # Ensure xgates was built properly:
    bazel run circuit_test
```

The main algorithms are all in `src`.
To run individual algorithms, run any of these command lines (note the missing `.py` extensions):

```
   bazel run arith_classic
   bazel run arith_quantum
   bazel run bernstein
   bazel run counting
   bazel run deutsch
   bazel run deutsch_jozsa
   bazel run entanglement_swap
   bazel run grover
   bazel run hadamard_test
   bazel run inversion_test
   bazel run max_cut
   bazel run order_finding
   bazel run pauli_rep
   bazel run phase_estimation
   bazel run phase_kick
   bazel run quantum_walk
   bazel run shor_classic
   bazel run simon
   bazel run simon_general
   bazel run solovay_kitaev
   bazel run spectral_decomp
   bazel run subset_sum
   bazel run superdense
   bazel run supremacy
   bazel run swap_test
   bazel run teleportation
   bazel run vqe_simple
   bazel run zy_decompose
```

or, more general:
```
for algo in `ls -1 *py | sed s@.py@@g`
do
   bazel run $algo
done
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
is a Distinguished Enginer at Google. However, this is a private project, developed on
personal infrastructure and in private time. It is completely independent of Robert's work
at Google.

Reach Robert at
*  https://www.linkedin.com/in/robert-hundt-2000/
*  qcc4cp@gmail.com (site-specific email account)

### Additional Thanks
Colin Zhu, for pointing to coding problems.  
Kevin Crook, Univ. of CA, Berkeley, for feedback and discussion of the Chinese Remainder Theorem.  
[Moez A. AbdelGawad](http://eng.staff.alexu.edu.eg/~moez/), Alexandria University, Egypt, for suggesting Windows and SageMath ports.  
