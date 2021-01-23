# Quantum Computing for Classical Programmers

Source code for the book project by Robert Hundt. In this project, we build infrastructure from the ground up, implement standard algorithms, such as Quantum Teleportation, Grover's Search, QFT, and Shor's integer factorization. We also implement high performance quantum simulation and a transpilation technique to compile our circuits to other infrastructures, such as Qiskit or Cirq. The book itself details this implementation, its motivation and the underlying math, in great detail. At this point, the book has not yet been published.

The code organization is fairly simple. 
*  `src` is the main source directory. All key algorithms are in this directory.
*  `src/lib` contains the library functions for tensors, states, operators, circuits, and so on, as well as their corresponding tests. The algorithms only depend on these library functions.

To run the code we need a few tools:
*  The `bazel` build system. Install from [bazel's homepage](https://docs.bazel.build/versions/master/install.html)
*  Google's `absl` library. Install with `pip install absl-py`
*  `numpy`. Install with `pip install numpy`
*  `scipy`. This library is only used in phase estimation (and could be skipped). Install with `pip install scipy`.
    
To test for correct installation, go to `src` and run:
    `bazel test ...`
    
To run individual algorithms, for example `order_finding.py`, run (and note the missing `.py` extension):
   `bazel run order_finding`
