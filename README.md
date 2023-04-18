# Quantum Computing for Programmers

This is the open-source repository for the book [Quantum Computing for Programmers](https://www.cambridge.org/us/academic/subjects/computer-science/algorithmics-complexity-computer-algebra-and-computational-g/quantum-computing-programmers?format=HB) by Robert Hundt, Cambridge University Press (QCC4CP for short). The book describes this implementation in great detail, including all the underlying math and derivations. Note, however, that this code base is evolving - not all algorithms found here are discussed in the book.

To get started quickly on the Python sources, you may find the [Quickstart Guide](https://github.com/qcc4cp/qcc/blob/main/resources/quickstart.md) helpful.

This project builds vendor-independent infrastructure from the ground up and implements standard algorithms, such as Quantum Teleportation, Superdense coding, Deutsch-Jozsa, Bernstein-Vazirani, Quantum Phase estimation (QPE), Grover's Search (with application to Quantum counting, amplitude estimation, Mean and Median estimation, 3SAT, Graph Coloring, and Minimum finding), Quantum random walks, VQE, Max-Cut, Subset-Sum, Quantum Fourier Transform (QFT), Shor's integer factorization, Solovay-Kitaev, Principal Component Analysis, and a few more. It also implements high performance quantum simulation and a transpilation technique to compile circuits to other infrastructures, such as Qiskit or Cirq.

The code is organized as follows:
*  `src` is the main source directory. All algorithms are in this directory.
*  `src/lib` contains the library functions for tensors, states, operators, circuits, and so on, as well as their corresponding tests. All algorithms depend on these library functions.
*  `src/libq` contains the sparse implementation.
*  `src/benchmarks` contains a few benchmarks, as they are mentioned in the book.
*  `resources` contains additional text, sections and chapters.
*  `errata` contains the errata for the book - corrections and clarifications.

## Installation

There are several ways to get started on this code base:

*   Instructions for a **Python-only**, minimal setup can be found [here](https://github.com/qcc4cp/qcc/blob/main/resources/quickstart.md#setup).
*   If you have access to **Docker**, the corresponding simple instructions are [here](resources/README.Docker.md)
*   Manual installation on **Linux** (Debian / Ubuntu) are [here](resources/README.Linux.md)
*   For **MacOS**, see [README.MacOS.md](resources/README.MacOS.md).
*   For **Windows** (partially supported), see [README.Windows.md](resources/README.Windows.md).
*   For interactive **SageMath**, see [README.SageMath.md](resources/README.SageMath.md).
*   **CentOS** is also supported (see [README.CentOS.md](resources/README.CentOS.md)).


## Run

The main algorithms are all in `src`.
To run individual algorithms via `bazel`, run any of these command lines. Note the missing `.py` extensions when using `bazel`). Alternatively, run each Python file with `python <file>` (PYTHONPATH must point to the root directory):

```
# Algorithms discussed in the book:
   bazel run arith_classic
   bazel run arith_quantum
   bazel run bernstein
   bazel run counting
   bazel run deutsch
   bazel run deutsch_jozsa
   bazel run entanglement_swap
   bazel run grover
   bazel run max_cut
   bazel run order_finding
   bazel run phase_estimation
   bazel run phase_kick
   bazel run quantum_walk
   bazel run shor_classic
   bazel run simon
   bazel run simon_general
   bazel run solovay_kitaev
   bazel run subset_sum
   bazel run superdense
   bazel run supremacy
   bazel run swap_test
   bazel run teleportation
   bazel run vqe_simple

# Additional algorithms and techniques, to clarify, or
# in preparation of a new edition of the book:
   bazel run amplitude_estimation
   bazel run chsh
   bazel run estimate_pi
   bazel run euclidean_distance
   bazel run graph_coloring
   bazel run hadamard_test
   bazel run hamiltonian_encoding
   bazel run hhl
   bazel run hhl_2x2
   bazel run inversion_test
   bazel run minimum_finding
   bazel run oracle_synth
   bazel run pauli_rep
   bazel run purification
   bazel run qram
   bazel run quantum_mean
   bazel run quantum_median
   bazel run quantum_pca
   bazel run sat3
   bazel run schmidt_decomp
   bazel run spectral_decomp
   bazel run state_prep
   bazel run state_prep_mottonen
   bazel run zy_decomp

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

## Transpilation

To experiment with transpilation, a few things must work together:
   * Specify a target output. For example, to generate a `libq` C++ file, use `--libq=./test.cc`

   * The code should only contain a single `circuit.qc()`-generated circuit. This circuit will not
     be eagerly executed. Instead, all gates and qubits will be collected in an internal IR.

   * There must be a single call to `qc.dump_to_file()`. The circuit as that point
     will be transpiled to the target platform (an example of this can be found in
     `order_finding.py`).

For the given example, the generated file `test.cc` can be compiled and linked with `libq`
with a command-line similar to this one:
```
$ cd qcc/src
$ cc -O2 -Ilibq test.cc libq/qureg.cc libq/apply.cc libq/gates.cc -o a.out -lc++
$ a.out
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
*  Colin Zhu, for pointing to coding problems.
*  Kevin Crook, Univ. of CA, Berkeley, for feedback and discussion of the Chinese Remainder Theorem.
*  [Moez A. AbdelGawad](http://eng.staff.alexu.edu.eg/~moez/), Alexandria University, Egypt, for suggesting Windows and SageMath ports.
*  Stefanie Scherzinger, Universitaet Passau, for corrections and suggesting Docker.
*  Abdolhamid Pourghazi and Stefan Klessinger for providing and maintaining the Dockerfile.
*  Michael Broughton for help with purification.
