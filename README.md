# Quantum Computing for Programmers

This is the open-source repository for the book [Quantum Computing for Programmers, 2nd Edition](https://www.cambridge.org/gb/universitypress/subjects/computer-science/algorithmics-complexity-computer-algebra-and-computational-g/quantum-computing-programmers-2nd-edition?format=AR&isbn=9781009548564) by Robert Hundt, Cambridge University Press. The book describes the implementations in this reposoitory in great detail, including all the underlying math and derivations. Note, however, that this code base is evolving.

To get started quickly on the Python sources, you may find the [Quickstart Guide](https://github.com/qcc4cp/qcc/blob/main/resources/quickstart.md) helpful.

This project builds vendor-independent infrastructure from the ground up and implements standard algorithms, such as Quantum Teleportation, Superdense coding, Deutsch-Jozsa, Bernstein-Vazirani, Quantum Phase estimation (QPE), Grover's Search (with application to Quantum counting, amplitude estimation, Mean and Median estimation, 3SAT, Graph Coloring, and Minimum finding), Quantum random walks, VQE, Max-Cut, Subset-Sum, Quantum Fourier Transform (QFT), Shor's integer factorization, Solovay-Kitaev, Principal Component Analysis, and a few more. It also implements high performance quantum simulation and a transpilation technique to compile circuits to other infrastructures, such as Qiskit or Cirq.

The code is organized as follows:
*  `src` is the main source directory. All algorithms are in this directory.
*  `src/lib` contains the library functions for tensors, states, operators, circuits, and so on, as well as their corresponding tests. All algorithms depend on these library functions.
*  `src/libq` contains the sparse implementation.
*  `src/benchmarks` contains a few benchmarks, as they are mentioned in the book.
*  `resources` contains additional text, sections and chapters.
*  `errata` contains the errata for the book - corrections and clarifications.
*  `bazel/` contains the Bazel module extension that locates the `python` and
   `numpy` C headers from the active interpreter (used to build the C++
   accelerator). External dependencies are managed with Bzlmod in
   `MODULE.bazel`.

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

The main algorithms are all in `src`. Because the algorithms import the
library as `from src.lib import ...`, run them **as modules from the
repository root** so that `src` is importable:

```
   export PYTHONPATH=$PWD/src/lib   # so 'import libxgates' finds the accelerator
   python3 -m src.arith_classic     # note: no .py, and the 'src.' prefix
```

Equivalently, run everything at once with the helper script, which also builds
the C++ accelerator on first use:

```
   ./src/runall.sh                  # runs every algorithm
   ./src/runall.sh grover           # or just one
```

With `bazel`, run an algorithm by its target label (no `.py` extension):

```
   bazel run //src:arith_classic
```

The available algorithms are:

```
# Algorithms discussed in the book:
   arith_classic     deutsch_jozsa     phase_estimation  simon
   arith_quantum     entanglement_swap phase_kick        simon_general
   bernstein         grover            quantum_walk      solovay_kitaev
   counting          max_cut           shor_classic      subset_sum
   deutsch           order_finding     superdense        supremacy
   swap_test         teleportation     vqe_simple

# Additional algorithms and techniques (2nd edition):
   amplitude_estimation  hamiltonian_encoding  quantum_mean     spectral_decomp
   bell_basis            hhl                   quantum_median   state_prep
   chsh                  hhl_2x2               quantum_pca      state_prep_mottonen
   estimate_pi           inversion_test        sat3             zy_decomp
   euclidean_distance    minimum_finding       schmidt_decomp
   graph_coloring        oracle_synth          purification
   hadamard_test         pauli_rep             qram
```

Run any of them with `python3 -m src.<name>` from the repository root or
`bazel run //src:<name>`.

To test aspects of the sparse implementation:
```
  bazel test //src/libq:all
```

To run the benchmarks:
```
  bazel run //src/benchmarks:larose_benchmark
  bazel run //src/benchmarks:tensor_math
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
*  Colin Zhu, for pointing out coding problems.
*  Kevin Crook, Univ. of CA, Berkeley, for feedback and discussion of the Chinese Remainder Theorem.
*  [Moez A. AbdelGawad](http://eng.staff.alexu.edu.eg/~moez/), Alexandria University, Egypt, for suggesting Windows and SageMath ports.
*  Stefanie Scherzinger, Universitaet Passau, for corrections and suggesting Docker.
*  Abdolhamid Pourghazi and Stefan Klessinger, for providing and maintaining the Dockerfile.
*  Michael Broughton, for help with purification.
*  Mikhail Remnev, for pointing out a .dylib problem in MacOS
*  Andrea Novellini, for fixing a WORKSPACE issue with bazel 7.0.x
*  Pinkman for helping on code quality
*  Pijus Petkevicius for many helpful comments on the book

