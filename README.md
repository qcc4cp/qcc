# Quantum Computing for Programmers

This is the open-source repository for the book [Quantum Computing for Programmers, 2nd Edition](http://www.cambridge.org/9781009548533) by Robert Hundt, Cambridge University Press. The book describes the implementations in this reposoitory in great detail, including all the underlying math and derivations. Note, however, that this code base is evolving.

To get started quickly on the Python sources, you may find the [Quickstart Guide](https://github.com/qcc4cp/qcc/blob/main/resources/quickstart.md) helpful.

This project builds vendor-independent infrastructure from the ground up and implements standard algorithms, such as Quantum Teleportation, Superdense coding, Deutsch-Jozsa, Bernstein-Vazirani, Quantum Phase estimation (QPE), Grover's Search (with application to Quantum counting, amplitude estimation, Mean and Median estimation, 3SAT, Graph Coloring, and Minimum finding), Quantum random walks, VQE, Max-Cut, Subset-Sum, Quantum Fourier Transform (QFT), Shor's integer factorization, Solovay-Kitaev, Principal Component Analysis, and a few more. It also implements high performance quantum simulation and a transpilation technique to compile circuits to other infrastructures, such as Qiskit or Cirq.

The code is organized as follows:
*  `src` is the main source directory. All algorithms are in this directory.
*  `src/lib` contains the library functions for tensors, states, operators, circuits, and so on, as well as their corresponding tests. All algorithms depend on these library functions.
*  `src/libq` contains the sparse implementation.
*  `src/benchmarks` contains a few benchmarks, as they are mentioned in the book.
*  `resources` contains additional text, sections and chapters.
*  `errata` contains the errata for the book - corrections and clarifications.
*  `external` contains the *.BUILD files to point `bazel` to `python` and `numpy`.

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
To run individual algorithms, run any of these command lines. To run a Python file with `bazel`, run `bazel run python-file` but omit the `.py` extensions (for example `bazel run arith_classic`. PYTHONPATH must point to the root directory:

```
# Algorithms discussed in the book:
   python3 arith_classic
   python3 arith_quantum
   python3 bernstein
   python3 counting
   python3 deutsch
   python3 deutsch_jozsa
   python3 entanglement_swap
   python3 grover
   python3 max_cut
   python3 order_finding
   python3 phase_estimation
   python3 phase_kick
   python3 quantum_walk
   python3 shor_classic
   python3 simon
   python3 simon_general
   python3 solovay_kitaev
   python3 subset_sum
   python3 superdense
   python3 supremacy
   python3 swap_test
   python3 teleportation
   python3 vqe_simple

# Additional algorithms and techniques, to clarify, and
# for the 2nd edition of the book (which will come out
# end of 2025):
   python3 amplitude_estimation
   python3 bell_basis
   python3 chsh
   python3 estimate_pi
   python3 euclidean_distance
   python3 graph_coloring
   python3 hadamard_test
   python3 hamiltonian_encoding
   python3 hhl
   python3 hhl_2x2
   python3 inversion_test
   python3 minimum_finding
   python3 oracle_synth
   python3 pauli_rep
   python3 purification
   python3 qram
   python3 quantum_mean
   python3 quantum_median
   python3 quantum_pca
   python3 sat3
   python3 schmidt_decomp
   python3 spectral_decomp
   python3 state_prep
   python3 state_prep_mottonen
   python3 zy_decomp

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

