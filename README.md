# Quantum Computing for Programmers

This is the open-source repository for the book [Quantum Computing for Programmers](https://www.cambridge.org/us/academic/subjects/computer-science/algorithmics-complexity-computer-algebra-and-computational-g/quantum-computing-programmers?format=HB) by Robert Hundt, Cambridge University Press. The book describes this implementation in great detail, including all the underlying math and derivations. To get started quickly on the Python sources, you may find the [Quickstart Guide](https://github.com/qcc4cp/qcc/blob/main/resources/quickstart.md) helpful.

This project builds vendor-independent infrastructure from the ground up and implements standard algorithms, such as Quantum Teleportation, Quantum Phase estimation (QPE), Grover's Search, Quantum counting, Quantum random walks, VQE, QAOA, Max-Cut, Subset-Sum, Quantum Fourier Transform (QFT), Shor's integer factorization, and Solovay-Kitaev. It also implements high performance quantum simulation and a transpilation technique to compile circuits to other infrastructures, such as Qiskit or Cirq. 

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
*   For **MacOS**, see [README.MacOS.md](resource/README.MacOS.md). 
*   For **Windows** (partially supported), see [README.Windows.md](resources/README.Windows.md). 
*   For interactive **SageMath**, see [README.SageMath.md](resources/README.SageMath.md). 
*   **CentOS** is also supported (see [README.CentOS.md](resources/README.CentOS.md)).


## Run

The main algorithms are all in `src`.
To run individual algorithms via `bazel`, run any of these command lines (note the missing `.py` extensions):

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
*  Colin Zhu, for pointing to coding problems.  
*  Kevin Crook, Univ. of CA, Berkeley, for feedback and discussion of the Chinese Remainder Theorem.  
*  [Moez A. AbdelGawad](http://eng.staff.alexu.edu.eg/~moez/), Alexandria University, Egypt, for suggesting Windows and SageMath ports.  
*  Stefanie Scherzinger, Universitaet Passau, for corrections and suggesting Docker.
*  Abdolhamid Pourghazi and Stefan Klessinger for providing and maintaining the Dockerfile.

