# Planned CPU backend

qcc includes an opt-in CPU backend that captures gates and executes them in
small fused blocks. Native `libxgates` remains the default.

```bash
# Native libxgates backend (default)
python3 order_finding.py

# Planned and fused CPU backend
QCC_BACKEND=planner python3 order_finding.py
```

The planned backend is integrated into qcc. It does not require a separate
`libxgates.py` module earlier on `PYTHONPATH`.

## Execution model

`src/lib/planned_xgates.py` implements the same `apply1` and `applyc` entry
points used by `src/lib/circuit.py`. Instead of modifying the state for each
gate, these functions copy the gate matrix and append an immutable operation
to the current state's pending epoch.

An epoch is materialized before an operation that must observe or replace the
state, including:

- probabilities, amplitudes, phases, dumps, comparisons, and indexed reads;
- measurement and arbitrary dense-unitary application;
- extending or replacing the circuit state; and
- interpreter shutdown.

Materialization executes the pending operations and copies the resulting
vector back into the original qcc `State` object. Later gates begin a new
epoch.

## Block planning

`src/lib/block_planner.py` performs four steps:

1. Compose adjacent operations with identical target and control semantics.
2. Partition the ordered gate stream into blocks using at most a configured
   number of active qubits. Gates are never reordered across dependencies.
3. Execute profitable multi-qubit blocks as matrix operations and small blocks
   as state-vector butterflies.
4. Retain the current tensor-axis layout between matrix blocks, transposing
   back to canonical qcc qubit order only when required.

By default, a matrix block needs at least two gates, operates on two to seven
qubits, and is used only for state vectors with at least eight qubits.

## Fused unitary construction

`src/lib/planned_fusion.py` builds a `2**k` by `2**k` unitary for each selected
`k`-qubit block. It starts from the identity and applies each gate directly to
pairs of unitary rows using qcc's most-significant-bit qubit convention.
Controlled gates update only row pairs whose control bit is set.

The resulting block unitary is applied to all remaining state dimensions in
one BLAS operation:

```text
state tensor -> [2**k, remaining columns]
result = block_unitary @ state_matrix
```

This avoids constructing a full `2**n` operator and amortizes gate dispatch
over every state column.

Construction costs `O(gates * 4**k)`. On the machine measured below that is
about 20 µs per gate for a 7-qubit block, and the cost grows 4x for every
qubit a block adds (see [Block size](#block-size)).

## Configuration

| Environment variable | Default | Meaning |
|---|---:|---|
| `QCC_BACKEND` | `libxgates` | Set to `planner` to enable capture and fusion |
| `QCC_PLANNER_MAX_QUBITS` | `7` | Maximum active qubits in one planned block |
| `QCC_PLANNER_MIN_GATES` | `2` | Minimum gates required for matrix execution |
| `QCC_PLANNER_STATS_FILE` | unset | Write epoch statistics to JSON |

The current minimal integration targets qcc's default 64-bit tensor mode
(`complex64`). It should not be used with `--tensor_width=128` until the
planner is made dtype-generic (see [Correctness and accuracy](#correctness-and-accuracy)).

`VECLIB_MAXIMUM_THREADS` has no measurable effect on the planner, and
`XGATES_PARALLEL` applies to the `libxgates` backend only.

## Performance

### Summary

On order finding with 22 qubits, the planner runs the gate phase 7x faster than
single-threaded `libxgates` (4.26 s against 29.9 s). It is also 1.3x faster than
`libxgates` on 18 threads (5.72 s). Three effects combine:

1. **Fusion.** About 26 gates share one block, so the state is traversed 1,558
   times instead of 20,271.
2. **Matrix hardware.** Each block is one complex64 GEMM in Accelerate, which
   runs at about 1 TFLOP/s on the measured machine; the `libxgates` loop is
   scalar. The blocks perform 4.8x more arithmetic than the gates they replace,
   at about 45x the throughput.
3. **The retained layout.** The axis permutation that turns a block into a GEMM
   is a stable partition and is not undone between blocks. Consecutive blocks
   reuse the same working qubits, so most permutations copy long contiguous runs.
   Restoring canonical order after every block makes the same run 1.6x slower.

The gain depends on the circuit. Circuits whose consecutive gates stay on a few
qubits gain the most: modular arithmetic, oracles, walks. Layered circuits that
touch every qubit in every layer run 1.5x slower than `libxgates`. Building
block unitaries in parallel, and using 9-qubit blocks once that is cheap, makes
the 22-qubit run another 3x faster in a prototype
(see [Parallel matrix generation](#parallel-matrix-generation)).

### Measurement setup

- Apple M5 Pro (18 cores, SME2), 48 GB, macOS. Python 3.14.7, numpy 2.5.3 linked
  to Accelerate. `libxgates` built by `make_libxgates.sh` (`-O3 -ffast-math`).
- Times are for the gate phase: from the first gate until the state is complete.
  For `libxgates` that is when the last `apply1` or `applyc` returns; for the
  planner it is when the epoch has been flushed. Whole-program times also include
  Python start-up and each algorithm's classical work. Order finding's final loop
  over all `2**n` basis states takes far longer than its gates at 22 qubits.
- Planner breakdowns come from `QCC_PLANNER_STATS_FILE`.
- Single runs; repeated runs vary by 10-15%.
- GEMM throughput is specific to Apple Silicon with Accelerate. A machine without
  a matrix unit, for example Linux with OpenBLAS, was not measured.

### Where the time goes

Order finding, `--N=21 --a=11`: 22 qubits, 20,271 gates (3,461 `apply1`,
16,810 `applyc`).

| Configuration | Gate phase | vs. `libxgates`, 1 thread |
|---|---:|---:|
| `libxgates`, 1 thread | 29.9 s | 1.0x |
| `libxgates`, `XGATES_PARALLEL=8` | 6.70 s | 4.5x |
| `libxgates`, `XGATES_PARALLEL=18` | 5.72 s | 5.2x |
| planner, default (7-qubit blocks) | 4.26 s | 7.0x |
| planner, `VECLIB_MAXIMUM_THREADS=1` | 4.22 s | 7.1x |
| planner, canonical order restored after every block | 6.95 s | 4.3x |
| prototype: 9-qubit blocks, unitaries built on 8 threads during the GEMMs | 1.41 s | 21x |

At other sizes:

| Circuit | Qubits | Gates | `libxgates`, 1 thread | `libxgates`, 18 threads | planner |
|---|---:|---:|---:|---:|---:|
| order finding, `--N=15 --a=4` | 18 | 10,297 | 0.95 s | -- | 0.31 s |
| order finding, `--N=21 --a=11` | 22 | 20,271 | 29.9 s | 5.72 s | 4.26 s |
| order finding, `--N=33 --a=5`, first modular multiplication only | 26 | 2,987 | 72.8 s | 10.05 s | 12.2 s |

The planner's 4.26 s at 22 qubits:

| Part | Time | Per unit |
|---|---:|---|
| Block GEMMs | 3.18 s | 779 blocks, about 4.1 ms each |
| Layout permutations and planning | 0.58 s | 779 moves, about 0.74 ms each |
| Building block unitaries | 0.41 s | about 20 µs per gate |
| Gate capture | 0.07 s | about 3 µs per gate |

A block holds 25.7 gates on average. The planner spends about 5.4 ms on it:
GEMM, permutation and construction. The same gates cost about 38 ms in
`libxgates`: 25.7 passes at 2.13 ms per `apply1` and 1.35 ms per `applyc`.

What the planner makes of four circuits, with 7-qubit blocks. A state pass is one
read and one write of the whole state vector. `libxgates` makes one per gate; the
planner makes one per block GEMM and one per layout move.

| Circuit | Qubits | Gates | Blocks | Gates per block | State passes, `libxgates` | State passes, planner | GEMM GFLOP | Butterfly GFLOP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order finding, `--N=15 --a=4` | 18 | 10,297 | 334 | 30.3 | 10,297 | 668 | 89 | 22 |
| order finding, `--N=21 --a=11` | 22 | 20,271 | 779 | 25.7 | 20,271 | 1,558 | 3,342 | 697 |
| order finding, `--N=33 --a=5` | 26 | 35,797 | 1,826 | 19.4 | 35,797 | 3,652 | 125,387 | 19,489 |
| layered random circuit, depth 20 | 22 | 860 | 200 | 4.3 | 860 | 399 | 601 | 38 |

GEMM work is `8 * 2**(n+k)` real flops per block. Butterfly work is 14 flops
per amplitude for `apply1` and 7 for `applyc`, which updates half the amplitude
pairs.

### Why it is fast

#### Fusion: fewer passes over the state

A gate in `libxgates` reads and writes all `2**n` amplitudes. A block reads and
writes them once for its GEMM and once more for the permutation that precedes
it. With 26 gates per block, order finding at 22 qubits makes 1,558 passes
instead of 20,271. Merging adjacent gates on the same qubits (step 1 of the
planner) removes only 1.3% of the gates; nearly all of the reduction comes from
the blocks.

#### Matrix hardware against a scalar loop

The block GEMM is `complex64`, `[2**k, 2**k] @ [2**k, 2**(n-k)]`, through
numpy and Accelerate. Measured at 22 qubits with random operands, best of 5:

| k | GEMM | GFLOP/s | State bandwidth |
|---:|---:|---:|---:|
| 1 | 2.44 ms | 28 | 27 GB/s |
| 2 | 1.82 ms | 74 | 37 GB/s |
| 3 | 1.89 ms | 142 | 36 GB/s |
| 4 | 3.52 ms | 153 | 19 GB/s |
| 5 | 2.86 ms | 375 | 24 GB/s |
| 6 | 3.47 ms | 619 | 19 GB/s |
| 7 | 4.67 ms | 921 | 14 GB/s |
| 8 | 6.67 ms | 1,287 | 10 GB/s |
| 9 | 10.68 ms | 1,608 | 6 GB/s |
| 10 | 18.06 ms | 1,903 | 4 GB/s |

What this shows:

- **Up to 6 qubits a block costs less than two passes.** One pass is the 1.8-2.4 ms
  of the smallest blocks. A 7-qubit block costs about two passes and applies about
  26 gates.
- **The throughput comes from the chip's matrix unit, not from threads.**
  - `VECLIB_MAXIMUM_THREADS=1` gives the same numbers.
  - 1.9 TFLOP/s is far beyond what one core's vector units can do, and the chip
    reports SME2. This is consistent with Accelerate using the matrix unit; it
    was not traced directly.
  - Splitting one block's GEMM by columns over 2 threads is 1.26x faster. 4 to 8
    threads are slower than one, because the unit is shared.
- **`complex128` is much slower.** `zgemm` reaches 229 GFLOP/s at k = 7, against
  921 for `cgemm`.

`libxgates` computes each amplitude pair with scalar code: its binary contains no
vector floating-point instructions. Per amplitude per gate:

| Kernel | 1 thread, 14 to 26 qubits | 18 threads, 22 to 26 qubits |
|---|---:|---:|
| `apply1` | 0.50-0.52 ns (31 GB/s) | 0.07 ns (about 220 GB/s) |
| `applyc` | 0.32-0.35 ns (47-50 GB/s) | 0.05-0.07 ns (230-350 GB/s) |

Single-threaded, the cost per amplitude is the same whether the state fits in L1
or lives in DRAM. So the loop is limited by its instruction stream, not by
memory, and only becomes memory-bound when threaded. At 18 qubits, where the state
fits in cache, threads gain only 1.1-1.4x. Below 16 qubits `libxgates` does not
thread at all.

For order finding at 22 qubits, the planner's GEMMs perform 3,342 GFLOP in 3.18 s,
about 1.05 TFLOP/s. The `libxgates` butterflies perform 697 GFLOP in 29.9 s,
about 23 GFLOP/s. That is 4.8x the arithmetic at about 45x the rate.

#### The retained layout

`LayoutState` keeps the state as an `n`-axis tensor together with the qubit held
by each axis. Before a block it reorders the axes:

```python
selected = [q for q in self.axis_qubits if q in segment.qubits]       # block qubits, current order
remaining = [q for q in self.axis_qubits if q not in segment.qubits]  # the rest, current order
desired = selected + remaining
matrix = np.transpose(self.tensor, permutation).reshape(1 << k, -1)   # the copy happens here
```

The block's qubits become the matrix rows; every other qubit is folded into
contiguous columns. The result stays in this order, and canonical order is
restored only when the epoch ends or a butterfly segment needs it.

Both groups keep their relative order. So the axes that no move touches stay at
the end of the layout and are copied as contiguous runs. In order finding,
consecutive blocks work on the same register qubits, the ancillas and one
control. 653 of the 779 moves at 22 qubits copy runs of 4,096 to 16,384
contiguous elements, at about 0.74 ms per move. In the layered random circuit,
198 of 199 moves copy runs of 1 or 2 elements, at about 8.4 ms per move. For
scale, a scattered permutation of a 22-qubit state measures 2.7 ms on its own.

Restoring canonical order after every block costs 1.76 s more, and the whole run
becomes 6.95 s instead of 4.26 s.

### Block size

`QCC_PLANNER_MAX_QUBITS` on order finding, `--N=21`, 22 qubits:

| Max qubits | Gate phase | Blocks | GEMM | Unitary construction | Permutations and planning |
|---:|---:|---:|---:|---:|---:|
| 3 | 14.43 s | 5,631 | 10.58 s | 0.17 s | 3.59 s |
| 4 | 12.25 s | 3,290 | 9.82 s | 0.16 s | 2.19 s |
| 5 | 7.40 s | 2,095 | 5.71 s | 0.16 s | 1.44 s |
| 6 | 5.70 s | 1,372 | 4.34 s | 0.22 s | 1.05 s |
| 7 (default) | 4.26 s | 779 | 3.18 s | 0.41 s | 0.58 s |
| 8 | 4.84 s | 527 | 3.34 s | 0.98 s | 0.44 s |
| 9 | 4.24 s | 116 | 1.20 s | 2.82 s | 0.14 s |
| 10 | 12.62 s | 63 | 1.12 s | 11.31 s | 0.10 s |

How the trade-off moves:

- **Below 7 qubits**, GEMMs and permutations dominate: there are many blocks, and
  each costs at least one pass.
- **At 9 qubits** the circuit collapses into 116 blocks of about 172 gates, and the
  GEMM time falls to 1.2 s.
- **Unitary construction** grows 4x per qubit and takes over above 8. That is why
  7 is the best default for the planner as written, and why construction is worth
  making faster or parallel.

### Where it is slower

**Layered circuits.** The random circuit used here applies a random `h`, `t`,
`v` or `s` to every qubit in each layer. Pairs of neighbouring qubits then get a
`cx` or `cz`, followed by a second, offset row of `cx`. A block fills up with 7
distinct qubits after about 4 gates, and every move scatters the state:

| Layered random circuit, depth 20 | Qubits | `libxgates`, 1 thread | planner |
|---|---:|---:|---:|
| 860 gates | 22 | 1.53 s | 2.41 s |
| 940 gates | 24 | 6.88 s | 10.44 s |

The block finder takes gates in program order and does not move commuting gates
past each other. That is the limit here.

**Large states against threaded `libxgates`.** With 18 threads `libxgates` is
memory-bound at about 220-350 GB/s. The planner's GEMMs do not scale with threads.
At 26 qubits the threaded native backend is ahead: 10.05 s against 12.2 s for one
modular multiplication.

**Small states and frequent flushes.** Whole runs of the 46 algorithms at their
default sizes: 8 jobs in parallel, so these times are noisier than those above.

| Algorithm | `libxgates` | planner | Ratio | Main cost in the planner |
|---|---:|---:|---:|---|
| hamiltonian_cycle | 4.23 s | 2.32 s | 0.55 | GEMMs, 22-qubit state |
| order_finding | 2.02 s | 1.25 s | 0.62 | unitary construction, 18 qubits |
| graph_coloring | 2.83 s | 2.17 s | 0.77 | GEMMs |
| quantum_walk | 2.06 s | 1.58 s | 0.77 | unitary construction, 25,694 gates |
| vqe_simple | 5.71 s | 6.85 s | 1.20 | 41,500 epochs of at most 2 qubits |
| arith_quantum | 1.22 s | 1.55 s | 1.27 | unitary construction |
| state_prep_mottonen | 0.91 s | 1.16 s | 1.27 | unitary construction, 10 qubits |
| quantum_mean | 0.83 s | 1.17 s | 1.41 | unitary construction, 14 qubits |

The other 38 algorithms are within 0.86 to 1.13.

When a state has few qubits, every gate is cheap in `libxgates` (5-8 µs at 14
qubits). The planner's construction cost of about 20 µs per gate then
dominates, and so does planning a new epoch every time the program reads the
state.

## Correctness and accuracy

**Validation.** The integration changes no algorithm source. All 46 algorithms
that `src/runall.sh` runs were run with both backends. A `sitecustomize.py` on
`PYTHONPATH` seeds Python's and numpy's random generators identically for every
run, so outputs can be compared directly:

- 42 outputs are byte-identical.
- `hhl.py` and `hhl_2x2.py` differ only in `-0.0` against `0.0`.
- `sat3.py` differs only in the last float32 digits: `0.56249994` against `0.5624998`.
- `supremacy.py` differs only in its timing lines.

**Accuracy.** Measured on order finding with `--N=15` (18 qubits). The reference is
`libxgates` with `--tensor_width=128`:

| Backend | Arithmetic | Max abs error | L2 error |
|---|---|---:|---:|
| `libxgates` | complex64 | 2.19e-5 | 4.36e-5 |
| planner | complex64 | 2.08e-5 | 4.18e-5 |
| planner, `--tensor_width=128` | complex64 into complex128 storage | 3.02e-6 | 5.14e-6 |

Fusion costs no accuracy: the planner matches `libxgates` at the same precision.
With `--tensor_width=128`, however, the planner returns a complex128 state computed
in complex64. Nothing refuses that combination, so its results silently carry
complex64 error.

**Reads that bypass the flush.** An epoch is flushed only by:
- the hooked `State` methods: `prob`, `ampl`, `maxprob`, `phase`, `density`,
  `adjoint`, `normalize`, `diff`, `dump`, and indexing (which also covers iteration);
- `Tensor.is_close` and `np.allclose`;
- `qc.unitary`, `qc.measure_bit` and `_tprod`.

Any other numpy access reads the array as it was before the pending gates:

```python
qc = circuit.qc('stale')
qc.reg(2, 0)
qc.h(0)
qc.cx(0, 1)
np.abs(qc.psi) ** 2            # planner: [1, 0, 0, 0]    libxgates: [0.5, 0, 0, 0.5]
(qc.psi > 0.1).nonzero()[0]    # planner: [0]             libxgates: [0, 3]
qc.psi.prob(1, 1)              # 0.5 on both: prob() flushes first
```

An arithmetic expression such as `qc.psi * phase` is stale too: the product is
formed before `np.allclose` can flush it. Three algorithms read the state this way
and are correct only because a flush happens just before:
- `hhl.py:36` and `hhl_2x2.py:98` follow a `measure_bit`;
- `state_prep_mottonen.py:128` follows the indexed read `qc.psi[0]`.

Code that reads the state through numpy should call `planned_xgates.flush(qc.psi)`
or a hooked method first.

## Parallel matrix generation

A block's unitary depends only on its gates and on the order of its qubits, never
on the state. The planner can therefore construct every unitary before the state
is touched, or while it is being touched. At the default 7-qubit blocks this buys
almost nothing, because construction is under 10% of the flush. What it buys is
cheap construction, which makes 9-qubit blocks the better choice. In a prototype
outside the planner, order finding at 22 qubits flushes in 1.34 s, against 4.36 s
for the committed planner in the same session. Its states match the committed
planner to within 4e-7.

### What the prototype does

1. **Plan the epoch, then replay the layout.** Each block's local qubit order is
   decided by `LayoutState` from the axis order the previous block left behind. That
   order depends only on the plan, so it can be computed up front:

   ```python
   axis = list(range(nbits))
   for segment in segments:                      # all matrix segments, in order
     selected = [q for q in axis if q in segment.qubits]
     desired = selected + [q for q in axis if q not in segment.qubits]
     permutations.append([axis.index(q) for q in desired])
     local = {q: i for i, q in enumerate(selected)}
     blocks.append((len(selected), [(op.gate, local[op.target],
                                     None if op.control is None else local[op.control])
                                    for op in segment.operations]))
     axis = desired
   ```

2. **Construct each unitary with `libxgates`.** A `2**k` by `2**k` matrix in
   row-major order is a `2k`-qubit state whose first `k` qubits are its row bits.
   Applying a gate to row qubit `t` of that state computes `U <- G_t U`, which is
   exactly what `build_block_unitary` computes:

   ```python
   def build_with_xgates(k, operations):
     u = np.eye(1 << k, dtype=np.complex64).reshape(-1)    # a 2k-qubit state
     for gate, target, control in operations:
       g = np.ascontiguousarray(gate, dtype=np.complex64).reshape(4)
       if control is None:
         xgates.apply1(u, g, 2 * k, target, 64)
       else:
         xgates.applyc(u, g, 2 * k, control, target, 64)
     return u.reshape(1 << k, 1 << k)
   ```

   This is 2.6x faster than the numpy builder for 7-qubit blocks (0.14 s against
   0.37 s for the whole run) and 1.4x faster for 9-qubit blocks (1.89 s against
   2.73 s). It agrees with it to within 4e-7.

3. **Construct on a thread pool and consume the results in block order.** The main
   thread runs each block's permutation and GEMM as soon as that block's unitary is
   ready:

   ```python
   with ThreadPoolExecutor(8) as pool:
     futures = [pool.submit(build_with_xgates, k, ops) for k, ops in blocks]
     for (k, _), permutation, future in zip(blocks, permutations, futures):
       matrix = np.transpose(tensor, permutation).reshape(1 << k, -1)
       tensor = (future.result() @ matrix).reshape((2,) * nbits)
   ```

   The threads run in parallel only if `libxgates` releases the GIL while its kernels
   run. In the prototype this is two additions to `src/lib/xgates.cc`, in
   `apply1_python` and `applyc_python`:

   ```cpp
   Py_BEGIN_ALLOW_THREADS
   apply1<cmplx_type>(psi, gate, nbits, tgt);
   Py_END_ALLOW_THREADS
   ```

   The kernels touch only the two arrays, which stay referenced throughout the call.

### Measured

Order finding, `--N=21`, 22 qubits; times for the whole flush:

| Max qubits | As committed | All unitaries prebuilt | Built on 8 threads during the GEMMs |
|---:|---:|---:|---:|
| 7 | 4.36 s | 3.72-4.31 s | 3.59-3.77 s |
| 9 | 4.11 s | 1.30-1.39 s | 1.34 s |
| 10 | 12.34 s | 1.23 s | 1.98 s (8 processes) |

The "prebuilt" column is the floor that faster construction can reach: GEMMs plus
permutations. With 9-qubit blocks, construction on 8 threads hides behind the
GEMMs and the flush reaches that floor.

Construction alone, 9-qubit blocks:

| Builder | 1 thread | 4 threads | 8 threads | 16 threads |
|---|---:|---:|---:|---:|
| numpy (`build_block_unitary`) | 2.73 s | 1.68 s | 2.82 s | -- |
| `libxgates`, GIL released | 1.96 s | 0.52 s | 0.30 s | 0.20 s |

### What does not work

- **Python threads with the numpy builder.** The builder is many small numpy
  operations that hold the GIL. For 7-qubit blocks, 8 threads take 1.85 s against
  0.37 s on one thread.
- **Bigger blocks than 9 qubits.** At 10 qubits construction needs 7.6 s of CPU
  (with `libxgates`), and 8 workers can no longer hide it: 1.98 s.
- **More threads on the GEMM.** See [Matrix hardware](#matrix-hardware-against-a-scalar-loop).
- Worker processes work, but are second best. 8 spawned processes construct the
  9-qubit unitaries in 0.32 s, and the flush overlapped with them takes 1.44 s. But a
  pool costs 0.17-0.29 s to start, and every unitary is copied between processes.

### Notes for an implementation

- Keep the numpy builder as the fallback when `libxgates` cannot be imported. The
  planner itself does not require the library.
- Construct only a few blocks ahead of the GEMM. Fully prebuilt, the 116 9-qubit
  unitaries hold 232 MB.
- Leave small epochs on the current serial path, since planning and dispatch
  overheads dominate them.
- The layered random circuits were not re-measured with 9-qubit blocks.

## Reproducing

Build the native library and make it importable:

```bash
./make_libxgates.sh
export PYTHONPATH=$PWD/src/lib
```

Whole runs, with the planner's own breakdown:

```bash
python3 -m src.order_finding --N=21 --a=11
QCC_BACKEND=planner QCC_PLANNER_STATS_FILE=/tmp/stats.json python3 -m src.order_finding --N=21 --a=11
QCC_BACKEND=planner QCC_PLANNER_MAX_QUBITS=9 QCC_PLANNER_STATS_FILE=/tmp/stats9.json \
  python3 -m src.order_finding --N=21 --a=11
```

The statistics file holds one entry per epoch. Among its fields are `capture_ms`,
`flush_wall_ms`, `unitary_build_ms`, `matrix_apply_ms`, `total_execute_ms`,
`segments` and `layout_transitions`.

To time only the gate phase on either backend, build the circuit through `qc` and
stop the clock after the flush. This is `order_finding.main` without its final loop
over all basis states:

```python
import time

from src import order_finding as of
from src.lib import circuit, planned_xgates

nbits = number.bit_length()
qc = circuit.qc('order_finding')
aux, up, down = qc.reg(nbits + 2), qc.reg(nbits * 2), qc.reg(nbits)
start = time.perf_counter()
qc.h(up)
qc.x(down[0])
for i in range(nbits * 2):
  of.cmultmodn(qc, up[i], down, aux, int(a ** (2**i)), number, nbits)
of.inverse_qft(qc, up, 2 * nbits, with_swaps=True)
planned_xgates.flush(qc.psi)          # QCC_BACKEND=planner only
seconds = time.perf_counter() - start
```

Here `number` and `a` are the values of `--N` and `--a`.

Block GEMM throughput for one block size:

```python
import timeit

import numpy as np

state = np.ones(1 << n, dtype=np.complex64)
u = np.ones((1 << k, 1 << k), dtype=np.complex64)
m = state.reshape(1 << k, -1)
seconds = min(timeit.repeat(lambda: u @ m, number=1, repeat=5))
gflops = 8 * 2 ** (n + k) / seconds / 1e9
```

## Implementation map

- `src/lib/circuit.py`: backend selection
- `src/lib/planned_xgates.py`: capture, flush barriers, and statistics
- `src/lib/block_planner.py`: block formation and retained-layout executor
- `src/lib/planned_fusion.py`: butterfly-based unitary construction
