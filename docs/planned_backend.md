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

## Configuration

| Environment variable | Default | Meaning |
|---|---:|---|
| `QCC_BACKEND` | `libxgates` | Set to `planner` to enable capture and fusion |
| `QCC_PLANNER_MAX_QUBITS` | `7` | Maximum active qubits in one planned block |
| `QCC_PLANNER_MIN_GATES` | `2` | Minimum gates required for matrix execution |
| `QCC_PLANNER_STATS_FILE` | unset | Write epoch statistics to JSON |

The current minimal integration targets qcc's default 64-bit tensor mode
(`complex64`). It should not be used with `--tensor_width=128` until the
planner is made dtype-generic.

## Implementation map

- `src/lib/circuit.py`: backend selection
- `src/lib/planned_xgates.py`: capture, flush barriers, and statistics
- `src/lib/block_planner.py`: block formation and retained-layout executor
- `src/lib/planned_fusion.py`: butterfly-based unitary construction

The integration changes no algorithm source. Deterministic `runall.sh`
validation completed all 49 algorithms with both backends. Forty-five output
sections were byte-identical; the remaining semantic output matched after
excluding backend timing and bounded FP32 formatting differences (`-0.0`,
`1.4e-7`, and one `0.001` display-rounding boundary).
