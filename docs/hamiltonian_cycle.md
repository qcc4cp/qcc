# Finding Hamiltonian Cycles with Grover's Algorithm

This document explains, step by step, how the program `src/hamiltonian_cycle.py`
in the qcc repository finds Hamiltonian cycles with a quantum computer. It is
written for readers who know some programming and some linear algebra
(vectors, matrices, complex numbers), but who are new to quantum computing.

Every step comes with a short piece of Python code that you can run. The code
uses the small quantum simulator that ships with qcc. The printed outputs in
this document were produced by running exactly this code.

## Contents

1. [The problem: Hamiltonian cycles](#1-the-problem-hamiltonian-cycles)
2. [The quantum computing you need](#2-the-quantum-computing-you-need)
3. [Grover's algorithm in one page](#3-grovers-algorithm-in-one-page)
4. [Step 1: Encode the problem in qubits](#4-step-1-encode-the-problem-in-qubits)
5. [Step 2: A classical checker](#5-step-2-a-classical-checker)
6. [Step 3: A test that a circuit can compute](#6-step-3-a-test-that-a-circuit-can-compute)
7. [Step 4: Reserve the qubits](#7-step-4-reserve-the-qubits)
8. [Step 5: Try all edge subsets at once](#8-step-5-try-all-edge-subsets-at-once)
9. [Step 6: The degree check circuit](#9-step-6-the-degree-check-circuit)
10. [Step 7: Turning a yes into a minus sign](#10-step-7-turning-a-yes-into-a-minus-sign)
11. [Step 8: The complete oracle](#11-step-8-the-complete-oracle)
12. [Step 9: The diffuser](#12-step-9-the-diffuser)
13. [Step 10: How many times to repeat](#13-step-10-how-many-times-to-repeat)
14. [Step 11: Read and check the answer](#14-step-11-read-and-check-the-answer)
15. [The whole program](#15-the-whole-program)
16. [Limits of this approach](#16-limits-of-this-approach)
17. [Glossary](#17-glossary)

## How to run the code

From the root of the qcc repository, with Python 3 and the packages `absl-py`,
`numpy`, and `scipy` installed:

```sh
PYTHONPATH=.:src/lib python3 src/hamiltonian_cycle.py
```

`src/lib` is on the path so that Python finds `libxgates.so`, a fast C++
implementation of the gates. Without it, the program still works, but it is
much slower.

To try the examples in this document, start Python the same way
(`PYTHONPATH=.:src/lib python3`) and first run this setup code:

```python
import itertools
import math

import numpy as np

from src import hamiltonian_cycle as hc
from src.lib import circuit, helper, ops
```

---

## 1. The problem: Hamiltonian cycles

A **graph** is a set of points, called **vertices**, connected by lines, called
**edges**. We number the vertices 0, 1, 2, and so on, and write an edge as a
pair of vertices, for example `(0, 1)`.

A **Hamiltonian cycle** is a round trip through the graph that visits every
vertex exactly once and ends where it started. It may only travel along edges.

Here is a square with one diagonal (the graph `rect+diag` in the program):

```text
  0 ----- 1
  | \     |
  |   \   |
  |     \ |
  3 ----- 2
```

The edges are `(0,1)`, `(1,2)`, `(2,3)`, `(3,0)` and `(0,2)`. The trip
0 → 1 → 2 → 3 → 0 is a Hamiltonian cycle. It visits all four vertices once and
returns to 0. It does not use the diagonal.

Now look at a triangle with an extra vertex attached to it:

```text
  0 ----- 1
   \     /
    \   /
      2
      |
      3
```

Vertex 3 has only one edge. A trip that enters vertex 3 cannot leave it
without using the same edge again, so this graph has **no** Hamiltonian cycle.

**Why this is hard.** For small graphs you can find a cycle by hand. For large
graphs, nobody knows a fast method. The problem is *NP-complete*, which means
that all known exact algorithms take time that grows exponentially with the
size of the graph in the worst case. The simplest method is to try every
possibility. Grover's algorithm, which we use here, is a quantum way of trying
every possibility that needs fewer steps (about the square root of the number
of possibilities). It does not make the problem easy, but it is a clean
example of how a quantum search works.

---

## 2. The quantum computing you need

This section covers only the ideas that the program uses.

### Qubits and amplitudes

A classical bit is either 0 or 1. A **qubit** is described by two complex
numbers, called **amplitudes**, one for 0 and one for 1. We write its state as

```text
|ψ⟩ = a|0⟩ + b|1⟩      with |a|² + |b|² = 1
```

The symbols `|0⟩` and `|1⟩` (read "ket 0" and "ket 1") are the two
**basis states**. As vectors, `|0⟩ = (1, 0)` and `|1⟩ = (0, 1)`, so the state
above is just the vector `(a, b)`.

When you **measure** the qubit, you get 0 with probability `|a|²` and 1 with
probability `|b|²`. After the measurement, the qubit is in the state you
measured. You cannot read `a` and `b` directly. This is important later: a
quantum algorithm has to make the right answer *likely*, because one
measurement is all you get from a run.

### Many qubits

With n qubits there are 2ⁿ basis states, one for each string of n bits, and
the state is a vector of 2ⁿ amplitudes. For two qubits:

```text
|ψ⟩ = a₀₀|00⟩ + a₀₁|01⟩ + a₁₀|10⟩ + a₁₁|11⟩
```

In qcc, qubit 0 is the leftmost bit. The amplitude of a bit string is stored
at the index you get when you read the bit string as a binary number. For
example, the amplitude of `|110⟩` is at index 6.

This is where the power and the cost of quantum computing come from. A
quantum computer with 50 qubits holds 2⁵⁰ amplitudes, and one gate changes
all of them at once. A classical simulator, like the one in qcc, has to store
and update all of these numbers, which is why we can only simulate small
examples.

### Gates

A **gate** changes the state. Mathematically, a gate is a unitary matrix that
is multiplied with the state vector. The ones we need:

| Gate | Effect on basis states | Notes |
|---|---|---|
| X (NOT) | `|0⟩ → |1⟩`, `|1⟩ → |0⟩` | The quantum version of NOT. |
| H (Hadamard) | `|0⟩ → (|0⟩ + |1⟩)/√2`, `|1⟩ → (|0⟩ − |1⟩)/√2` | Creates superpositions. Applying it twice does nothing. |
| CX (CNOT) | Applies X to the target qubit if the control qubit is `|1⟩` | A controlled gate. |
| Multi-controlled X | Applies X to the target if **all** control qubits are `|1⟩` | With two controls, this is the Toffoli gate. |

The two states that H produces have their own names:

```text
|+⟩ = (|0⟩ + |1⟩)/√2
|−⟩ = (|0⟩ − |1⟩)/√2
```

A **controlled-by-0** gate fires when its control qubit is `|0⟩` instead of
`|1⟩`. You build one by applying X to the control before and after the
controlled gate. In qcc, you mark a control as controlled-by-0 by putting its
index in a list: `[3]` instead of `3`.

### Superposition and interference

A state where several basis states have non-zero amplitudes is a
**superposition**. Since amplitudes can be negative (or complex), they can
cancel or reinforce each other when gates combine them. This is called
**interference**. Grover's algorithm uses interference to make the amplitude
of the right answer large and all the others small.

### Every gate can be undone

Gates are unitary matrices, so every gate has an inverse. Apply a sequence of
gates, then the inverse gates in reverse order, and you are back where you
started. This is called **uncomputation**, and the program relies on it.

---

## 3. Grover's algorithm in one page

Grover's algorithm searches a list of N items for the M items that pass a test.
A classical search checks items one at a time and needs about N/M checks.
Grover's algorithm needs only about √(N/M) "rounds".

It works like this:

1. **Start with all items at once.** Put the search register into an equal
   superposition of all N basis states. Every amplitude is 1/√N.
2. **Repeat a Grover iteration k times.** Each iteration has two parts:
   - The **oracle** flips the sign of the amplitude of every item that passes
     the test. The probabilities don't change yet, since `(−a)² = a²`.
   - The **diffuser** reflects all amplitudes about their mean value.
     Amplitudes below the mean move above it and the other way around. The
     marked items now have negative amplitudes, far below the mean, so they
     grow, and all the others shrink a little.
3. **Measure.** With the right k, the result is very likely one of the
   marked items.

There is a nice picture for this. Let θ be the angle with sin θ = √(M/N).
After k iterations, the probability of measuring a marked item is

```text
P(k) = sin²((2k + 1) θ)
```

Each iteration turns the state by an angle of 2θ toward the marked items. We
want (2k + 1)θ to be close to 90°, which gives about

```text
k = (π/4) · √(N/M)
```

Too few iterations and we have not turned far enough. Too many and we turn
past the target, and the probability goes down again. We will see both in
step 10.

The only part of Grover's algorithm that depends on the problem is the
oracle. Most of this document is about how to build an oracle for
Hamiltonian cycles.

---

## 4. Step 1: Encode the problem in qubits

Grover's algorithm searches over bit strings, so we have to describe candidate
answers as bit strings. The program uses **one qubit per edge**:

- `|1⟩` means "this edge is part of the trip".
- `|0⟩` means "this edge is not part of the trip".

A bit string then describes a *subset of the edges*. For a graph with |E|
edges there are N = 2^|E| subsets, and the search runs over all of them.

We use the triangle as the running example. It has three vertices and three
edges:

```text
      0
     / \
 e0 /   \ e2
   /     \
  1 ----- 2
     e1

e0 = (0, 1),  e1 = (1, 2),  e2 = (2, 0)
```

The bit string `110` means "edges e0 and e1, but not e2", which is the path
0 – 1 – 2. That is not a cycle. The bit string `111` means "all three edges",
which is the cycle 0 → 1 → 2 → 0. For the triangle, N = 2³ = 8, and exactly
one of the 8 subsets (`111`, index 7) is a Hamiltonian cycle.

In the code, a graph is a list of edges. The constructor rejects edges that
can't be right: vertex numbers outside the graph, edges from a vertex to
itself, and the same edge listed twice. Each of these would make the circuit
count edges incorrectly.

```python
class Graph:
  """Hold a graph definition."""

  def __init__(self, num_vertices: int, expected: bool, desc: str,
               edges: List[Tuple[int, int]]):
    for fr, to in edges:
      if not (0 <= fr < num_vertices and 0 <= to < num_vertices) or fr == to:
        raise ValueError(f'{desc}: invalid edge ({fr}, {to}).')
    if len({frozenset(edge) for edge in edges}) != len(edges):
      raise ValueError(f'{desc}: duplicate edges.')

    self.num = num_vertices
    self.edges = edges
    self.desc = desc
    self.expected = expected

  def incident(self, v: int) -> List[int]:
    """Return the indices of the edges of vertex v."""

    return [idx for idx, edge in enumerate(self.edges) if v in edge]
```

`expected` records whether we know the graph to have a Hamiltonian cycle. The
program uses it as a test. `incident(v)` lists the edges that touch vertex v,
by their index. We will need it to count the selected edges of each vertex.

```python
g = hc.Graph(3, True, 'triangle', [(0, 1), (1, 2), (2, 0)])
print(g.incident(0))
```

```text
[0, 2]
```

Vertex 0 touches edge 0, `(0, 1)`, and edge 2, `(2, 0)`.

**Why edges and not vertex orders?** You could also encode a trip as an
ordering of the vertices, for example "first vertex 2, then vertex 0, then
...". That needs about n · log₂(n) qubits for the positions, and the oracle
has to check that the ordering is a valid permutation. One qubit per edge is
simpler, and for the small graphs in this example it needs fewer qubits.

---

## 5. Step 2: A classical checker

Before building anything quantum, we write an ordinary Python function that
decides whether a subset of edges is a Hamiltonian cycle. We will use it to
check the results of the quantum search, and to count the solutions, which we
need to choose the number of Grover iterations.

A subset of edges is a Hamiltonian cycle if and only if:

- **(a)** every vertex has exactly two selected edges (one to arrive, one to
  leave), and
- **(b)** the selected edges are connected, that is, they form one single loop.

```python
  def cycle(self, bits: Sequence[int]) -> Optional[List[int]]:
    """Return the vertices of the cycle, or None if bits is not a cycle."""

    selected = [edge for edge, bit in zip(self.edges, bits) if bit]
    if len(selected) != self.num:
      return None
    neighbors = [[] for _ in range(self.num)]
    for fr, to in selected:
      neighbors[fr].append(to)
      neighbors[to].append(fr)
    if any(len(n) != 2 for n in neighbors):
      return None

    # Walk along the cycle that contains vertex 0. It has to visit all
    # vertices before it returns to 0.
    path = [0]
    prev, cur = 0, neighbors[0][0]
    while cur != 0:
      path.append(cur)
      a, b = neighbors[cur]
      prev, cur = cur, (a if b == prev else b)
    return path if len(path) == self.num else None
```

The function works in three parts:

1. It collects the selected edges. A Hamiltonian cycle through n vertices
   has exactly n edges, so any other count fails right away.
2. It builds a neighbor list for each vertex and checks condition (a).
3. It checks condition (b) by walking along the loop that starts at vertex 0.
   At each vertex, it continues to the neighbor it did not come from. If the
   walk returns to 0 after visiting all n vertices, there is one loop and the
   function returns the vertices in the order of the trip.

`find_solutions` tries all 2^|E| subsets and returns those that pass, as
numbers. `helper.val2bits(7, 3)` turns the number 7 into the bit list
`[1, 1, 1]`.

```python
def find_solutions(g: Graph) -> List[int]:
  """Classically find all Hamiltonian cycles, as edge subsets (numbers)."""

  nbits = len(g.edges)
  return [val for val in range(2**nbits)
          if g.cycle(helper.val2bits(val, nbits))]
```

```python
print(g.cycle([1, 1, 1]))
print(g.cycle([1, 1, 0]))
print(hc.find_solutions(g))
```

```text
[0, 1, 2]
None
[7]
```

This classical brute force is exactly the work that Grover's algorithm speeds
up. For the small graphs here, it is instant.

---

## 6. Step 3: A test that a circuit can compute

The oracle has to compute the test "is this a Hamiltonian cycle?" with
quantum gates. Condition (a), "every vertex has exactly two selected edges",
is easy to compute with gates. Condition (b), "the edges are connected", is
much harder.

For small graphs, there is a way out. If every vertex has exactly two selected
edges, the selected edges always form one or more separate loops that together
pass through every vertex. Each loop needs at least 3 vertices (a loop through
2 vertices would need the same edge twice). So two separate loops need at
least 3 + 3 = 6 vertices. **For graphs with fewer than 6 vertices, condition
(a) alone already means a Hamiltonian cycle**, and the oracle only needs to
check (a).

With 6 vertices, this fails. Here are two triangles that don't touch:

```text
  0 --- 1        3 --- 4
   \   /          \   /
     2              5
```

Every vertex has exactly two edges, but there is no single trip through all
six vertices. The program refuses such graphs:

```python
  if g.num >= 6:
    raise ValueError('The oracle requires graphs with fewer than 6 vertices.')
```

Larger graphs would need an additional connectivity check in the oracle. Step
16 comes back to this.

---

## 7. Step 4: Reserve the qubits

The circuit needs four groups of qubits, called **registers**:

| Register | Size | Purpose |
|---|---|---|
| `e` | one per edge | The search register. Each basis state is a subset of edges. |
| `deg` | one per vertex | Work qubits. `deg[v]` becomes `|1⟩` if vertex v has exactly two selected edges. |
| `chk` | 1 | The qubit that turns the oracle's "yes" into a minus sign (step 7). |
| `aux` | max(edges, vertices) − 1 | Scratch qubits that the multi-controlled gates need (step 6). |

```python
  nedges = len(g.edges)
  qc = circuit.qc('hamiltonian cycle')
  e = qc.reg(nedges, 0)
  deg = qc.reg(g.num, 0)
  chk = qc.reg(1, 0)[0]
  # The multi-controlled gates have at most max(nedges, g.num) controls.
  aux = qc.reg(max(nedges, g.num) - 1, 0)
```

`circuit.qc` creates an empty circuit together with its state vector.
`qc.reg(size, 0)` adds `size` new qubits, all in `|0⟩`, and returns their
indices. The edge register comes first, so the edge qubits are the leftmost
bits of every basis state. This makes it easy to read the result at the end.

For the triangle, this gives 3 + 3 + 1 + 2 = 9 qubits, so the simulator
tracks 2⁹ = 512 amplitudes. Only 8 of the possible bit patterns matter for
the answer (the edge register). The other qubits are tools to compute the
oracle, and they are returned to their starting state after each use.

---

## 8. Step 5: Try all edge subsets at once

Applying H to every qubit of the edge register gives the equal superposition
of all subsets. For three qubits:

```text
H|0⟩ ⊗ H|0⟩ ⊗ H|0⟩ = (1/√8) (|000⟩ + |001⟩ + |010⟩ + ... + |111⟩)
```

Each of the 8 subsets has amplitude 1/√8 ≈ 0.354, and therefore probability
1/8. In the code, this is a single line, since `qc.h` accepts a whole register:

```python
qc = circuit.qc('demo')
e = qc.reg(3, 0)
qc.h(e)
print(np.round(np.real(np.asarray(qc.psi)), 3))
```

```text
[0.354 0.354 0.354 0.354 0.354 0.354 0.354 0.354]
```

`qc.psi` is the state vector. `np.asarray` turns it into a numpy array, and
`np.real` drops the imaginary parts, which are all zero here.

If we measured now, we would get a random subset, and the chance that it is
the cycle would be 1/8. The rest of the algorithm raises this chance.

---

## 9. Step 6: The degree check circuit

For each vertex v, we want to set `deg[v]` to `|1⟩` if exactly two of v's
edges are selected, and leave it at `|0⟩` otherwise.

### The idea

Suppose vertex v has edges e₁, e₂, ..., e_k. "Exactly two are selected" is
the same as: **for some pair (i, j), edges eᵢ and eⱼ are selected and all
other edges of v are not.** We can test one pair with one multi-controlled X
gate:

- controls on eᵢ and eⱼ that fire on `|1⟩`,
- controls on all other edges of v that fire on `|0⟩` (controlled-by-0),
- target `deg[v]`.

We apply one such gate for every pair. If exactly two edges are selected,
exactly one pair matches, and `deg[v]` flips once, from `|0⟩` to `|1⟩`. If
fewer or more than two edges are selected, no pair matches, because each gate
requires *all other* edges to be `|0⟩`. Then `deg[v]` stays `|0⟩`.

The "at most one pair matches" part is important. A gate that flips `deg[v]`
twice would flip it back. Since no two pairs can match at the same time, this
cannot happen.

### The code

```python
def check_degrees(qc: circuit.qc, g: Graph, e, deg, aux):
  """Flip deg[v] if exactly two of the edges of vertex v are selected."""

  for v in range(g.num):
    inc = g.incident(v)
    for i, j in itertools.combinations(inc, 2):
      # Control by |1> for edges i and j, and by |0> for all other edges.
      ctl = [e[k] if k in (i, j) else [e[k]] for k in inc]
      qc.multi_control(ctl, deg[v], aux, ops.PauliX(), f'degree({v})')
```

`itertools.combinations(inc, 2)` produces all pairs of v's edges. The list
`ctl` holds the controls. An index on its own, like `e[k]`, fires on `|1⟩`.
An index in brackets, like `[e[k]]`, fires on `|0⟩`.

Let's look at vertex 0 of the `rect+diag` graph from section 1. It has the
edges with index 0, 3 and 4, which gives three pairs:

```python
g2 = hc.Graph(4, True, 'rect+diag', [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
inc = g2.incident(0)
print(inc)
for i, j in itertools.combinations(inc, 2):
  print((i, j), [k if k in (i, j) else [k] for k in inc])
```

```text
[0, 3, 4]
(0, 3) [0, 3, [4]]
(0, 4) [0, [3], 4]
(3, 4) [[0], 3, 4]
```

The first line reads: "flip `deg[0]` if edges 0 and 3 are selected and edge 4
is not". (In the circuit, the numbers are qubit indices. Here they are the
same as the edge indices, because the edge register starts at qubit 0.)

A vertex with fewer than two edges gets no gates at all, so its `deg` qubit
always stays `|0⟩`. That is correct: such a vertex can't be on a cycle.

### Trying it on single subsets

We can test the degree check on one subset at a time, without any
superposition. `qc.reg(3, bits)` starts the edge register in the given bit
string. `maxprob()` returns the most likely basis state, which is the only
one here.

```python
for bits in ([1, 1, 1], [1, 1, 0], [0, 0, 0]):
  qc = circuit.qc('demo')
  e = qc.reg(3, bits)
  deg = qc.reg(3, 0)
  aux = qc.reg(2, 0)
  hc.check_degrees(qc, g, e, deg, aux)
  maxbits, p = qc.psi.maxprob()
  print(bits, '-> deg', maxbits[3:6], 'aux', maxbits[6:], round(p, 3))
```

```text
[1, 1, 1] -> deg [1, 1, 1] aux [0, 0] 1.0
[1, 1, 0] -> deg [0, 1, 0] aux [0, 0] 1.0
[0, 0, 0] -> deg [0, 0, 0] aux [0, 0] 1.0
```

- With all three edges, every vertex has two edges, so all `deg` qubits are 1.
- With edges `(0,1)` and `(1,2)`, only vertex 1 has two edges.
- With no edges, no vertex has two edges.

The `aux` qubits are back at 0 after each gate. The next part explains why.

### How a multi-controlled gate works

A real quantum computer only has gates on one or two qubits (and the
simulator works the same way). A gate with many controls has to be built from
smaller gates. `qc.multi_control` in `src/lib/circuit.py` does this with
Toffoli gates (X with two controls) and scratch qubits:

```text
aux[0] = ctl[0] AND ctl[1]
aux[1] = ctl[2] AND aux[0]
aux[2] = ctl[3] AND aux[1]
...
apply the gate to the target, controlled by the last aux qubit
undo the AND steps in reverse order
```

With k controls, the chain needs k − 1 scratch qubits. The undo steps return
all of them to `|0⟩`, so they can be used again by the next gate. The largest
multi-controlled gates in our program are the oracle (one control per vertex)
and the diffuser (one control per edge), which is why the `aux` register has
max(edges, vertices) − 1 qubits. Gates with one or two controls don't need
scratch qubits at all. The Toffoli gate itself is built from controlled
square roots of X, but you don't need to know the details for this example.

### Superposition makes this a parallel check

When the edge register is in a superposition, the degree check runs on all
subsets at once. Each subset gets its own `deg` values:

```text
(1/√8) ( |000⟩|000⟩ + |001⟩|000⟩ + ... + |110⟩|010⟩ + |111⟩|111⟩ )
          edges deg
```

The edge register and the `deg` register are now **entangled**: the value of
`deg` depends on which subset you look at. The subset `111` is the only one
where all `deg` qubits are 1.

---

## 10. Step 7: Turning a yes into a minus sign

Grover's oracle has to flip the sign of the amplitude of good subsets. A
multi-controlled X on `deg` would flip a qubit, not a sign. The trick that
converts one into the other is called **phase kickback**.

Look at what X does to `|−⟩`:

```text
X|−⟩ = X (|0⟩ − |1⟩)/√2 = (|1⟩ − |0⟩)/√2 = −|−⟩
```

The state stays the same, but it gets a factor of −1. So if the target of a
controlled X is in `|−⟩`, then:

- if the control is `|0⟩`, nothing happens;
- if the control is `|1⟩`, the whole state gets a factor of −1.

The target qubit doesn't change, and the minus sign is "kicked back" onto the
basis state of the controls. This is exactly what the oracle needs.

The program puts `chk` into `|−⟩` at the start with X (`|0⟩ → |1⟩`) and then H
(`|1⟩ → |−⟩`):

```python
  qc.x(chk)
  qc.h(chk)
```

Here is phase kickback with two qubits, a control in `|1⟩` and `chk`:

```python
qc = circuit.qc('demo')
a = qc.reg(1, 1)[0]      # control qubit, starts in |1>
chk = qc.reg(1, 0)[0]
qc.x(chk)
qc.h(chk)
print('before', np.round(np.real(np.asarray(qc.psi)), 3))
qc.cx(a, chk)
print('after ', np.round(np.real(np.asarray(qc.psi)), 3))
```

```text
before [ 0.     0.     0.707 -0.707]
after  [ 0.     0.    -0.707  0.707]
```

The four entries are the amplitudes of `|00⟩`, `|01⟩`, `|10⟩`, and `|11⟩`.
Before the CX, the state is `|1⟩|−⟩`. After the CX, every amplitude has
changed sign: the state is `−|1⟩|−⟩`. The second qubit is still `|−⟩`.

---

## 11. Step 8: The complete oracle

The oracle has three parts:

1. **Compute**: run the degree check, so that `deg[v]` is `|1⟩` for every
   vertex with exactly two selected edges.
2. **Mark**: apply an X to `chk`, controlled by all `deg` qubits. This fires
   only when every vertex has degree two, and through phase kickback it
   flips the sign of exactly those subsets.
3. **Uncompute**: run the degree check backwards, so that all `deg` qubits
   return to `|0⟩`.

```python
  # The degree checks compute deg, the inverse uncomputes it.
  sc = qc.sub()
  check_degrees(sc, g, e, deg, aux)
  sc_inv = sc.inverse()

  ...
    qc.qc(sc)                                                # 1. compute
    qc.multi_control(deg, chk, aux, ops.PauliX(), 'Oracle')  # 2. mark
    qc.qc(sc_inv)                                            # 3. uncompute
```

`qc.sub()` creates a separate, empty circuit that only records gates and
doesn't run them. We record the degree check in it once, and also build its
inverse once (`sc.inverse()` reverses the gate order and inverts each gate).
`qc.qc(sc)` then applies the recorded gates to the main circuit. This avoids
rebuilding the same gates in every iteration.

### Why uncompute?

After step 1, the edge register is entangled with the `deg` register (see the
end of section 9). The diffuser in the next step only works on the edge
register, and it relies on interference between the amplitudes of different
subsets. Two basis states can only interfere if *all* of their qubits agree.
If the subset `111` came with `deg = 111` and the subset `110` with
`deg = 010`, these amplitudes would sit in different parts of the state
vector and could not interfere. The diffuser would not work.

Running the degree check backwards sets every `deg` qubit back to `|0⟩` for
every subset. What is left is the edge register with a minus sign on the good
subsets, and all other qubits back in their starting state, the same for
every subset.

### Seeing the sign flip

To see the effect, we build the whole triangle circuit and print the
amplitudes of the 8 edge subsets before and after the oracle. Outside of the
oracle, `deg` and `aux` are `|0⟩` and `chk` is `|−⟩`, so the state has the form
"(edge amplitudes) ⊗ `|000⟩` ⊗ `|−⟩` ⊗ `|00⟩`". The helper below picks out
the edge amplitudes. It views the 512 amplitudes as a 4-dimensional array,
with one axis per register, selects `deg = 0`, `chk = 0`, `aux = 0`, and
multiplies by √2 to undo the `1/√2` of `|−⟩`.

```python
def edge_amplitudes(qc):
  """Amplitudes of the 8 edge subsets of the triangle circuit."""
  psi = np.asarray(qc.psi).reshape(8, 8, 2, 4)  # axes: e, deg, chk, aux
  return np.round(np.real(psi[:, 0, 0, 0]) * np.sqrt(2), 3)

qc = circuit.qc('demo')
e = qc.reg(3, 0)
deg = qc.reg(3, 0)
chk = qc.reg(1, 0)[0]
aux = qc.reg(2, 0)
qc.x(chk)
qc.h(chk)
qc.h(e)

sc = qc.sub()
hc.check_degrees(sc, g, e, deg, aux)
sc_inv = sc.inverse()

print('start       ', edge_amplitudes(qc))
qc.qc(sc)
qc.multi_control(deg, chk, aux, ops.PauliX(), 'Oracle')
qc.qc(sc_inv)
print('after oracle', edge_amplitudes(qc))
```

```text
start        [0.354 0.354 0.354 0.354 0.354 0.354 0.354 0.354]
after oracle [ 0.354  0.354  0.354  0.354  0.354  0.354  0.354 -0.354]
```

Only the last amplitude, for the subset `111`, changed sign. The probability
of every subset is still 1/8. The oracle has *marked* the answer, but a
measurement can't see the mark yet. The diffuser turns it into a change in
probability.

---

## 12. Step 9: The diffuser

The diffuser reflects every amplitude about the mean of all amplitudes:

```text
new amplitude = 2 · mean − old amplitude
```

Let's do this by hand for the triangle, right after the oracle. Seven
amplitudes are 0.354 and one is −0.354:

```text
mean = (7 · 0.354 − 0.354) / 8 = 0.265

unmarked:  2 · 0.265 − 0.354 =  0.177   (shrinks)
marked:    2 · 0.265 + 0.354 =  0.884   (grows)
```

The probability of the marked subset goes from 0.354² = 0.125 to
0.884² = 0.781 in a single iteration.

### Building the diffuser from gates

As a matrix, "reflect about the mean" is `2|s⟩⟨s| − I`, where `|s⟩` is the
equal superposition from step 5. We can build it from gates we already have:

```python
def diffuser(qc: circuit.qc, reg, checker, aux):
  """Simple diffuser gate. Input qubits are in a register."""

  qc.h(reg)
  qc.x(reg)
  qc.multi_control(reg, checker,
                   aux, ops.PauliX(), 'Diffuser Gate')
  qc.x(reg)
  qc.h(reg)
```

Read it from the middle out:

- `multi_control(reg, checker, ...)` with `chk` in `|−⟩` flips the sign of
  the single basis state `|11...1⟩`. That's the same phase kickback as in the
  oracle.
- The X gates before and after turn this into a sign flip of `|00...0⟩`
  instead. (X maps `|00...0⟩` to `|11...1⟩` and back.)
- The H gates before and after turn this into a sign flip of `|s⟩`, because H
  on all qubits maps `|00...0⟩` to `|s⟩` and back.

Flipping the sign of `|s⟩` is the matrix `I − 2|s⟩⟨s|`. That is exactly
−1 times the reflection about the mean. A factor of −1 on the whole state is
called a **global phase**. It does not change any probability and cannot be
measured, so it doesn't matter. You will see it in the printed numbers,
though: after this diffuser, all amplitudes have the opposite sign of the
hand calculation above.

### One full iteration

Continuing the code from step 8:

```python
hc.diffuser(qc, e, chk, aux)
print('iteration 1 ', edge_amplitudes(qc))
```

```text
iteration 1  [-0.177 -0.177 -0.177 -0.177 -0.177 -0.177 -0.177 -0.884]
```

These are the numbers from the hand calculation, with the global sign of −1.

---

## 13. Step 10: How many times to repeat

Each iteration (oracle followed by diffuser) turns the state further toward
the marked subsets. From section 3, after k iterations the probability of
measuring a marked subset is `sin²((2k + 1) θ)` with `sin θ = √(M/N)`.

For the triangle, N = 8 and M = 1, so θ = arcsin(1/√8) ≈ 0.361 radians, or
about 20.7°. Let's run four iterations and compare with the formula:

```python
for k in range(2, 5):
  qc.qc(sc)
  qc.multi_control(deg, chk, aux, ops.PauliX(), 'Oracle')
  qc.qc(sc_inv)
  hc.diffuser(qc, e, chk, aux)
  amps = edge_amplitudes(qc)
  probs = (np.abs(np.asarray(qc.psi)) ** 2).reshape(8, -1).sum(axis=1)
  theory = math.sin((2 * k + 1) * math.asin(1 / math.sqrt(8))) ** 2
  print(f'iteration {k}', amps, f'P(111) = {probs[7]:.3f}',
        f'formula = {theory:.3f}')
```

The table shows what this prints, together with the starting state and the
first iteration from step 9.

| Iterations k | Amplitude of `111` | P(`111`) | Formula sin²((2k+1)θ) |
|---|---|---|---|
| 0 | 0.354 | 0.125 | 0.125 |
| 1 | −0.884 | 0.781 | 0.781 |
| 2 | 0.972 | **0.945** | 0.945 |
| 3 | −0.575 | 0.330 | 0.330 |
| 4 | −0.110 | 0.012 | 0.012 |

After two iterations, we find the cycle with probability 0.945. After that,
the state turns past the target, and the probability drops again. After four
iterations the cycle is *less* likely than at the start. So the number of
iterations matters, and more is not better.

The program uses the formula from section 3:

```python
  solutions = find_solutions(g)
  if bool(solutions) != g.expected:
    raise AssertionError(f'{g.desc}: expected {g.expected}, but the graph '
                         f'has {len(solutions)} Hamiltonian cycles.')
  iterations = int(math.pi / 4 *
                   math.sqrt(2**nedges / max(len(solutions), 1)))
```

For the triangle, this is int(π/4 · √8) = int(2.22) = 2.

This code needs M, the number of solutions, which we count with the
classical `find_solutions`. That is a shortcut. Counting the solutions
classically is as much work as finding them, so a real application wouldn't
do it. There are two standard alternatives:

- **Quantum counting** estimates M with a quantum circuit (see
  `src/counting.py` in the same repository).
- If M is unknown, you can run Grover's algorithm with a randomly chosen
  number of iterations, check the result, and increase the range after each
  failure. This still needs about √(N/M) oracle calls on average.

The program also uses the classical count as a test: if the count disagrees
with the `expected` value written next to the graph, it stops with an error.
This catches mistakes in the test graphs.

**Graphs without a cycle.** If M = 0, the oracle marks nothing. The program
then runs the number of iterations for M = 1. The oracle does nothing, and
the diffuser only multiplies the equal superposition by −1, so the state
doesn't change and all subsets stay equally likely.

---

## 14. Step 11: Read and check the answer

On a real quantum computer, you would now measure the edge register and get
one bit string. A simulator can do better: it can show the probability of
every subset. The probability of a subset is the sum of the probabilities of
all basis states that have this subset in the edge register, whatever the
other qubits are.

```python
  probs = (np.abs(np.asarray(qc.psi)) ** 2).reshape(2**nedges, -1).sum(axis=1)
  best = int(np.argmax(probs))
  cycle = g.cycle(helper.val2bits(best, nedges))
```

The edge qubits are the leftmost bits, so `reshape(2**nedges, -1)` arranges
the state into a table with one row per subset and one column for each
setting of the other qubits. Summing each row gives the probability of that
subset. `best` is the most likely subset, and `g.cycle` checks it classically.

The classical check is a normal part of Grover's algorithm. The algorithm
gives a likely answer, not a guaranteed one. Checking a proposed cycle is
fast, so you check the answer, and if it fails, you run the search again.

The program then checks two cases.

**The graph has no cycle.** No subset passes the classical check, so the
answer is "no cycle". The program also checks that all subsets are still
equally likely. If the oracle marked a subset by mistake, that subset would
have grown, and this check would catch it.

```python
  if not solutions:
    assert cycle is None, 'Found a cycle in a graph without one.'
    assert np.allclose(probs, 1 / 2**nedges), 'Oracle marked a non-solution.'
    print('  No Hamiltonian cycle (correct)')
    return
```

**The graph has a cycle.** The program checks that:

- the most likely subset is a Hamiltonian cycle,
- the most likely subsets are exactly the solutions found classically (all
  solutions get the same probability, since the oracle treats them the same),
- and the solutions together have a probability above 0.9.

```python
  assert cycle is not None, 'Most likely state is not a Hamiltonian cycle.'
  likely = [val for val in range(2**nedges) if probs[val] > probs.max() - 1e-5]
  prob = float(np.sum(probs[solutions]))
  print(f'  Hamiltonian cycle: {cycle + [cycle[0]]}, {iterations} '
        f'iterations, probability of the {len(solutions)} solution(s): '
        f'{prob:.3f} (correct)')
  assert likely == solutions, 'Not all solutions are the most likely states.'
  assert prob > 0.9, 'Solutions are not amplified enough.'
```

The last check matters. A search that only made the cycle *slightly* more
likely than the other subsets would still find it as the most likely state,
but it wouldn't be a working Grover search.

### A graph with several cycles

The complete graph on 4 vertices (every vertex connected to every other one)
has 6 edges and three different Hamiltonian cycles:

```python
k4 = hc.Graph(4, True, 'complete-4',
              [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])
for s in hc.find_solutions(k4):
  bits = helper.val2bits(s, 6)
  print(s, bits, k4.cycle(bits))
```

```text
23 [0, 1, 0, 1, 1, 1] [0, 3, 1, 2]
43 [1, 0, 1, 0, 1, 1] [0, 1, 3, 2]
60 [1, 1, 1, 1, 0, 0] [0, 1, 2, 3]
```

Here N = 64 and M = 3, so the program runs int(π/4 · √(64/3)) = 3 iterations.
The oracle marks all three subsets, Grover's algorithm raises all three
together, and a measurement returns one of them, each with about the same
probability.

---

## 15. The whole program

Here is `build_circuit` from `src/hamiltonian_cycle.py`, with the steps of
this document marked:

```python
def build_circuit(g: Graph):
  """Build and run the circuit for the Grover Search."""

  # Step 3: the degree test only works for fewer than 6 vertices.
  if g.num >= 6:
    raise ValueError('The oracle requires graphs with fewer than 6 vertices.')

  # Step 4: reserve the qubits.
  nedges = len(g.edges)
  qc = circuit.qc('hamiltonian cycle')
  e = qc.reg(nedges, 0)
  deg = qc.reg(g.num, 0)
  chk = qc.reg(1, 0)[0]
  aux = qc.reg(max(nedges, g.num) - 1, 0)

  # Step 7: put chk into |->, for phase kickback.
  qc.x(chk)
  qc.h(chk)

  # Step 10: count the solutions, and choose the number of iterations.
  solutions = find_solutions(g)
  iterations = int(math.pi / 4 *
                   math.sqrt(2**nedges / max(len(solutions), 1)))

  # Step 6 and 8: record the degree check and its inverse.
  sc = qc.sub()
  check_degrees(sc, g, e, deg, aux)
  sc_inv = sc.inverse()

  # Step 5: equal superposition of all edge subsets.
  qc.h(e)

  # Steps 8 and 9: the Grover iterations.
  for _ in range(iterations):
    qc.qc(sc)
    qc.multi_control(deg, chk, aux, ops.PauliX(), 'Oracle')
    qc.qc(sc_inv)
    diffuser(qc, e, chk, aux)

  # Step 11: read and check the answer.
  probs = (np.abs(np.asarray(qc.psi)) ** 2).reshape(2**nedges, -1).sum(axis=1)
  ...
```

(The listing leaves out the print statements and the checks shown in step 10
and step 11.)

`main` runs the search on 16 small graphs, with and without Hamiltonian
cycles. The output:

```text
Quantum Hamiltonian Cycle Finder
Graph [triangle]: 3 vertices, 3 edges -> 9 qubits
  Hamiltonian cycle: [0, 1, 2, 0], 2 iterations, probability of the 1 solution(s): 0.945 (correct)
Graph [rect]: 4 vertices, 4 edges -> 12 qubits
  Hamiltonian cycle: [0, 1, 2, 3, 0], 3 iterations, probability of the 1 solution(s): 0.961 (correct)
Graph [rect+diag]: 4 vertices, 5 edges -> 14 qubits
  Hamiltonian cycle: [0, 1, 2, 3, 0], 4 iterations, probability of the 1 solution(s): 0.999 (correct)
Graph [complete-4]: 4 vertices, 6 edges -> 16 qubits
  Hamiltonian cycle: [0, 3, 1, 2, 0], 3 iterations, probability of the 3 solution(s): 0.998 (correct)
Graph [loop-5]: 5 vertices, 5 edges -> 15 qubits
  Hamiltonian cycle: [0, 1, 2, 3, 4, 0], 4 iterations, probability of the 1 solution(s): 0.999 (correct)
Graph [loop-5+diag]: 5 vertices, 6 edges -> 17 qubits
  Hamiltonian cycle: [0, 1, 2, 3, 4, 0], 6 iterations, probability of the 1 solution(s): 0.997 (correct)
Graph [line]: 4 vertices, 3 edges -> 11 qubits
  No Hamiltonian cycle (correct)
Graph [triangle with stray 0]: 4 vertices, 4 edges -> 12 qubits
  No Hamiltonian cycle (correct)
Graph [triangle with stray 1]: 4 vertices, 4 edges -> 12 qubits
  No Hamiltonian cycle (correct)
Graph [triangle with stray 2]: 4 vertices, 4 edges -> 12 qubits
  No Hamiltonian cycle (correct)
Graph [star formation]: 4 vertices, 3 edges -> 11 qubits
  No Hamiltonian cycle (correct)
Graph [two lines]: 4 vertices, 2 edges -> 10 qubits
  No Hamiltonian cycle (correct)
Graph [two lines with stray]: 5 vertices, 4 edges -> 14 qubits
  No Hamiltonian cycle (correct)
Graph [two lines with connector]: 5 vertices, 5 edges -> 15 qubits
  No Hamiltonian cycle (correct)
Graph [bowtie]: 5 vertices, 6 edges -> 17 qubits
  No Hamiltonian cycle (correct)
Graph [K(2,3)]: 5 vertices, 6 edges -> 17 qubits
  No Hamiltonian cycle (correct)
```

Some of the graphs without a cycle are there to test specific traps:

- `two lines with connector` contains the cycle 1 → 2 → 3 → 4 → 1, but vertex 0
  has only one edge. A cycle that doesn't visit every vertex is not enough.
- `bowtie` is two triangles that share vertex 2. Every vertex has at least two
  edges, but a trip through all vertices would have to pass vertex 2 twice.
- `K(2,3)` connects each of the vertices 0 and 1 to each of the vertices 2, 3
  and 4. A trip has to alternate between the two groups, which is impossible
  when the groups have different sizes.

---

## 16. Limits of this approach

**Only graphs with fewer than 6 vertices.** As explained in step 3, the oracle
checks "every vertex has exactly two selected edges", which is only the same
as "Hamiltonian cycle" below 6 vertices. For larger graphs, the oracle would
also need to check that the selected edges are connected. One way is to
compute, with reversible gates, which vertices are reachable from vertex 0
using only the selected edges, and require that all are. This needs many more
qubits and gates.

**The search space grows with the edges.** There are 2^|E| subsets of edges.
Dense graphs have many edges, so the search space can be much larger than the
number of possible trips. The complete graph on 5 vertices has 10 edges and
1024 subsets, but only 12 different Hamiltonian cycles.

**Grover's speedup is only quadratic.** Grover's algorithm needs about
√(N/M) oracle calls, compared with about N/M checks for a classical brute
force. For N = 2^|E|, that is about 2^(|E|/2) calls. This is still
exponential. Grover's algorithm makes the brute-force search faster, but it
doesn't turn an NP-complete problem into an easy one. Good classical
algorithms for Hamiltonian cycles are also much smarter than brute force.

**Simulating is expensive.** The simulator stores 2ⁿ complex numbers for n
qubits. The largest example here has 17 qubits, or about 131,000 amplitudes,
which is fine. Each additional qubit doubles the memory and the time.

**Qubit and gate counts are not optimized.** The circuit uses a separate
`deg` qubit for every vertex and a simple multi-controlled gate construction
with k − 1 scratch qubits. The number of degree check gates also grows with
the square of the vertex degree, since there is one gate per pair of edges. A
real implementation would try to reduce all of these, because qubits and
gates are scarce on real hardware.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| Amplitude | A complex number attached to a basis state. Its squared magnitude is the probability of measuring that state. |
| Ancilla, scratch qubit | A helper qubit that holds intermediate results and is returned to `|0⟩` afterwards. The `aux` register. |
| Basis state | A state that corresponds to one bit string, such as `|101⟩`. |
| Controlled gate | A gate that only acts if its control qubits are `|1⟩` (or `|0⟩`, for controlled-by-0). |
| Diffuser | The part of a Grover iteration that reflects all amplitudes about their mean. |
| Entanglement | A state of several qubits that can't be described qubit by qubit, for example when the `deg` values depend on the edge subset. |
| Global phase | A factor, such as −1, on the whole state. It can't be measured and has no effect. |
| Grover iteration | One application of the oracle followed by the diffuser. |
| Hamiltonian cycle | A round trip through a graph that visits every vertex exactly once. |
| Interference | Amplitudes adding up or canceling when gates combine them. |
| Oracle | A circuit that flips the sign of the amplitude of every basis state that passes a test. |
| Phase kickback | Using a target qubit in `|−⟩`, so that a controlled X multiplies the state by −1 instead of flipping the target. |
| Register | A group of qubits used together, such as the edge register `e`. |
| Superposition | A state with non-zero amplitudes on several basis states. |
| Uncomputation | Running a computation backwards to return work qubits to their starting state. |
