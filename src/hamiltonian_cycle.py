# python3
"""Example: Hamiltonian Cycle via Grover's Algorithm."""

# A Hamiltonian cycle is a cycle that visits every vertex of a graph exactly
# once. We search for one with Grover's algorithm.
#
# Encoding: There is one qubit per edge. A basis state of this edge register
# is a subset of the edges, with |1> meaning the edge is part of the subset.
#
# A subset of edges is a Hamiltonian cycle if:
#   (a) every vertex has exactly two selected edges, and
#   (b) the selected edges are connected.
#
# The oracle only checks (a). With (a), the selected edges form disjoint
# cycles that together cover all vertices. Each of these cycles needs at
# least 3 vertices, hence there must be at least 6 vertices for two of them.
# For graphs with fewer than 6 vertices, (a) is a Hamiltonian cycle, and we
# limit this example to these graphs. Larger graphs would also need a
# connectivity check, e.g., to reject two disjoint triangles.
#
# The check for a vertex v with incident edges e_1, ..., e_k is an OR over
# all pairs (i, j): e_i and e_j are |1> and all other edges of v are |0>.
# At most one pair can match, which allows us to implement the OR with one
# multi-controlled X gate per pair, each one flipping the result for v.

import itertools
import math
from typing import List, Optional, Sequence, Tuple

from absl import app
import numpy as np

from src.lib import circuit
from src.lib import helper
from src.lib import ops


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

  def cycle(self, bits: Sequence[int]) -> Optional[List[int]]:
    """Return the vertices of the cycle, or None if bits is not a cycle."""

    # bits selects the edges, in the order of self.edges. This is the
    # classical check for both (a) and (b) from above.
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


def find_solutions(g: Graph) -> List[int]:
  """Classically find all Hamiltonian cycles, as edge subsets (numbers)."""

  nbits = len(g.edges)
  return [val for val in range(2**nbits)
          if g.cycle(helper.val2bits(val, nbits))]


def check_degrees(qc: circuit.qc, g: Graph, e, deg, aux):
  """Flip deg[v] if exactly two of the edges of vertex v are selected."""

  for v in range(g.num):
    inc = g.incident(v)
    for i, j in itertools.combinations(inc, 2):
      # Control by |1> for edges i and j, and by |0> for all other edges.
      ctl = [e[k] if k in (i, j) else [e[k]] for k in inc]
      qc.multi_control(ctl, deg[v], aux, ops.PauliX(), f'degree({v})')


def diffuser(qc: circuit.qc, reg, checker, aux):
  """Simple diffuser gate. Input qubits are in a register."""

  qc.h(reg)
  qc.x(reg)
  qc.multi_control(reg, checker,
                   aux, ops.PauliX(), 'Diffuser Gate')
  qc.x(reg)
  qc.h(reg)


def build_circuit(g: Graph):
  """Build and run the circuit for the Grover Search."""

  if g.num >= 6:
    raise ValueError('The oracle requires graphs with fewer than 6 vertices.')

  nedges = len(g.edges)
  qc = circuit.qc('hamiltonian cycle')
  e = qc.reg(nedges, 0)
  deg = qc.reg(g.num, 0)
  chk = qc.reg(1, 0)[0]
  # The multi-controlled gates have at most max(nedges, g.num) controls.
  aux = qc.reg(max(nedges, g.num) - 1, 0)

  print(f'Graph [{g.desc}]: {g.num} vertices, {nedges} edges -> '
        f'{qc.nbits} qubits')

  # The oracle and the diffuser are multi-controlled X gates with chk as the
  # target. They only flip the phase of a state if chk is in |->.
  qc.x(chk)
  qc.h(chk)

  # With N states and M solutions, the best number of iterations is
  # pi/4 sqrt(N / M). We count the solutions classically, a quantum counting
  # algorithm (see counting.py) could estimate M instead. Without solutions,
  # we run the iterations for M = 1. The oracle marks no state then, and the
  # state does not change.
  solutions = find_solutions(g)
  if bool(solutions) != g.expected:
    raise AssertionError(f'{g.desc}: expected {g.expected}, but the graph '
                         f'has {len(solutions)} Hamiltonian cycles.')
  iterations = int(math.pi / 4 *
                   math.sqrt(2**nedges / max(len(solutions), 1)))

  # The degree checks compute deg, the inverse uncomputes it.
  sc = qc.sub()
  check_degrees(sc, g, e, deg, aux)
  sc_inv = sc.inverse()

  qc.h(e)
  for _ in range(iterations):
    qc.qc(sc)
    qc.multi_control(deg, chk, aux, ops.PauliX(), 'Oracle')
    qc.qc(sc_inv)
    diffuser(qc, e, chk, aux)

  # The probability of an edge subset is the sum over all states of the
  # other qubits. The edge register holds the first qubits.
  probs = (np.abs(np.asarray(qc.psi)) ** 2).reshape(2**nedges, -1).sum(axis=1)
  best = int(np.argmax(probs))
  cycle = g.cycle(helper.val2bits(best, nedges))

  if not solutions:
    # The oracle marks no state, and the edge subsets stay equally likely.
    # The most likely one is not a cycle, which is the answer.
    assert cycle is None, 'Found a cycle in a graph without one.'
    assert np.allclose(probs, 1 / 2**nedges), 'Oracle marked a non-solution.'
    print('  No Hamiltonian cycle (correct)')
    return

  # The solutions have the same probability, and all of them have to be the
  # most likely states. With the best number of iterations, the search finds
  # one of them with a probability of more than 0.9 for these graphs.
  assert cycle is not None, 'Most likely state is not a Hamiltonian cycle.'
  likely = [val for val in range(2**nedges) if probs[val] > probs.max() - 1e-5]
  prob = float(np.sum(probs[solutions]))
  print(f'  Hamiltonian cycle: {cycle + [cycle[0]]}, {iterations} '
        f'iterations, probability of the {len(solutions)} solution(s): '
        f'{prob:.3f} (correct)')
  assert likely == solutions, 'Not all solutions are the most likely states.'
  assert prob > 0.9, 'Solutions are not amplified enough.'


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Quantum Hamiltonian Cycle Finder')

  build_circuit(Graph(3, True, 'triangle',
                      [(0, 1), (1, 2), (2, 0)]))
  build_circuit(Graph(4, True, 'rect',
                      [(0, 1), (1, 2), (2, 3), (3, 0)]))
  build_circuit(Graph(4, True, 'rect+diag',
                      [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]))
  build_circuit(Graph(4, True, 'complete-4',
                      [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]))
  build_circuit(Graph(5, True, 'loop-5',
                      [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]))
  build_circuit(Graph(5, True, 'loop-5+diag',
                      [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 0)]))
  build_circuit(Graph(4, False, 'line',
                      [(0, 1), (1, 2), (2, 3)]))
  build_circuit(Graph(4, False, 'triangle with stray 0',
                      [(0, 1), (1, 2), (2, 0), (0, 3)]))
  build_circuit(Graph(4, False, 'triangle with stray 1',
                      [(0, 1), (1, 2), (2, 0), (1, 3)]))
  build_circuit(Graph(4, False, 'triangle with stray 2',
                      [(0, 1), (1, 2), (2, 0), (2, 3)]))
  build_circuit(Graph(4, False, 'star formation',
                      [(0, 1), (0, 2), (0, 3)]))
  build_circuit(Graph(4, False, 'two lines',
                      [(0, 1), (2, 3)]))
  build_circuit(Graph(5, False, 'two lines with stray',
                      [(0, 1), (1, 2), (2, 3), (1, 4)]))
  # This graph has the cycle 1-2-3-4, but vertex 0 only has one edge.
  build_circuit(Graph(5, False, 'two lines with connector',
                      [(0, 1), (1, 2), (2, 3), (1, 4), (3, 4)]))
  # Two triangles that share vertex 2. All vertices have two or more edges,
  # but a cycle through all of them would have to visit vertex 2 twice.
  build_circuit(Graph(5, False, 'bowtie',
                      [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]))
  # The complete bipartite graph K(2,3). A cycle alternates between the two
  # sides, which have different sizes.
  build_circuit(Graph(5, False, 'K(2,3)',
                      [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]))


if __name__ == '__main__':
  app.run(main)
