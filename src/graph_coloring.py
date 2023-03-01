# python3
"""Example: Graph Coloring via Grover's Algorithm."""


# =========================================================
# Note: To simplify and reduce the number of solutions, we
#       look for colorings where all colors are the _same_.
# =========================================================


import itertools
from typing import List, Tuple

from absl import app
import numpy as np

from src.lib import circuit
from src.lib import helper
from src.lib import ops


def compare_pairs_equal(qc, a, b, c, d, w0, w1, chk):
  """Compare two pairs of qubits."""

  # We can test whether two qubits are the same with a simple
  # CNOT gate:
  #
  # a----o----
  #      |
  # b----X----
  #
  # The truth table is:
  #   a   b  ->   b
  # -------------------
  #   0   0       0 (unmodified)
  #   0   1       1 (unmodified)
  #   1   0       1 (flipped to 1)
  #   1   1       0 (flipped to 0)
  #
  # If two qubits are the same, the cx will result in a |0>
  # on the second qubit!
  #
  # We then Controlled-Not-By-0 the 2nd register to the work registers
  # add fiunally AND the work registers. If both qubits were equal, the
  # chk register will turn into |1>. Of course, we have to uncompute this to
  # restore b at the end to its previous state.
  #
  qc.cx(a, c)
  qc.cx0(c, w0)
  qc.cx(b, d)
  qc.cx0(d, w1)

  qc.ccx(w0, w1, chk)

  qc.cx0(d, w1)
  qc.cx(b, d)
  qc.cx0(c, w0)
  qc.cx(a, c)


def test_qubit_equality_circuit():
  """Test the equality circuitty for correctness."""

  # To check whether pairs of qubits are the same we
  # follow this recipe:
  #   To check whether two qubits are the same, a simple
  #   Controlled-Not will do:
  #     q0 ---o---
  #     q1    |
  #     q2 ---X---
  #
  #   If the qubits were the same, q2 will be |0>.
  #
  #   However, this also changes q2, so we have to store away
  #   the result and undo the Controlled-Not:
  #     q0 ---o-------o---
  #     q1    |       |
  #     q2 ---X-X-o-X-X---
  #     q3        |
  #      w -------X------- 1 if q0 == q2
  #
  # To do this for two qubits, we have to compute two worker qubits
  # and finally AND them together. To make the AND work, both inputs
  # should be 1, hence we have to add an X-gate to the workers before
  # applying a CXX gate (or use cx0 for convenience).
  #
  # As a result, if the pairs are equal, the final result of
  # the AND will be |1> in the chk qubit.
  #
  for bits in itertools.product([0, 1], repeat=4):
    qc = circuit.qc()
    qc.reg(4, bits)
    aux = qc.reg(2, 0)
    chk = qc.reg(1)[0]

    compare_pairs_equal(qc, 0, 1, 2, 3, aux[0], aux[1], chk)

    maxbits, _ = qc.psi.maxprob()
    if bits[0:2] == bits[2:4]:
      if maxbits[chk] == 0:
        raise AssertionError('Incorrect equality check')


# Example graph (adding additional edges between C-F and/or
# D/F has the ability to force certain color assignments):
# This example is identical to:
#   https://learn.microsoft.com/en-us/training/modules/\
#     solve-graph-coloring-problems-grovers-search/4-implement-quantum-oracle
#
#  A---B
#  |\ /|
#  | X |
#  |/ \|
#  D---C
#       \
#        F
#
# The problem in our implementation is that we are very generous
# with ancilla qubits. Furthermore, for Grover to work, the number
# of solutions must be small.
#
# Hence, we don't check for valid color assignments, but for simpler
# cases, eg., all colors are the same. There are far fewer of those
# cases.  We will only implement very small graphs to keep computation
# time somewhat bounded.


class Graph:
  """Hold a graph definition."""

  def __init__(
      self, num_vertices: int, desc: str, edges: List[Tuple[int, int]]
  ):
    self.num = num_vertices
    self.edges = edges
    self.desc = desc

  def verify(self, bits, n: int = 2):
    """Verify that no connected vertices have the same color."""

    # For each edge, we check whether the colors assigned to fr/to
    # are different.
    #   bits: are the measured bits from the quantum state.
    #   n   : number of bits used to encode colors.
    different = 0
    for edge in self.edges:
      if (bits[edge[0] * n: edge[0] * n + n] !=
          bits[edge[1] * n: edge[1] * n + n]):
        different += 1
    return different


def diffuser(qc: circuit.qc, reg, checker, aux):
  """Simple diffuser gate. Input qubits are in a register."""

  qc.h(reg)
  qc.x(reg)
  qc.multi_control(reg, checker, aux, ops.PauliX(), 'Diffuser Gate')
  qc.x(reg)
  qc.h(reg)


def build_circuit(g: Graph):
  """Build a circuit from a graph."""

  qc = circuit.qc('Graph Circuit')
  reg = qc.reg(g.num * 2)
  chk = qc.reg(len(g.edges))
  res = qc.reg(1)[0]
  tmp = qc.reg(g.num * 2 - 1)

  print(f'Solving [{g.desc}]: ', end='')
  print(f'{g.num} vertices, {len(g.edges)} edges -> {qc.nbits} qubits')
  iterations = 1

  qc.h(reg)
  for _ in range(iterations):
    sc = qc.sub()
    for idx, edge in enumerate(g.edges):
      fr = edge[0] * 2
      to = edge[1] * 2
      compare_pairs_equal(sc, fr, fr + 1, to, to + 1, tmp[0], tmp[1], chk[idx])

    qc.qc(sc)
    qc.multi_control(chk, res, tmp, ops.PauliX(), 'multi')
    qc.qc(sc.inverse())

    diffuser(qc, reg, res, tmp)

  # Now let's measure and find all states that have the highest
  # probability. These will be the possible
  # solutions to the (inverse) graph coloring problem, which
  # is to find the coloring with all colors being the same.
  #
  _, maxprob = qc.psi.maxprob()
  for idx, val in enumerate(qc.psi):
    if np.real(val.conj() * val) > (maxprob - 0.005):
      bits = helper.val2bits(idx, qc.nbits)
      print('  Color:', bits[0 : g.num * 2])
      if g.verify(bits):
        raise AssertionError('Incorrect color assignment found.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print("Graph coloring via Grover's Search. ", end='')
  print('Find identical colors (2 qubits each).')

  test_qubit_equality_circuit()

  build_circuit(Graph(2, 'simple line', [(0, 1)]))
  build_circuit(Graph(3, 'simple triangle', [(0, 1), (1, 2), (2, 0)]))

  # These may run too slow, disable if that's the case
  build_circuit(Graph(4, 'star formation', [(0, 1), (0, 2), (0, 3)]))
  build_circuit(Graph(4, 'rectangle', [(0, 1), (1, 2), (2, 3)]))

  # This is the slowest. Disabled by default.
  # build_circuit(Graph(4, 'rectangle+diag', [(0, 1), (1, 2), (2, 3), (3, 0)]))


if __name__ == '__main__':
  app.run(main)
