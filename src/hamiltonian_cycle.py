# python3
"""Example: Hamiltonian Cycle via Grover's Algorithm."""


# ==============================================
# This is WIP (also not working as intended yet)
# ==============================================

import math
from typing import List
from absl import app
import numpy as np

from src.lib import circuit
from src.lib import ops


class Graph:
  """Hold a graph definition."""

  def __init__(self, num_vertices: int, expected: bool, desc: str,
               edges: List[int]):
    self.num = num_vertices
    self.edges = edges
    self.desc = desc
    self.expected = expected

  def verify(self, bits):
    """Verify whether there is a non-|0> in bits."""

    for bit in bits:
      if bit:
        return True
    return False


def diffuser(qc: circuit.qc, reg, checker, aux):
  """Simple diffuser gate. Input qubits are in a register."""

  qc.h(reg)
  qc.x(reg)
  qc.multi_control(reg, checker,
                   aux, ops.PauliX(), 'Diffuser Gate')
  qc.x(reg)
  qc.h(reg)


def build_circuit(g: Graph):
  """Build the circuit for the Grover Search."""

  print(f'Graph {g.desc}: v: {g.num}, e: {len(g.edges)}', end='')
  iterations = int(math.pi / 4 * math.sqrt(2**len(g.edges)))
  qc = circuit.qc('graph')
  v = qc.reg(g.num, 0)
  e = qc.reg(len(g.edges), 0)
  chk = qc.reg(1, 0)[0]
  aux = qc.reg(g.num*2)

  qc.h(e)
  qc.x(chk)
  qc.h(chk)

  for _ in range(iterations):
    sc = qc.sub()
    for idx, edge in enumerate(g.edges):
      sc.cry(e[idx], edge[0], np.pi/2)
      sc.cry(e[idx], edge[1], np.pi/2)

    qc.qc(sc)
    qc.multi_control(v, chk, aux, ops.PauliX(), 'multi-X')
    qc.qc(sc.inverse())

    diffuser(qc, e, chk, aux)

  maxbits, _ = qc.psi.maxprob()
  if g.expected != g.verify(maxbits[:g.num]):
    print(' INCORRECT', maxbits[:g.num])
    # raise AssertionError('INCORRECT')
  else:
    print('  Has circle: ', g.expected, ' (correct)')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Quantum Hamiltonian Cycle Finder (WIP)')

  build_circuit(Graph(3, True, 'triangle',
                      [(0, 1), (1, 2), (2, 1)]))
  build_circuit(Graph(4, True, 'rect',
                      [(0, 1), (1, 2), (2, 3), (3, 0)]))
  build_circuit(Graph(4, True, 'rect+diag',
                      [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]))
  build_circuit(Graph(5, True, 'loop-5',
                      [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]))
  build_circuit(Graph(5, True, 'loop-5+diag',
                      [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5)]))
  build_circuit(Graph(4, False, 'line',
                      [(0, 1), (1, 2), (2, 3)]))
  build_circuit(Graph(5, False, 'triangle with stray 0',
                      [(0, 1), (1, 2), (2, 1), (0, 4)]))
  build_circuit(Graph(5, False, 'triangle with stray 1',
                      [(0, 1), (1, 2), (2, 1), (1, 4)]))
  build_circuit(Graph(5, False, 'triangle with stray 2',
                      [(0, 1), (1, 2), (2, 1), (2, 4)]))
  build_circuit(Graph(4, False, 'star formation',
                      [(0, 1), (0, 2), (0, 3)]))
  build_circuit(Graph(4, False, 'two lines',
                      [(0, 1), (2, 3)]))
  build_circuit(Graph(5, False, 'two lines with stray',
                      [(0, 1), (1, 2), (2, 3), (1, 4)]))
  build_circuit(Graph(5, True, 'two lines with connector',
                      [(0, 1), (1, 2), (2, 3), (1, 4), (3, 4)]))


if __name__ == '__main__':
  app.run(main)
