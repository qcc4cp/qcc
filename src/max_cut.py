# python3
"""Example: Max Cut Algorithm for bi-directional graph."""

import random
import timeit
from typing import List, Tuple

from absl import app
from absl import flags

import numpy as np
from src.lib import helper
from src.lib import ops


flags.DEFINE_integer('nodes', 12, 'Number of graph nodes')
flags.DEFINE_boolean('graph', False, 'Dump graph in dot format')
flags.DEFINE_integer('iterations', 10, 'Number of experiments')


# Nodes are tuples: (from, to, edge weight).
#
def build_graph(num: int = 0) -> Tuple[int, List[Tuple[int, int, float]]]:
  """Build a graph of num nodes."""

  if num < 3:
    raise app.UsageError('Must request graph of at least 3 nodes.')
  weight = 5.0
  nodes = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]
  for i in range(num - 3):
    l = random.sample(range(0, 3 + i - 1), 2)
    nodes.append((3 + i, l[0], weight * np.random.random()))
    nodes.append((3 + i, l[1], weight * np.random.random()))
  return num, nodes


def graph_to_dot(n: int, nodes: List[Tuple[int, int, float]], max_cut) -> None:
  """Convert graph (up to 64 nodes) to dot file."""

  print('graph {')
  print('  {\n    node [  style=filled ]')
  pattern = bin(max_cut)[2:].zfill(n)
  for idx, val in enumerate(pattern):
    if val == '0':
      print(f'    "{idx}" [fillcolor=lightgray]')
  print('  }')
  for node in nodes:
    print('  "{}" -- "{}" [label="{:.1f}",weight="{:.2f}"];'
          .format(node[0], node[1], node[2], node[2]))
  print('}')


def graph_to_adjacency(n: int,
                       nodes: List[Tuple[int, int, float]]) -> ops.Operator:
  """Compute adjacency matrix from graph."""

  op = np.zeros((n, n))
  for node in nodes:
    op[node[0], node[1]] = node[2]
    op[node[1], node[0]] = node[2]
  return ops.Operator(op)


def graph_to_hamiltonian(n: int,
                         nodes: List[Tuple[int, int, float]]) -> ops.Operator:
  """Compute Hamiltonian matrix from graph."""

  hamil = np.zeros((2**n, 2**n))
  for node in nodes:
    idx1 = node[0]
    idx2 = node[1]
    if idx1 > idx2:
      idx1, idx2 = idx2, idx1
    op = 1.0
    for _ in range(idx1):
      op = op * ops.Identity()
    op = op * (node[2] * ops.PauliZ())
    for _ in range(idx1 + 1, idx2):
      op = op * ops.Identity()
    op = op * (node[2] * ops.PauliZ())
    for _ in range(idx2 + 1, n):
      op = op * ops.Identity()
    hamil = hamil + op
  return ops.Operator(hamil)


def tensor_diag(n: int, fr: int, to: int, w: float):
  """Construct a tensor product from diagonal matrices."""

  def tensor_product(w1: float, w2: float, diag):
    # pylint: disable=g-complex-comprehension
    return [j for i in zip([x * w1 for x in diag],
                           [x * w2 for x in diag]) for j in i]

  diag = [w, -w] if (0 == fr or 0 == to) else [1, 1]
  for i in range(1, n):
    if i == fr or i == to:
      diag = tensor_product(w, -w, diag)
    else:
      diag = tensor_product(1, 1, diag)
  return diag


def graph_to_diagonal_h(n: int,
                        nodes: List[Tuple[int, int, float]]) -> List[float]:
  """Construct diag(H)."""

  h = [0.0] * 2**n
  for node in nodes:
    diag = tensor_diag(n, node[0], node[1], node[2])
    for idx, val in enumerate(diag):
      h[idx] += val
  return h


def compute_max_cut(n: int,
                    nodes: List[Tuple[int, int, float]]) -> int:
  """Compute (inefficiently) the max cut, exhaustively."""

  max_cut = -1000.0
  for bits in helper.bitprod(n):
    # Collect in/out sets.
    iset = []
    oset = []
    for idx, val in enumerate(bits):
      if val == 0:
        iset.append(idx)
      else:
        oset.append(idx)

    # Compute costs for this cut, record maximum.
    cut = 0.0
    for node in nodes:
      if node[0] in iset and node[1] in oset:
        cut += node[2]
      if node[1] in iset and node[0] in oset:
        cut += node[2]
    if cut > max_cut:
      max_cut_in, max_cut_out = iset.copy(), oset.copy()
      max_cut = cut
      max_bits = bits

  state = bin(helper.bits2val(max_bits))[2:].zfill(n)
  print('Max Cut. N: {}, Max: {:.1f}, {}-{}, |{}>'
        .format(n, np.real(max_cut), max_cut_in, max_cut_out,
                state))
  return helper.bits2val(max_bits)


def benchmark() -> None:
  """Simple benchmark for exhaustive max cut computation."""

  num = 0
  def compute_max():
    n, nodes = build_graph(num)
    compute_max_cut(n, nodes)

  for num in range(18):
    print('Nodes: {}: Time:{:.2f} secs'
          .format(num+6, timeit.timeit(compute_max, number=1)))


def run_experiment(num_nodes: int):
  """Run an experiment, compute H, match against max-cut."""

  n, nodes = build_graph(num_nodes)
  max_cut = compute_max_cut(n, nodes)
  #
  # These two lines are the basic implementation, where
  # a full matrix is being constructed. However, these
  # are diagonal, and can be constructed much faster.
  #  H       = graph_to_hamiltonian(n, nodes)
  #  diag    = H.diagonal()
  #
  diag = graph_to_diagonal_h(n, nodes)
  min_idx = np.argmin(diag)
  if flags.FLAGS.graph:
    graph_to_dot(n, nodes, max_cut)

  # Results...
  if min_idx == max_cut:
    print('SUCCESS: {:+10.2f} |{}>'
          .format(np.real(diag[min_idx]),
                  bin(min_idx)[2:].zfill(n)))
  else:
    print('FAIL   : {:+10.2f} |{}>  '
          .format(np.real(diag[min_idx]),
                  bin(min_idx)[2:].zfill(n)),
          end='')
    print('Max-Cut: {:+10.2f} |{}>'
          .format(np.real(diag[max_cut]),
                  bin(max_cut)[2:].zfill(n)))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for _ in range(flags.FLAGS.iterations):
    run_experiment(flags.FLAGS.nodes)


if __name__ == '__main__':
  app.run(main)
