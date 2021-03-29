# python3
"""Example: Max Cut Algorithm for bi-directional graph."""

import math
import numpy as np
import timeit

from absl import app
from absl import flags

from src.lib import helper
from src.lib import ops

flags.DEFINE_integer('nodes', 12, 'Number of graph nodes')
flags.DEFINE_boolean('graph', False, 'Dump graph in dot format')
flags.DEFINE_integer('iterations', 10, 'Number of experiments')


# Nodes are tuples: (from, to, edge weight).
#
def build_graph(num:int=0) -> (int, list):
    """Build a graph of num nodes."""

    if num < 3:
        raise app.UsageError('Must request graph of at least 3 nodes.')
    weight = 5.0
    nodes = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0) ]
    for i in range(num-3):
        node0 = np.random.randint(0, 3 + i)
        node1 = np.random.randint(0, 3 + i)
        while node1 == node0:
            node1 = np.random.randint(0, 3 + i)

        nodes.append((3 + i, node0,
                      weight*np.random.random()))
        nodes.append((3 + i, node1,
                      weight*np.random.random()))
    return num, nodes



def graph_to_dot(n:int, nodes:list, max_cut) -> None:
    """Convert graph (up to 64 nodes) to dot file."""

    print('graph {')
    print('  {\n    node [  style=filled ]')
    pattern = bin(max_cut)[2:].zfill(n)
    for i in range(len(pattern)):
        if pattern[i] == '0':
            print(f'    "{i}" [fillcolor=lightgray]')
    print('  }')
    for node in nodes:
        print('  "{}" -- "{}" [label="{:.1f}",weight="{:.2f}"];'
              .format(node[0], node[1], node[2], node[2]))
    print('}')


def graph_to_adjacency(n:int, nodes:list) -> ops.Operator:
    """Compute adjacency matrix from graph."""

    op = np.zeros((n, n))
    for node in nodes:
        op[node[0], node[1]] = node[2]
        op[node[1], node[0]] = node[2]
    return ops.Operator(op)


def graph_to_hamiltonian(n:int, nodes:list) -> ops.Operator:
    """Compute Hamiltonian matrix from graph."""

    H = np.zeros((2**n, 2**n))
    for node in nodes:
        idx1 = node[0]
        idx2 = node[1]
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        op = 1.0
        for i in range(idx1):
            op = op * ops.Identity()
        op = op * (node[2] * ops.PauliZ())
        for i in range(idx1 + 1, idx2):
            op = op * ops.Identity()
        op = op * (node[2] * ops.PauliZ())
        for i in range(idx2 + 1, n):
            op = op * ops.Identity()
        H = H + op
    return ops.Operator(H)


def tensor_diag(n:int, fr:int, to:int, w:float):
    """Construct a tensor product from diagonal matrices."""

    def tensor_product(w1:float, w2:float, diag):
        l = []
        for x in diag:
            l.append(x * w1)
            l.append(x * w2)
        return l

    diag = []
    for i in range(n):
        # Sigma Z
        if i == fr or i == to:
            if diag == []:
                diag = [w, -w]
                continue
            diag = tensor_product(w, -w, diag)
            continue
        # Identity
        if diag == []:
            diag = [1, 1]
            continue
        diag = tensor_product(1, 1, diag)
    return diag


def graph_to_diagonal_h(n:int, nodes:list) -> np.ndarray:
    """Construct diag(H)."""

    h = [0.0] * 2**n
    for node in nodes:
      diag = tensor_diag(n, node[0], node[1], node[2])
      for i in range(len(diag)):
          h[i] = h[i] + diag[i]
    return h


def compute_max_cut(n:int, nodes:list) -> None:
    """Compute (inefficiently) the max cut, exhaustively."""

    max_cut = -1000
    for bits in helper.bitprod(n):
        # Collect in/out sets.
        iset = []
        oset = []
        for i in range(len(bits)):
            iset.append(i) if bits[i] == 0 else oset.append(i)

        # Compute costs for this cut, record maximum.
        cut = 0
        for node in nodes:
            if node[0] in iset and node[1] in oset:
                cut += node[2]
            if node[1] in iset and node[0] in oset:
                cut += node[2]
        if cut > max_cut:
            max_cut_in, max_cut_out = iset.copy(), oset.copy()
            max_cut  = cut
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


def run_experiment(num_nodes):
    """Run an experiment, compute H, match against max-cut."""

    n, nodes = build_graph(num_nodes)
    max_cut  = compute_max_cut(n, nodes)
    #
    # These two lines are the basic implementation, where
    # a full matrix is being constructed. However, these
    # are diagonal, and can be constructed much faster.
    #  H       = graph_to_hamiltonian(n, nodes)
    #  diag    = H.diagonal()
    #
    diag     = graph_to_diagonal_h(n, nodes)
    min_val  = np.amin(diag)
    min_idx  = np.argmin(diag)
    if flags.FLAGS.graph:
        graph_to_dot(n, nodes, max_cut)

    # Results...
    if min_idx == max_cut:
        print('SUCCESS : {:+10.2f} |{}>'
              .format(np.real(diag[min_idx]),
                      bin(min_idx)[2:].zfill(n)))
    else:
        print('FAIL    : {:+10.2f} |{}>  '
              .format(np.real(diag[min_idx]),
                      bin(min_idx)[2:].zfill(n)),
              end='')
        print('Max-Cut: {:+10.2f} |{}>'
              .format(np.real(diag[max_cut]),
                      bin(max_cut)[2:].zfill(n)))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    for i in range(flags.FLAGS.iterations):
        run_experiment(flags.FLAGS.nodes)


if __name__ == '__main__':
  app.run(main)
