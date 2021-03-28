# python3
"""Example: Max Cut Algorithm for bi-directional graph."""

import math
import numpy as np
import timeit

from absl import app
from absl import flags

from src.lib import helper
from src.lib import ops

flags.DEFINE_integer('nodes', 5, 'Number of graph nodes')


# Nodes are tuples: (from, to, edge weight).
#
def build_graph(num:int=0) -> (int, list):
    """Build core graph of 4 nodes."""

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


def graph_to_dot(n:int, nodes:list) -> None:
    """Convert graph to dot file."""

    print('graph {')
    # TODO: Color Max-Cut sets.
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
    """Compute adjacency matrix from graph."""

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


def compute_max_cut(n:int, nodes:list) -> None:
    """Compute (inefficiently) the max cut, exhaustively."""

    max_cut = -1000
    for bits in helper.bitprod(n):
        # Collect in/out sets.
        in_set = []
        out_set = []
        for i in range(len(bits)):
            in_set.append(i) if bits[i] == 0 else out_set.append(i)

        # Compute costs for this cut, record maximum.
        cut = 0
        for node in nodes:
            if node[0] in in_set and node[1] in out_set:
                cut += node[2]
            if node[1] in in_set and node[0] in out_set:
                cut += node[2]
        if cut > max_cut:
            max_cut_in, max_cut_out = in_set.copy(), out_set.copy()
            max_cut  = cut
            max_bits = bits

    state = bin(helper.bits2val(max_bits))[2:].zfill(n)
    print('Max Cut. Nodes: {}, Max H: {:.1f}, {}  -  {}, State: |{}>'
          .format(n, np.real(max_cut), max_cut_in, max_cut_out, state))
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
    graph_to_dot(n, nodes)
    H        = graph_to_hamiltonian(n, nodes)
    diag     = H.diagonal()
    min_val  = np.amin(diag)

    # Results...
    for i in range(len(diag)):
        marker = ''
        if diag[i] == min_val:
            marker += 'Minimum '
        if i == max_cut:
            marker += '= Max Cut'
        if len(marker):
            print(f'{np.real(diag[i]):+10.2f} |{bin(i)[2:].zfill(n)}> {marker}')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    for i in range(1):
        run_experiment(flags.FLAGS.nodes)

    #(ops.Identity() * ops.Identity() * ops.PauliZ()).dump('IIZ')
    #(ops.Identity() * ops.PauliZ() * ops.Identity()).dump('IZI')
    #(ops.Identity() * ops.PauliZ() * ops.PauliZ()).dump('IZZ')
    #(ops.PauliZ() * ops.Identity() * ops.Identity()).dump('ZII')


if __name__ == '__main__':
  app.run(main)
