# python3
"""Example: Number set partitioning such set sum(A) == sum(B)."""

# ================================
# EXPERIMENTAL: Doens't work yet
# ================================

import math
import numpy as np
import random

from absl import app
from absl import flags

from src.lib import helper
from src.lib import ops

flags.DEFINE_integer('nmax', 10, 'Maximum number')
flags.DEFINE_integer('nnum',  4, 'Maximum number of set elements [1-nmax]')
flags.DEFINE_integer('iterations', 20, 'Number of experiments')


# Select numbers are collected in a list.
#
def select_numbers() -> list:
  """Build a graph of num nodes."""
  
  l = []

  for i in range(flags.FLAGS.nnum):
    num = random.randint(1, flags.FLAGS.nmax)
    while num in l:
      num = random.randint(1, flags.FLAGS.nmax)
    l.append(num)
  return l


def tensor_diag(n:int, num):
    """Construct a tensor product from diagonal matrices."""

    def tensor_product(w1:float, w2:float, diag):
        return [j for i in zip([x * w1 for x in diag],
                               [x * w2 for x in diag]) for j in i]

    diag = [1, 1]
    for i in range(n):
        if i == num:
            diag = tensor_product(i, -i, diag)
        else:
            diag = tensor_product(1, 1, diag)
    return diag


def set_to_diagonal_h(l, nmax) -> np.ndarray:
    """Construct diag(H)."""

    h = [0.0] * 2**(nmax+1)
    for num in l:
      diag = tensor_diag(nmax, num)
      for i in range(len(diag)):
          h[i] = h[i] + diag[i]
    return h


def compute_partition(l):
    """Compute (inefficiently) the paritiona."""

    solutions = []
    for bits in helper.bitprod(len(l)):
       # Collect in/out sets.
       iset = []
       oset = []
       for i in range(len(bits)):
           iset.append(l[i]) if bits[i] == 0 else oset.append(l[i])

       lsum = rsum = 0
       for i in iset:
           lsum = lsum + i
       for i in oset:
           rsum = rsum + i

       if lsum == rsum:
           solutions.append(bits)
    return solutions


def dump_solution(bits, l):
    res = ''
    iset = []
    oset = []
    for i in range(len(bits)):
        if bits[i] == 0:
            iset.append(f'{l[i]:2d}')
        else:
            oset.append(f'{l[i]:2d}')
    res += '+'.join(iset)
    res += ' == '
    res += '+'.join(oset)
    return res


def run_experiment():
    """Run an experiment, compute H, match against 0."""

    l = select_numbers()
    solutions = compute_partition(l)
      
    diag = set_to_diagonal_h(l, flags.FLAGS.nmax)
    for i in range(2**(flags.FLAGS.nmax+1)):
       if diag[i] == 0.0:
          print('Solution should exist - ', end='')
          if len(solutions):
             print('Found Solution:', dump_solution(solutions[0], l))
             return +1
          print('FALSE Positive')
          return -1
    if len(solutions):
        print('FALSE Negative')
        return -1

    return 0


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    print('This code is EXPERIMENTAL, does not work yet')
    n_ok = n_fail = 0
    for i in range(flags.FLAGS.iterations):
        ret = run_experiment()
        if ret > 0:
           n_ok += 1
        if ret < 0:
           n_fail += 1

    print(f'Ok: {n_ok}, Fail: {n_fail}, Success: {100.0 * n_ok / (n_ok + n_fail)}')

if __name__ == '__main__':
  app.run(main)
