# python3
"""Example: Number set partitioning such set sum(A) == sum(B)."""


# Based on this paper:
#   https://cds.cern.ch/record/467590/files/0010018.pdf
#
# For a set A of integers, can A be partitioned into
# two sets A1 and A2, such that:
#    sum(A1) == sum(A2)
#
# For this to work, sum(A) must not be odd.
# We should reach 100% consistent results.

import math
import numpy as np
import random

from absl import app
from absl import flags

from src.lib import helper
from src.lib import ops

flags.DEFINE_integer('nmax', 15, 'Maximum number')
flags.DEFINE_integer('nnum',  6,
                     'Maximum number of set elements [1-nmax]')
flags.DEFINE_integer('iterations', 20, 'Number of experiments')


def select_numbers(nmax, nnum) -> list:
  """Select nnum random, unique numbers in range 1..nmax."""

  while True:
    l = random.sample(range(1, nmax), nnum)
    if sum(l) % 2 == 0:
        return l


def tensor_diag(n:int, num):
    """Construct tensor product from diagonal matrices."""

    def tensor_product(w1:float, w2:float, diag):
        return [j for i in zip([x * w1 for x in diag],
                               [x * w2 for x in diag]) for j in i]

    diag = [1, -1] if num == 0 else [1, 1] 
    for i in range(1, n):
        if i == num:
            diag = tensor_product(i, -i, diag)
        else:
            diag = tensor_product(1, 1, diag)
    return diag


def set_to_diagonal_h(l, nmax) -> np.ndarray:
    """Construct diag(H)."""

    h = [0.0] * 2**nmax
    for num in l:
      diag = tensor_diag(nmax, num)
      for i in range(len(diag)):
          h[i] = h[i] + diag[i]
    return h


def compute_partition(l):
    """Compute paritions that add up."""

    solutions = []
    for bits in helper.bitprod(len(l)):
       iset = []
       oset = []
       for i in range(len(bits)):
           (iset.append(l[i]) if bits[i] == 0 else
            oset.append(l[i]))
       if sum(iset) == sum(oset):
           solutions.append(bits)
    return solutions


def dump_solution(bits, l):
    iset = []
    oset = []
    for i in range(len(bits)):
        (iset.append(f'{l[i]:d}') if bits[i] == 0  else
         oset.append(f'{l[i]:d}'))
    return '+'.join(iset) + ' == ' + '+'.join(oset)


def run_experiment():
    """Run an experiment, compute H, match against 0."""

    nmax = flags.FLAGS.nmax
    l = select_numbers(nmax, flags.FLAGS.nnum)
    solutions = compute_partition(l)

    diag = set_to_diagonal_h(l, nmax)

    non_zero = np.count_nonzero(diag)
    if non_zero != 2**nmax:
       print('Solution should exist...', end='')
       if len(solutions):
           print(' Found Solution:', dump_solution(solutions[0], l))
           return True
       raise AssertionError('False positive found')
    if len(solutions):
       raise AssertionError('False negative found')
    return False


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    for i in range(flags.FLAGS.iterations):
        ret = run_experiment()


if __name__ == '__main__':
  app.run(main)
