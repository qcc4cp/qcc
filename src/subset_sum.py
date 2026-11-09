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

import random
from typing import List

from absl import app
from absl import flags
import numpy as np

from src.lib import helper

flags.DEFINE_integer('nmax', 15, 'Maximum number')
flags.DEFINE_integer('nnum', 6,
                     'Maximum number of set elements [1-nmax]')
flags.DEFINE_integer('iterations', 20, 'Number of experiments')


def select_numbers(nmax: int, nnum: int) -> List[int]:
  """Select nnum random, unique numbers in range 1 to nmax."""

  # With all numbers selected the sum is fixed. If it's odd, we would loop
  # forever below.
  if nnum == nmax and (nmax * (nmax + 1) // 2) % 2:
    raise ValueError(f'No even sum possible with all numbers 1..{nmax}.')
  while True:
    sample = random.sample(range(1, nmax + 1), nnum)
    if sum(sample) % 2 == 0:
      return sample


def tensor_diag(n: int, num: int):
  """Construct tensor product from diagonal matrices.

  Each number has its own qubit, the number k being represented by qubit
  k - 1 of n qubits. Hence, the numbers 1 to n can be represented.

  Args:
    n: The number of qubits.
    num: The number to construct the diagonal for, 1 <= num <= n.

  Returns:
    The diagonal of num * Z on the qubit of num, tensored with the identity
    on all other qubits.
  """

  def tensor_product(w1: float, w2: float, diag):
    # pylint: disable=g-complex-comprehension
    return [j for i in zip([x * w1 for x in diag],
                           [x * w2 for x in diag]) for j in i]

  # A number that doesn't have a qubit would silently be ignored.
  assert 1 <= num <= n, 'Invalid input, expected 1 <= num <= n.'
  diag = [1]
  for i in range(1, n + 1):
    if i == num:
      diag = tensor_product(i, -i, diag)
    else:
      diag = tensor_product(1, 1, diag)
  return diag


def set_to_diagonal_h(num_list: List[int], nbits: int) -> List[float]:
  """Construct diag(H) for nbits qubits, which can hold numbers 1..nbits."""

  h = [0.0] * 2**nbits
  for num in num_list:
    diag = tensor_diag(nbits, num)
    for idx, val in enumerate(diag):
      h[idx] += val
  return h


def compute_partition(num_list: List[int]):
  """Compute paritions that add up."""

  solutions = []
  for bits in helper.bitprod(len(num_list)):
    iset = []
    oset = []
    for idx, val in enumerate(bits):
      if val == 0:
        iset.append(num_list[idx])
      else:
        oset.append(num_list[idx])
    if sum(iset) == sum(oset):
      solutions.append(bits)
  return solutions


def dump_solution(bits: List[int], num_list: List[int]):
  """Simply print a solution."""

  iset = []
  oset = []
  for idx, val in enumerate(bits):
    if val == 0:
      iset.append(f'{num_list[idx]:d}')
    else:
      oset.append(f'{num_list[idx]:d}')
  return '+'.join(iset) + ' == ' + '+'.join(oset)


def run_experiment(num_list: List[int]) -> bool:
  """Run an experiment, compute H, match against 0."""

  nmax = flags.FLAGS.nmax
  if not num_list:
    num_list = select_numbers(nmax, flags.FLAGS.nnum)
  solutions = compute_partition(num_list)

  # Each number needs a qubit. If a caller passes numbers larger than nmax,
  # we extend the register, instead of silently ignoring these numbers.
  nbits = max(nmax, max(num_list))
  diag = set_to_diagonal_h(num_list, nbits)

  non_zero = np.count_nonzero(diag)
  if non_zero != 2**nbits:
    print('  Solution should exist...', end='')
    if solutions:
      print(' Found Solution:',
            dump_solution(solutions[0], num_list))
      return True
    assert False, 'False positive found.'

  print('  No Solution Found.', sorted(num_list))
  assert not solutions, 'False negative found.'
  return False


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print(f'Test random sets [1..{flags.FLAGS.nmax}]')
  for _ in range(flags.FLAGS.iterations):
    run_experiment(None)

  # A few negative tests.
  print('Test known-negative sets...')
  sets = [
      [1, 2, 3, 7],
      [1, 3, 5, 10],
      [2, 7, 8, 10, 12, 13],
      [1, 6, 8, 9, 12, 14],
      [3, 8, 9, 11, 13, 14],
      [7, 9, 12, 14, 15, 17],
  ]
  for s in sets:
    if run_experiment(s):
      raise AssertionError('Incorrect Classification')


if __name__ == '__main__':
  app.run(main)
