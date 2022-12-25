# python3
"""Example: Minimunm Finding via Grover's Algorithm."""

import math

from absl import app
import numpy as np

from src.lib import circuit
from src.lib import helper
from src.lib import ops
from src.lib import state


# This is the implementation of a quantum minimum finding algorithm
# after Durr and Hoyer, "A Quantum Algorithm for Finding the Minimum"
# (https://arxiv.org/abs/quant-ph/9607014).
#
# To establish a quantum advantage, the algorithm makes several
# pretty rough assumptions and simplifications. Here is the algorithm
# (and implementation) to the best of my understanding:
#
# 1) First we create a set of random numbers in function 'get_distro'.
#    The goal is to find the smallest of these numbers. Since we use
#    Grover, the number of solutions should be small'ish compared to
#    the available space to be able to find a solution with good
#    probability.
#
# 2) Simular to 'regular' Grover, as shown in grover.py, we now construct
#    a function in 'make_f' that returns a 1 for every number in the set.
#    However, there is a little twist: We only mark a value if it is
#    smaller than a given maximum.
#
# 3) The thirst step is now identical to what is found in grover.py
#    We simply use Grover to find one of the marked items. Using our
#    oracle approach, this works like a charm.
#
# 4) As for measurement, we simulate measurement by simply picking one
#    of the found elements - they all have the higher probability than
#    the other states. We now use this found element as a new upper bound
#    for marking elements.
#
# 5) Now, go back to step 2 and use the newly found upper bound as the
#    new maximum for marking numbers. This is sort of like binary search
#    towards the smallest element, except by using a random lower point
#    to restart the search.
#
# 6) As soon as we hit the smallest number in the set, we stop - we
#    have found the smallest element.
#
# Of course, there are concerns.
#    - the oracle has to be constructed. However, the assumption in the
#      paper is that this is free and comes at no cost.
#    - we terminate when we find the smallest number. This requires
#      knowledge of the smallest number.
#    - we must know how many numbers remain below a current maximum
#      in order to adjust the Grover iteration number.
#
# The last two points could be solved via quantum counting or perhaps
# other mechanisms to find when there is no more solution to be marked.
#
# All together, the description of the algorithm look quite cryptic and
# I have yet to find a working version on the interwebs. This implementation
# - assuming I understood the algorithm correctly - works.
#


def get_distro(min_value: int, max_value: int, num_vals: int):
  """Create 'num' random numbers from ranging from min to max."""

  return sorted(np.random.choice(np.arange(min_value, max_value), num_vals))


def make_f(d: int, numbers: list, max_value: int):
  """Construct function that will return 1 for each number up to max."""

  num_inputs = 2**d
  answers = np.zeros(num_inputs, dtype=np.int8)

  for _, val in enumerate(numbers):
    if val >= max_value:
      continue
      
    # Populate 'answer' array.
    answers[val] = 1

  # The actual function just returns an array elements.
  def func(*bits):
    return answers[helper.bits2val(*bits)]

  # Return the function we just made.
  return func


def run_experiment(nbits, numbers: list, max_value: int, solutions: int) -> None:
  """Run oracle-based experiment."""

  # The following is commented extensively in grover.py
  #
  zero_projector = np.zeros((2**nbits, 2**nbits))
  zero_projector[0, 0] = 1
  op_zero = ops.Operator(zero_projector)

  f = make_f(nbits, numbers, max_value)
  uf = ops.OracleUf(nbits+1, f)

  psi = state.zeros(nbits) * state.ones(1)
  for i in range(nbits + 1):
    psi.apply1(ops.Hadamard(), i)

  hn = ops.Hadamard(nbits)
  reflection = op_zero * 2.0 - ops.Identity(nbits)
  inversion = hn(reflection(hn)) * ops.Identity()
  grover = inversion(uf)

  iterations = int(math.pi / 4 * math.sqrt(2**nbits / solutions))

  for _ in range(iterations):
    psi = grover(psi)

  # Measurement - pick elements with highest probability.
  # For n marked numbers there should be n results.
  #
  maxbits, maxprob = psi.maxprob()
  results = []
  for idx in range(len(psi)):
    if psi[idx] * psi[idx].conj() >= maxprob - 0.01:
       bits = helper.val2bits(idx, 8)[:-1]
       val = helper.bits2val(bits)
       results.append(val)
       
  # Compute new max limit by randomly selecting one of the results,
  # this simulating an actual, random measurement result:
  #
  new_max = np.random.choice(results)
  print(' -> New Max:', new_max)
  result = f(helper.val2bits(new_max, nbits))
  if result != 1:
    raise AssertionError('something went wrong, measured invalid state')

  # Return the newly found upper limit for the search.
  #
  return new_max


def run_search(marked_numbers: int, qubits: int):
  """Run a single search for the minimum."""
  
  numbers = get_distro(3, 1 << qubits, marked_numbers)
  
  max_value = 1 << qubits
  print('Find minimum in:', numbers)
  
  for i in range(qubits):
     max_value = run_experiment(qubits, numbers, max_value, marked_numbers)
     
     if not max_value in numbers:
       raise AssertionError('*** Grover search failed')

     # Here we cheat a little bit. We terminate the search as we have
     # found the smallest number. Without this shortcut, we would somehow
     # have to identify that no solution has been marked, eg., via
     # quantum counting.
     #
     if max_value == numbers[0]:
       print(f'*** SUCCESS, found smallest element, {i + 1} iterations', max_value)
       break

     # In order to adjust the number of Grover iterations, we must know
     # how many marked elements there are. This is also a bit of cheating.
     # To make this fully quantum, we would have to employ techniques such
     # as quantum counting here as well.
     #
     marked = 0
     for i in range(marked_numbers):
       if max_value > i:
         marked += 1
     marked_numbers = marked - 1


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for i in range(10):
    run_search(10, 8)


if __name__ == '__main__':
  app.run(main)