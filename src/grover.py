# python3
"""Example: Grover Algorithm."""

import math

from absl import app
import numpy as np

from src.lib import helper
from src.lib import ops
from src.lib import state


#
# This function can be used if there is only 1 solution,
# which is the simplest case. We keep it here, as it is
# referenced in the book. It is no longer used in this code.
#
def make_f1(d=3):
  """Construct function that will return 1 for only one bit string."""

  num_inputs = 2**d
  answers = np.zeros(num_inputs, dtype=np.int32)
  answer_true = np.random.randint(0, num_inputs)

  bit_string = format(answer_true, '0{}b'.format(d))
  answers[answer_true] = 1

  # pylint: disable=no-value-for-parameter
  def func(*bits):
    return answers[helper.bits2val(*bits)]

  return func


def make_f(d=3, solutions=1):
  """Construct function that will return 1 for 'solutions' bits."""

  num_inputs = 2**d
  answers = np.zeros(num_inputs, dtype=np.int32)

  for i in range(solutions):
    idx = np.random.randint(0, num_inputs - 1)

    # Avoid collisions.
    while answers[idx] == 1:
      idx = np.random.randint(0, num_inputs - 1)

    # Found proper index. Populate 'answer' array.
    answers[idx] = 1

  # The actual function just returns an array elements.
  def func(*bits):
    return answers[helper.bits2val(*bits)]

  # Return the function we just made.
  return func


def run_experiment(nbits, solutions) -> None:
  """Run full experiment for a given flavor of f()."""

  # Note that op_zero multiplies the diagonal elements of the operator by -1,
  # except for element [0][0]. This can be interpreted as "rotating around
  # the |00..)>" state. More pragmatically, multiplying this op_zero with
  # a Hadamard from the left and right gives a matrix of this form:
  #
  # 2/N-1   2/N    2/N    ...   2/N
  # 2/N     2N-1   2/N    ...   2/N
  # 2/N     2/N    2/N-1  ...   2/N
  # 2/N     2/N    2/N    ...   2/N-1
  #
  # Multiplying this matrix with a state vector computes exactly:
  #     2u - c_x
  # for every vector element c_x, with u being the mean over the
  # state vector. This is the defintion of inversion about the mean.
  #
  zero_projector = np.zeros((2**nbits, 2**nbits))
  zero_projector[0, 0] = 1
  op_zero = ops.Operator(zero_projector)

  # Make f and Uf. Note:
  # We reserve space for an ancilla 'y', which is unused in
  # Grover's algorithm. This allows reuse of the Deutsch Uf builder.
  #
  # We use the Oracle construction for convenience. It is rather
  # slow (full matrix) for larger qubit counts. Once can construct
  # a 'regular' function for the grover search algorithms, but this
  # function is different for each bitstring and that quickly gets
  # confusing.
  #
  f = make_f(nbits, solutions)
  uf = ops.OracleUf(nbits+1, f)

  # Build state with 1 ancilla of |1>.
  #
  psi = state.zeros(nbits) * state.ones(1)
  for i in range(nbits + 1):
    psi.apply1(ops.Hadamard(), i)

  # Build Grover operator, note Id() for the ancilla.
  # The Grover operator is the combination of:
  #    - phase inversion via the u unitary
  #    - inversion about the mean (see matrix above)
  #
  hn = ops.Hadamard(nbits)
  reflection = op_zero * 2.0 - ops.Identity(nbits)
  inversion = hn(reflection(hn)) * ops.Identity()
  grover = inversion(uf)

  # Number of Grover iterations
  #
  # There are at least 2 ways to compute the required number of iterations.
  #
  # 1) Most text books, eg., page 157 of Kaye, Laflamme and Mosca, find
  #    that the highest probability of finding the right results occurs
  #    after pi/4 sqrt(n) rotations.
  #
  # 2) For grover specifically, with 1 solution the iteration count is:
  #        int(math.pi / 4 * math.sqrt(n))
  #
  # 3) For grover with multiple solutions:
  #        int(math.pi / 4 * math.sqrt(n / solutions))
  #
  # 4) For amplitude amplification, it's the probability of good
  #    solutions, which is trivial with the Grover equal
  #    superposition  here:
  #        int(math.sqrt(n / solutions))
  #
  iterations = int(math.pi / 4 * math.sqrt(2**nbits / solutions))

  for _ in range(iterations):
    psi = grover(psi)

  # Measurement - pick element with higher probability.
  #
  # Note: We constructed the Oracle with n+1 qubits, to allow
  # for the 'xor-ancillary'. To check the result, we need to
  # ignore this ancilla.
  #
  maxbits, maxprob = psi.maxprob()
  result = f(maxbits[:-1])
  print('Got f({}) = {}, want: 1, #: {:2d}, p: {:6.4f}'
        .format(maxbits[:-1], result, solutions, maxprob))
  if result != 1:
    raise AssertionError('something went wrong, measured invalid state')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for nbits in range(3, 8):
    run_experiment(nbits, 1)

  for solutions in range(1, 9):
    run_experiment(7, solutions)


if __name__ == '__main__':
  app.run(main)
