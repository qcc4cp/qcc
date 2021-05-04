# python3
"""Example: Quantum Counting - Find # of solutions for Grover."""

import math
import random

from absl import app
import numpy as np

from src.lib import helper
from src.lib import ops
from src.lib import state

# This algorithm is an interesting combination of Grover's search
# and phase estimation. Unlike Grover Search, which attempts to find
# the one special x with f(x) = 1, Quantum Counting tells us
# for how many x's the function returns 1. How many special elements are
# in the dataset, or the function?
#
# In essence we cleverly construct a phase estimation circuit with
# the Grover operator as it's unitary.
#
# We will find the phase phi and from it we estimate the number
# M of solutions out of the N element solution space, with:
#
#    sin(phi/2) = sqrt(M/N)
#             M = N * sin(phi/2)^2


def make_f(d=3, solutions=1):
  """Construct function that will return 1 for 'solutions' bits."""

  num_inputs = 2**d
  answers = np.zeros(num_inputs, dtype=np.int32)

  for i in range(solutions):
    idx = random.randint(0, num_inputs - 1)

    # Avoid collisions.
    while answers[idx] == 1:
      idx = random.randint(0, num_inputs - 1)

    # Found proper index. Populate 'answer' array.
    answers[idx] = 1

  # The actual function just returns an array elements.
  #
  # pylint: disable=no-value-for-parameter
  def func(*bits):
    return answers[helper.bits2val(*bits)]

  # Return the function we just made.
  return func


def run_experiment(nbits_phase, nbits_grover, solutions) -> None:
  """Run full experiment for a given number of solutions."""

  # Building the Grover operator, see grover.py
  n = 2**nbits_grover
  zero_projector = np.zeros((n, n))
  zero_projector[0, 0] = 1
  op_zero = ops.Operator(zero_projector)

  f = make_f(nbits_grover, solutions)
  u = ops.OracleUf(nbits_grover + 1, f)

  # The state for the counting algorithm.
  # We reserve nbits for the phase estimation.
  # We also reserve nbits for the Oracle.
  # These numbers could be adjusted to achieve better
  # accuracy. Yet, this keeps the code a little bit simpler,
  # while trading off a few off-by-1 estimation errors.
  #
  # We also add the |1> for the Oracle.
  #
  psi = (state.zeros(nbits_phase) * state.zeros(nbits_grover) * state.ones(1))

  # Apply Hadamard to all the qubits.
  for i in range(nbits_phase + nbits_grover + 1):
    psi.apply(ops.Hadamard(), i)

  # Construct the Grover operator.
  reflection = op_zero * 2.0 - ops.Identity(nbits_grover)
  hn = ops.Hadamard(nbits_grover)
  inversion = hn(reflection(hn)) * ops.Identity()
  grover = inversion(u)

  # Now that we have the Grover operator, we have to perform
  # phase estimation. This loop is a copy from phase_estimation.py
  # with more comments there.
  #
  for idx, inv in enumerate(range(nbits_phase - 1, -1, -1)):
    u2 = grover
    for _ in range(idx):
      u2 = u2(u2)
    psi = ops.ControlledU(inv, nbits_phase, u2)(psi, inv)

  # Reverse QFT gives us the phase as a fraction of 2*Pi
  psi = ops.Qft(nbits_phase).adjoint()(psi)

  # Get the state with highest probability and compute the phase
  # as a binary fraction. Note that the probability increases
  # as M, the number of solutions, gets closer and closer to N,
  # the total mnumber of states.
  maxbits, maxprob = psi.maxprob()
  phi_estimate = (sum(maxbits[i] * 2**(-i - 1)
                      for i in range(nbits_phase)))

  # We know that after phase estimation, this holds:
  #
  #    sin(phi/2) = sqrt(M/N)
  #             M = N * sin(phi/2)^2
  #
  # Hence we can compute M. We keep the result to 2 digit to visualize
  # the errors. Note that the phi_estimate is a fraction of 2*PI, hence
  # the 1/2 in above formula cancels out against the 2 and we compute:
  M = round(n * math.sin(phi_estimate * math.pi)**2, 2)

  print('Estimate: {:.4f} prob: {:5.2f}% --> M: {:5.2f}, want: {:2d}'
        .format(phi_estimate, maxprob * 100.0, M, solutions))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for solutions in range(1, 6):
    run_experiment(7, 4, solutions)


if __name__ == '__main__':
  app.run(main)
