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


def make_f(d: int = 3, nsolutions: int = 1):
  """Construct function that will return 1 for 'solutions' bits."""

  answers = np.zeros(1 << d, dtype=np.int32)
  solutions = random.sample(range(1 << d), nsolutions)
  answers[solutions] = 1
  return lambda bits: answers[helper.bits2val(bits)]


def run_experiment(nbits_phase: int, nbits_grover: int, solutions: int) -> None:
  """Run full experiment for a given number of solutions."""

  # The state for the counting algorithm.
  # We reserve nbits for the phase estimation.
  # We also reserve nbits for the oracle.
  # These numbers could be adjusted to achieve better
  # accuracy. Yet, this keeps the code a little bit simpler,
  # while trading off a few off-by-1 estimation errors.
  #
  # We also add the |1> for the oracle.
  #
  psi = state.zeros(nbits_phase + nbits_grover) * state.ones(1)

  # Apply Hadamard to all the qubits.
  for i in range(nbits_phase + nbits_grover + 1):
    psi.apply1(ops.Hadamard(), i)

  # Construct the Grover operator. First phase inversion via Oracle.
  f = make_f(nbits_grover, solutions)
  u = ops.OracleUf(nbits_grover + 1, f)

  # Reflection over mean.
  op_zero = ops.ZeroProjector(nbits_grover)
  reflection = op_zero * 2.0 - ops.Identity(nbits_grover)

  # Now construct the combined Grover operator, using
  # Hadamards as the 'algorithm' (equal superposition).
  hn = ops.Hadamard(nbits_grover)
  inversion = hn(reflection(hn)) * ops.Identity()
  grover = inversion(u)

  # Now that we have the Grover operator, we have to perform
  # phase estimation. This loop is a copy from phase_estimation.py
  # with more comments there.
  cu = grover
  for inv in reversed(range(nbits_phase)):
    psi = ops.ControlledU(inv, nbits_phase, cu)(psi, inv)
    cu = cu(cu)

  # Reverse QFT gives us the phase as a fraction of 2*pi.
  psi = ops.Qft(nbits_phase).adjoint()(psi)

  # Get the state with highest probability and compute the phase
  # as a binary fraction. Note that the probability decreases
  # as M, the number of solutions, gets closer and closer to N,
  # the total mnumber of states.
  maxbits, maxprob = psi.maxprob()
  phi_estimate = helper.bits2frac(maxbits)

  # We know that after phase estimation, this holds:
  #    sin(phi/2) = sqrt(M/N)
  #             M = N * sin(phi/2)^2
  # Hence we can compute M. We keep the result to 2 digit to visualize
  # the errors. Note that the phi_estimate is a fraction of 2*PI, hence
  # the 1/2 in above formula cancels out against the 2 and we compute:
  m = round(2**nbits_grover * math.sin(phi_estimate * math.pi) ** 2, 2)

  print(
      f'Estimate: {phi_estimate:.4f} prob: {maxprob * 100.0:5.2f}% '
      f'--> m: {m:5.2f}, want: {solutions:2d}'
  )
  if not np.allclose(np.round(m), solutions):
    raise AssertionError('Incorrect result.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for solutions in range(1, 6):
    run_experiment(7, 4, solutions)


if __name__ == '__main__':
  app.run(main)
