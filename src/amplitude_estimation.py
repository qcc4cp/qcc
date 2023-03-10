# python3
"""Example: Amplitude Estimation."""

import math
import random
from typing import List

from absl import app
import numpy as np
from src.lib import helper
from src.lib import ops
from src.lib import state


# Amplitude estimation (AE) is a generalization of the counting
# algorithm (or, rather, counting is a special case of AE).
#
# In counting, we have a state in equal superposition
# (achieved via Hadamard^\otimes(nbits) where some of the states
# are 'good' (solutions) and the rest are 'bad' (non-solutions).
#
# In the general case, the probabilities for each state can be
# different. A general algorithm A generates a state. Then, similar
# to grover, one can think of the space that the orthogonal good
# and bad states span as:
#     \psi = \alpha \psi_{good} + \beta \psi_{bad}
#
# AE estimates this amplitude \alpha.


def make_f(nbits: int, solutions: List[int]):
  """Construct function that will return 1 for 'solutions' bits."""

  answers = np.zeros(1 << nbits, dtype=np.int32)
  answers[solutions] = 1
  return lambda bits: answers[helper.bits2val(bits)]


def run_experiment(nbits_phase: int,
                   nbits_grover: int,
                   algo: ops.Operator,
                   solutions: List[int]) -> float:
  """Run full experiment for a given A and set of solutions."""

  # The state for the AE algorithm.
  # We reserve nbits_phase for the phase estimation.
  # We reserve nbits_grover for the oracle.
  # We also add the |1> for the oracle's y value.
  #
  # These numbers can be adjusted to achieve various levels
  # of accuracy.
  psi = state.zeros(nbits_phase + nbits_grover) * state.ones(1)

  # Apply Hadamard to all the qubits.
  psi = ops.Hadamard(nbits_phase + nbits_grover + 1)(psi)

  # Construct the Grover operator. First phase invesion via Oracle.
  f = make_f(nbits_grover, solutions)
  u = ops.OracleUf(nbits_grover + 1, f)

  # Reflection over mean.
  op_zero = ops.ZeroProjector(nbits_grover)
  reflection = op_zero * 2.0 - ops.Identity(nbits_grover)

  # Now construct the combined Grover operator.
  inversion = algo.adjoint()(reflection(algo)) * ops.Identity()
  grover = inversion(u)

  # Now that we have the Grover operator, we apply phase estimation.
  psi = ops.PhaseEstimation(grover, psi, nbits_phase, nbits_phase)

  # Reverse QFT gives us the phase as a fraction of 2*pi.
  psi = ops.Qft(nbits_phase).adjoint()(psi)

  # Get the state with highest probability and estimate a phase.
  maxbits, _ = psi.maxprob()
  ampl = np.sin(np.pi * helper.bits2frac(maxbits[:nbits_phase]))

  print('  AE: ampl: {:.2f} prob: {:5.1f}% {}/{} solutions ({})'
        .format(ampl, ampl * ampl * 100, len(solutions),
                1 << nbits_grover, solutions))
  return ampl


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('Amplitude Estimation...')

  # Equal superposition.
  print('Algorithm: Hadamard (equal superposition)')
  algorithm = ops.Hadamard(3)
  for nsolutions in range(9):
    ampl = run_experiment(7, 3, algorithm,
                          random.sample(range(2**3), nsolutions))
    if not math.isclose(ampl, np.sqrt(nsolutions / 2**3), abs_tol=0.02):
      raise AssertionError('Incorrect AE.')

  # Make a somewhat random algorithm (and state).
  print('Algorithm: Random (unequal superposition), single solution')
  i1 = ops.Identity(1)
  algorithm = (ops.Hadamard(3) @
               (ops.RotationY(random.random()/2) * i1 * i1) @
               (i1 * ops.RotationY(random.random()/2) * i1) @
               (i1 * i1 * ops.RotationY(random.random()/2)))
  psi = algorithm(state.zeros(3))
  for i in range(len(psi)):
    ampl = run_experiment(7, 3, algorithm, [i])
    if not np.allclose(ampl, psi[i], atol=0.02):
      raise AssertionError('Incorrect AE.')

  # Accumulative amplitude computation.
  print('Algorithm: Random (unequal superposition), multiple solutions')
  for i in range(len(psi) + 1):
    ampl = run_experiment(7, 3, algorithm, [i for i in range(i)])
    if not np.allclose(ampl, np.sqrt(sum([p*p.conj() for p in psi[0:i]])),
                       atol=0.02):
      raise AssertionError('Incorrect AE.')


if __name__ == '__main__':
  app.run(main)
