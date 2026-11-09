# python3
"""Tests for the Grover search example (grover.py)."""

import math
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from src import grover
from src.lib import helper
from src.lib import state


def marked_probability(nbits: int, targets) -> float:
  """Run grover.run_experiment() and return P(any marked state).

  The targets are fixed, instead of randomly selected. The final state is
  captured, so that the test can check the probability of the marked states
  and not just which bit string happens to have the highest probability.
  With ties, as for 2 qubits and 1 solution before the fix (all 4 states
  at 25%), the most likely bit string alone proves nothing.

  Args:
    nbits: Number of search qubits.
    targets: Values for which the oracle returns 1.

  Returns:
    The probability of measuring one of the targets.
  """

  captured = []
  original_maxprob = state.State.maxprob

  def observe(psi):
    captured.append(np.asarray(psi).copy())
    return original_maxprob(psi)

  def oracle(bits):
    return int(helper.bits2val(bits) in targets)

  with mock.patch.object(grover, 'make_f', return_value=oracle), \
      mock.patch.object(state.State, 'maxprob', observe):
    try:
      grover.run_experiment(nbits, len(targets))
    except AssertionError:
      # Reported via the probability check below, which is more specific.
      pass

  # The state has one trailing ancilla qubit for the oracle, sum it out.
  probs = (np.abs(captured[-1]) ** 2).reshape(2**nbits, 2).sum(axis=1)
  return float(sum(probs[t] for t in targets))


def expected_probability(nbits: int, nsolutions: int) -> float:
  """Textbook probability sin^2((2k+1) theta) after k Grover iterations."""

  num = 2**nbits
  iterations = int(math.pi / 4 * math.sqrt(num / nsolutions))
  theta = math.asin(math.sqrt(nsolutions / num))
  return math.sin((2 * iterations + 1) * theta) ** 2


class GroverTest(parameterized.TestCase):

  @parameterized.parameters(
      (2, (0,)),
      (2, (1,)),
      (2, (2,)),
      (2, (3,)),
      (3, (7,)),
      (4, (5,)),
      (5, (0,)),
      (5, (31,)),
      # Several solutions.
      (4, (1, 2)),
      (6, (3, 40, 41, 63)),
  )
  def test_marked_state_probability(self, nbits, targets):
    """Oracle first, then inversion about the mean, per iteration."""

    got = marked_probability(nbits, targets)
    want = expected_probability(nbits, len(targets))
    # The simulator computes in single precision, hence the tolerance.
    self.assertAlmostEqual(
        got, want, delta=1e-4,
        msg=f'{nbits} qubits, targets {targets}: P(marked) = {got:.6f}, '
        f'want sin^2((2k+1)theta) = {want:.6f}')

  def test_two_qubits_is_exact(self):
    """With N = 4 and 1 solution, a single iteration finds it with P = 1."""

    for target in range(4):
      self.assertAlmostEqual(
          marked_probability(2, (target,)), 1.0, delta=1e-4)


if __name__ == '__main__':
  absltest.main()
