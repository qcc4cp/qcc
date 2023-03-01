# python3
"""Example: Hadamard Test for two states."""

import cmath

from absl import app
import numpy as np
from scipy.stats import unitary_group

from src.lib import circuit
from src.lib import ops
from src.lib import state


def make_rand_operator():
  """Make a unitary operator U, derive u0, u1."""

  # We think of operators as the following,
  # so that multiplication with |0> extracts the state
  # values (u0, u1):
  #   | u0   u2  | | 1 |    | u0 |
  #   | u1   u3  | | 0 |  = | u1 |
  #
  # pylint: disable=invalid-name
  U = ops.Operator(unitary_group.rvs(2))
  if not U.is_unitary():
    raise AssertionError('Error: Generated non-unitary operator')
  psi = U(state.bitstring(0))
  u0 = psi[0]
  u1 = psi[1]
  return (U, u0, u1)


def hadamard_test():
  """Perform Hadamard Test."""

  # pylint: disable=invalid-name
  A, a0, a1 = make_rand_operator()
  B, b0, b1 = make_rand_operator()

  # ======================================
  # Step 1: Verify P(|0>) = 2 Re(<a|b>)
  # ======================================

  # Construct the desired end state psi as an explicit expression.
  #    psi = 1/sqrt(2)(|0>|a> + |1>|b>)
  psi = (1 / cmath.sqrt(2) *
         (state.bitstring(0) * state.State([a0, a1]) +
          state.bitstring(1) * state.State([b0, b1])))

  # Let's see how to make this state with a circuit.
  qc = circuit.qc('Hadamard test - initial state construction.')
  qc.reg(2, 0)
  qc.h(0)
  qc.applyc(A, [0], 1)  # Controlled-by-0
  qc.applyc(B, 0, 1)  # Controlled-by-1

  # The two states should be identical!
  if not np.allclose(qc.psi, psi):
    raise AssertionError('Incorrect result')

  # Now let's apply a final Hadamard to the ancilla.
  qc.h(0)

  # At this point, this inner product estimation should hold:
  #  P(|0>) = 1/2 + 1/2 Re(<a|b>)
  # Or
  #  2 * P(|0>) - 1 = Re(<a|b>)
  #
  # Let's verify...
  dot = np.dot(np.array([a0, a1]).conj(), np.array([b0, b1]))
  p0 = qc.psi.prob(0, 0) + qc.psi.prob(0, 1)
  if not np.allclose(2 * p0 - 1, dot.real, atol=1e-6):
    raise AssertionError('Incorrect inner product estimation')

  # ======================================
  # Step 2: Verify P(|1>) = 2 Im(<a|b>)
  # ======================================

  # Now let's try the same to get to the imaginary parts.
  #
  #    psi = 1/sqrt(2)(|0>|a> - i|1>|b>)
  #
  psi = (1 / cmath.sqrt(2) *
         (state.bitstring(0) * state.State([a0, a1]) -
          1.0j * state.bitstring(1) * state.State([b0, b1])))

  # Let's see how to make this state with a circuit.
  #
  qc = circuit.qc('Hadamard test - initial state construction.')
  qc.reg(2, 0)
  qc.h(0)
  qc.sdag(0)
  qc.applyc(A, [0], 1)  # Controlled-by-0
  qc.applyc(B, 0, 1)  # Controlled-by-1

  # The two states should be identical!
  #
  if not np.allclose(qc.psi, psi):
    raise AssertionError('Incorrect result')

  # Now let's apply a final Hadamard to the ancilla.
  qc.h(0)

  # At this point, this inner product estimation should hold:
  #  P(|0>) = 1/2 + 1/2 Im(<a|b>)
  # Or
  #  2 * P(|0>) - 1 = Im(<a|b>)
  #
  # Let's verify...
  dot = np.dot(np.array([a0, a1]).conj(), np.array([b0, b1]))
  p0 = qc.psi.prob(0, 0) + qc.psi.prob(0, 1)
  if not np.allclose(2 * p0 - 1, dot.imag, atol=1e-6):
    raise AssertionError('Incorrect inner product estimation')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  iterations = 1000
  print(f'Perform {iterations} random hadamard tests.')

  for _ in range(iterations):
    hadamard_test()
  print('Success')

if __name__ == '__main__':
  app.run(main)
