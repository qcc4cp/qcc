# python3
"""Example: Inversion Test to estimate dot product between two states."""

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


def inversion_test():
  """Perform Inversion Test."""

  # The inversion test allows keeping the number of qubits to a minimum
  # when trying to determine the overlap between two states. However, it
  # requires a precise way to generate states a and b, as well as the
  # adjoint for one of them.
  #
  # If we have operators A and B (similar to the Hadamard Test) producing:
  #     A |0> = a
  #     B |0> = b
  # To determine the overlap between a and be (<a|b>), we run:
  #     B_adjoint A |0>
  # and determine the probability p0 of measuring |0>. p0 is an
  # a precise estimate for <a|b>.

  # pylint: disable=invalid-name
  A, a0, a1 = make_rand_operator()
  B, b0, b1 = make_rand_operator()

  # For the inversion test, we will need B^\dagger:
  Bdag = B.adjoint()

  # Compute the dot product <a|b>:
  dot = np.dot(np.array([a0, a1]).conj(), np.array([b0, b1]))

  # Here is the inversion test. We run B^\dagger A |0> and find
  # the probability of measuring |0>:
  qc = circuit.qc('Hadamard test - initial state construction.')
  qc.reg(1, 0)
  qc.apply1(A, 0)
  qc.apply1(Bdag, 0)

  # The probability amplitude of measuring |0> should be the
  # same value as the dot product.
  p0, _ = qc.measure_bit(0, 0)
  if not np.allclose(dot.conj() * dot, p0):
    raise AssertionError('Incorrect inner product estimation')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  iterations = 1000
  print(f'Perform {iterations} random inversion tests.')

  for _ in range(iterations):
    inversion_test()
  print('Success')


if __name__ == '__main__':
  app.run(main)
