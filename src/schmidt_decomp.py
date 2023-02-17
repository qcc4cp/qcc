# python3
"""Example: Schmidt Decomposition."""


import random
from absl import app
import numpy as np

from src.lib import ops
from src.lib import state


# The Schmidt Decomposition of a (2-qubit) bipartite system (quantum state)
# states that for a state \psi from a Hilbert space:
#
#   H_1 \otimes H_2
#
# We can find new bases:
#   {u_0, u_1, ..., u_{n-1} \elem H_1
#   {v_0, v_1, ..., v_{n-1} \elem H_2
#
# such that:
#   \psi = \sum_i \sqrt(\alpha} u_i \otimes v_i
#
# Additionally, adding up the \alpha will result in 1.0.
#
# This method can be used for testing for entanglement. A state is
# separable only if the number of non-zero coefficients \alpha is
# exactly 1, else the state is entangled.
#
# More information is available online, eg:
#    https://en.wikipedia.org/wiki/Schmidt_decomposition


def compute_eigvals(psi: state.State, expected_nonzero: int, tolerance: float):
  """Compute the eigenvalues for the individial substates."""

  # To find the factors \alpha and the new bases,
  # we trace out the subspaces and find their eigenvalues.
  #
  rho = psi.density()

  rho0 = ops.TraceOut(rho, [1])
  eigvals0 = np.linalg.eigvalsh(rho0)

  rho1 = ops.TraceOut(rho, [0])
  eigvals1 = np.linalg.eigvalsh(rho1)

  # The set of eigenvalues must be identical between the two sub states.
  #
  if not np.allclose(eigvals0, eigvals1, atol=1e06):
    raise AssertionError('Invalid set of eigenvalues.')

  # The eigenvalues must add up to 1.0.
  #
  if not np.allclose(np.sum(eigvals0), 1.0):
    raise AssertionError('Eigenvalues do not add up to 1.0')

  # Count the number of non-zero eigenvalues and match against expected.
  #
  nonzero = np.sum(eigvals0 > tolerance)
  if nonzero != expected_nonzero:
    print(f'\t\tCase of unstable math: {eigvals0[0]:.4f}, {eigvals0[1]:.4f}')

  # Construct the state from the eigenvalues and the new bases
  # which we derive via SVD. Then we check whether the new state
  # matches the original state.
  #
  a0, d0, _ = np.linalg.svd(rho0)
  a1, _, _ = np.linalg.svd(rho1)
  newpsi = (np.sqrt(d0[0]) * np.kron(a0[:, 0], a1[0, :]) +
            np.sqrt(d0[1]) * np.kron(a0[:, 1], a1[1, :]))
  if not np.allclose(psi, newpsi, atol=1e-3):
    raise AssertionError('Found incorrect Schmidt basis.')

  return eigvals0


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('Schmidt Decomposition and Entanglement Test.')

  iterations = 1000
  random.seed(7)

  # Test a number of separable states.
  #
  print('\tSchmidt Decomposition for seperable states.')
  for _ in range(iterations):
    psi = state.qubit(random.random()) * state.qubit(random.random())

    # States are separable if they have only 1 non-zero eigenvalue.
    compute_eigvals(psi, 1, 1e-3)

  # Test a number of entangled states.
  #
  print('\tSchmidt Decomposition for entangled states.')
  for _ in range(iterations):
    psi = state.bitstring(0, 0)
    psi = ops.Hadamard()(psi)
    angle = random.random() * np.pi
    if abs(angle) < 1e-5:
      continue
    psi = ops.ControlledU(0, 1, ops.RotationY(angle))(psi)

    # For entangled 2-qubit states we expect 2 non-zero eigenvalues.
    compute_eigvals(psi, 2, 1e-9)

  # Maximally entangled state.
  #
  print('\tSchmidt Decomposition for max-entangled state.')
  psi = state.bitstring(0, 0)
  psi = ops.Hadamard()(psi)
  psi = ops.Cnot()(psi)
  eigv = compute_eigvals(psi, 2, 1e-9)
  if abs(eigv[0] - eigv[1]) > 0.001:
    raise AssertionError('Incorrect computation for max-entangled state.')

  print('Success')


if __name__ == '__main__':
  app.run(main)
