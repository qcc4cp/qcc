# python3
"""Example: Hamiltonian encoding and evolution. A few experiments."""

import random
from absl import app
import numpy as np

from src.lib import ops
from src.lib import state


def make_hermitian(a: ops.Operator):
  """Construct a Hermitian matrix from a given A."""

  if a.is_hermitian():
    return a

  # There are a few ways to make a Hermitian out of A. The first
  # trick is to make a block matrix of this form, which is of
  # twice the size of the original matrix.
  #
  b = ops.Operator(np.block([[np.zeros(a.shape), a],
                             [a.transpose().conjugate(), np.zeros(a.shape)]]))
  if not b.is_hermitian():
    raise AssertionError('Making 2*n Hermitian failed.')
  return b

  # Alternatively (disabled for now):
  # Another way is to simple compute A + A.transpose().conjugate()
  #
  # c = a + a.transpose().conjugate()
  # if not c.is_hermitian():
  #   raise AssertionError('Making A + A.T Hermitian failed.')
  # return c


def run_experiment(a):
  """Run a single experiment."""

  # We want to make A a Hamiltonian and for this purpose it has
  # to be made Hermitian.
  #
  a = make_hermitian(a)
  dim = a.shape[0]

  # Let's compute eigenvalues and eigenvectors:
  #
  lam, v = np.linalg.eig(a)

  # A is Hermitian with v being a basis for complex vectors in space dim x 1.
  #
  # This means than any dim x 1 state can be computed from this basis as:
  #   psi = gamma_0 v_0 + gamma_1 v_1 + ... + gamma_dim v_dim
  #
  # where we can compute:
  #   gamma_i = (psi^dagger * v_i)
  #
  # Let's try this out with a random complex state.
  #
  psi = state.State([complex(random.random(), random.random())
                     for _ in range(dim)]).normalize()
  print('Random complex state:', psi)

  # Let's compute gamma:
  #
  gamma = np.array([np.dot(psi.conj(), v[i]) for i in range(dim)])

  # Let's double check that we can construct psi from the basis v and gamma:
  #   psi = (gamma[0]^dagger * v[0]) + ... + (gamma[dim]^dagger * v[dim])
  #
  psi_new = np.zeros(dim)
  for i in range(dim):
    psi_new = psi_new + (gamma[i].conj() * v[i])
  if not np.allclose(psi, psi_new, atol=1e-5):
    raise AssertionError('Incorrect computation.')

  # Applying the Hamiltonian H_A to a state means:
  #    a' = exp(-i H_A t) a
  #       = exp(-i lam_0 t) gamma_0^dag v_0+...+exp(-i lam_3 t) gamma_3^dag v_3
  #
  def apply_hamiltonian(t: float):
    psi = np.zeros(dim)
    for i in range(dim):
      psi = psi + np.exp(-1j * lam[i] * t) * gamma[i].conj() * v[i]
    return psi

  # At time t = 0 we must get the same result as above:
  #
  psi_new = apply_hamiltonian(0)
  if not np.allclose(psi, psi_new, atol=1e-5):
    raise AssertionError('Incorrect computation.')

  # Print example evolutions:
  #
  for i in range(5):
    psi = apply_hamiltonian(0.2*i)
    print(f'Hamiltonian encoded at t = {0.2 * i:.1f}: ', end='')
    for i in range(dim):
      print(f'{psi[i]:.3f} ', end='')
    print()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Let's try a complex matrix:
  #
  a = ops.Operator([[2.0, -1/3 + 1j/8], [-1/3 - 1j/8, 1]])
  run_experiment(a)

  # Let's use the matrix that was described in:
  #   "Machine Learning with Quantum Computing" by
  #    Maria Schuld and Francesco Petruccione, page 118
  #
  a = ops.Operator([[0.073, -0.438], [0.730, 0.000]])
  run_experiment(a)

  # The numerical example from:
  #   "Step-by-Step HHL Algorithm Walkthrough..." by
  #    Morrell, Zaman, Wong
  # (which is a 2x2 Hermitian matrix)
  #
  a = ops.Operator([[1.0, -1/3], [-1/3, 1]])
  run_experiment(a)


if __name__ == '__main__':
  np.set_printoptions(precision=3)
  app.run(main)
