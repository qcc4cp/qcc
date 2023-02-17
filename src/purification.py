# python3
"""Example: Purification."""


# State purification is the complement to Schmidt decomposition.
# Given a possibly mixed state with n qubits, purification creates
# a state with 2n qubits which is pure.
#
# This code is based on a version provided by Michael Broughton
# (Thank you so much!)
#


from absl import app
import numpy as np

from src.lib import bell
from src.lib import ops
from src.lib import state


def purify(rho: ops.Operator, nbits: int):
  """Purify a quantum state / density matrix."""

  rho_eig_val, rho_eig_vec = np.linalg.eig(rho)

  # Use stinespring dilation.
  # We know
  #    rho = sum_k pk |psi_k><psi_k|
  #
  # Purification is given by
  #    rho = sum_k sqrt(pk) |psi_k> tensor |psi_k>
  #
  # There are two (equivalent) ways to implement this:
  #
  # Version 1:
  #
  psi1 = np.zeros((2**(nbits * 2)), dtype=np.complex128)
  for i in range(len(rho_eig_val)):
    psi1 += (np.sqrt(rho_eig_val[i]) *
             np.kron(rho_eig_vec[:, i], rho_eig_vec[:, i]))

  # Version 2 using einsum's:
  #
  psi2 = np.einsum('k,ki,kj->ij', np.sqrt(rho_eig_val),
                   rho_eig_vec.T, rho_eig_vec.T).reshape(-1)

  if not np.allclose(psi1, psi2):
    raise AssertionError('Something wrong with purification.')

  # Verify the original reduced density matrix with the method
  # used in quantum_pca.py:
  #
  reduced = np.dot(psi1.reshape((2**nbits, 2**nbits)),
                   psi1.reshape((2**nbits, 2**nbits)).transpose())
  if not np.allclose(rho, reduced):
    raise AssertionError('Something wrong with reduced density')

  # Another way to compute the reduced density matrix:
  #
  reduced = state.State(psi1).density()
  reduced = (ops.TraceOut(rho,
                          [x for x in range(int(nbits),
                                            int(nbits*2))]) / (2**nbits))

  if not np.allclose(rho, reduced):
    raise AssertionError('Something wrong with reduced density')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('State purification(s).')

  # This is the density matrix from the example in:
  #   quantum_pca.py:
  #
  print('  Single qubit.')
  purify(ops.Operator([(0.22704306, 0.34178495),
                       (0.34178495, 0.77295694)]), 1)

  print('  Bell states.')
  purify(bell.bell_state(0, 0).density(), 2)
  purify(bell.bell_state(0, 1).density(), 2)
  purify(bell.bell_state(1, 0).density(), 2)
  purify(bell.bell_state(1, 1).density(), 2)

  print('  GHZ state.')
  purify(bell.ghz_state(4).density(), 4)

  # A handful of random states.
  print('  Random 2 qubit states.')
  for _ in range(1000):
    psi = state.State(np.random.rand(4)).normalize()
    purify(psi.density(), 2)

  print('  Random 4 qubit states.')
  for _ in range(100):
    psi = state.State(np.random.rand(16)).normalize()
    purify(psi.density(), 4)

  print('Done.')


if __name__ == '__main__':
  app.run(main)
