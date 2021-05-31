# python3
"""Example: Phase Estimation, following Nielsen/Chuang Section 5.2."""

from absl import app

import numpy as np
import scipy.stats

from src.lib import ops
from src.lib import state


def expo_u(psi: state.State, u: ops.Operator, t: int) -> state.State:
  """Exponentiate U."""

  # Unpack the binary fractions of the phase into the first t qubits.
  #
  # t qubits
  # |0> - H -------------------- ... ----o --
  # |0> - H -----------------o-- ... --- | --
  # |0> - H ---------o-------|-- ... --- | --
  # |0> - H -o-------|-------|-- ... --- | --
  #          |       |       |           |
  # nbits qubits     |       |           |
  # |u> --- U^1 --- U^2 --- U^4 ... --- U^s^(t-1)
  #
  psi = ops.Hadamard(t)(psi)
  for idx, inv in enumerate(range(t-1, -1, -1)):
    u2 = u
    for _ in range(idx):
      u2 = u2(u2)
    psi = ops.ControlledU(inv, t, u2)(psi, inv)
  return psi


def run_experiment(nbits: int, t: int = 8):
  """Run single phase estimation experiment."""

  # Make a unitary and find eigenvalue/vector to estimate.
  #
  umat = scipy.stats.unitary_group.rvs(2**nbits)
  eigvals, eigvecs = np.linalg.eig(umat)
  u = ops.Operator(umat)

  # Pick eigenvalue at 'eigen_index'
  # (any eigenvalue / eigenvector pair will work).
  eigen_index = 1
  phi = np.real(np.log(eigvals[eigen_index]) / (2j*np.pi))
  if phi < 0:
    phi += 1

  # Make state + circuit to estimate phi.
  # Pick eigenvector 'eigen_index' to math the eigenvalue.
  psi = state.zeros(t) * state.State(eigvecs[:, eigen_index])
  psi = expo_u(psi, u, t)
  psi = ops.Qft(t).adjoint()(psi)

  # Find state with highest measurement probability and show results.
  #
  maxbits, maxprob = psi.maxprob()
  phi_estimate = sum(maxbits[i] * 2**(-i-1) for i in range(t))

  delta = abs(phi - phi_estimate)
  print('Phase   : {:.4f}'.format(phi))
  print('Estimate: {:.4f} delta: {:.4f} probability: {:5.2f}%'
        .format(phi_estimate, delta, maxprob * 100.0))
  if delta > 0.02 and phi_estimate < 0.98:
    print('*** Warning: Delta is large')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  nbits = 3
  t = 6
  print('Estimating {} qubits random unitary eigenvalue '
        .format(nbits) + 'with {} bits of accuracy.'.format(t))
  for i in range(10):
    run_experiment(nbits, t)


if __name__ == '__main__':
  app.run(main)
