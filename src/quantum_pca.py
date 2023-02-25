# python3
"""Example: Quantum Principal Component Analysis (PCA)."""


# This PCA data set and implementation is from this paper:
#   Quantum Algorithm Implementation for Beginners
#   https://arxiv.org/pdf/1804.03719.pdf
#
# A sample implementation can be found here:
#   https://github.com/Haokai-Zhang/ExampleQPCA
#
# The implementation here closely follows the sample implementation
# in order to avoid confusion. Thanks to Haokai-Zhang for his
# contributions (couldn't have done it without it!).

import random
from absl import app
import numpy as np

from src.lib import circuit
from src.lib import state


def pca(x):
  """A single quantum principal component analysis."""

  # We center and normalize (by dividing by 1000) the data.
  #
  # Note: The factor 1000 helps with the numerics. Other factors
  #       work, but the atol would have to be adjusted when
  #       comparing the results below.
  #
  x[0] = x[0] - np.average(x[0])
  x[1] = (x[1] - np.average(x[1])) / 1000

  # Compute the unbiased covariance matrix
  #   It is unbiased, hence the 15, else we could use 15-1
  #   (which doesn't make a difference).
  #
  m = np.array([[np.dot(x[0], x[0]), np.dot(x[0], x[1])],
                [np.dot(x[1], x[0]), np.dot(x[1], x[1])]]) / 15

  # We scale down M to make it a density matrix (where the trace
  # has to be 1). Later we must not forget to scale up the
  # results again.
  #
  rho = m / np.trace(m)

  # Construct purified state \psi. This is a bit like cheating
  # since in order to construct this state, we have to compute
  # the eigenvalues - the computation of which is the whole point of
  # this algorithm. On a real quantum machine, this has to be
  # done via state preparation.
  #
  rho_eig_val, rho_eig_vec = np.linalg.eig(rho)
  p_vec = np.concatenate((np.sqrt(rho_eig_val), np.sqrt(rho_eig_val)))
  u_vec = rho_eig_vec.reshape((4))
  psi = state.State(p_vec * u_vec)

  # Construct swap test. The expectation value of the swap gate under
  # the purified state allows us to re-construct the eigenvalues.
  #
  # Here we just initialize qubits [1,2] and [3,4] with the state
  # as we want it to be (from above calculations). On a real quantum
  # computer we would have to add circuitry to actually generate
  # this state.
  #
  qc = circuit.qc('pca')
  qc.reg(1, 0)
  qc.state(psi)  # qubits 1, 2
  qc.state(psi)  # qubits 3, 4
  qc.h(0)
  qc.cswap(0, 1, 3)
  qc.h(0)

  # We know that for the diagonal of rho:
  #   p0^2 + p1^2 = purity
  #   p0 + p1 = 1
  #
  # Squaring:
  #     (p0 + p1)^2 = p0^2 + p1^2 + 2p0p1
  # ->  p0p1 = (1-P)/2
  # ->  p0 = 1 - p1
  #     p0 - p0^2 = (1-P)/2
  # ->  p0^2 - p0 + (1-P)/2 = 0
  #
  # Quadratic formula:
  #     p0,1 = 1 -+ sqrt(2 * P - 1) / 2
  #
  # We have to scale the results up to the original covariance
  # matrix by multiplying with np.trace(M).
  #
  purity = qc.pauli_expectation(idx=0)
  m_0 = (1 - np.sqrt(2 * purity - 1)) / 2 * np.trace(m)
  m_1 = (1 + np.sqrt(2 * purity - 1)) / 2 * np.trace(m)
  print(f'Eigenvalues Quantum PCA: {m_0:.6f}, {m_1:.6f}')

  # Compare to classically derived values, which must match.
  m, _ = np.linalg.eig(m)
  if (not np.isclose(m_0, m[0], atol=1e-5) or
      not np.isclose(m_1, m[1], atol=1e-5)):
    raise AssertionError('Incorrect Computation.')
  print(f'Eigenvalues Classically: {m[0]:.6f}, {m[1]:.6f}. Correct')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Quantum Principal Component Analysis (PCA).')

  # Data set from the paper is the correlation of
  #   - number of bedrooms
  #   - square footage
  #
  x = [[4, 3, 4, 4, 3, 3, 3, 3, 4, 4, 4, 5, 4, 3, 4],
       [3028, 1365, 2726, 2538, 1318, 1693, 1412, 1632, 2875,
        3564, 4412, 4444, 4278, 3064, 3857]]
  pca(x)

  for _ in range(10):
    for idx in range(len(x[0])):
      x[1][idx] = random.random() * 10000
    pca(x)


if __name__ == '__main__':
  app.run(main)
