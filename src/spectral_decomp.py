# python3
"""Example: Spectral Decomposition."""

from absl import app
import numpy as np
import scipy.stats

from src.lib import ops


def spectral_decomp(ndim: int):
  """Implement and verify spectral decomposition theorem."""

  # The spectral theorem says that a Hermitian matrix can be written as
  # the sum over eigenvalues lambda_i and eigenvectors u_i, as:
  #
  #   A = sum_i {lambda_i * |u_i><u_i|}
  #
  # Let's check by example...

  # Create random unitary. It will be unitary, but not Hermitian.
  #
  u = scipy.stats.unitary_group.rvs(ndim)
  umat = ops.Operator(u)

  # Make it Hermitian by computing:
  #    A = 1/2 * (A + A^*)
  #
  # This makes A Hermitian (but no longer unitary).
  #
  hmat = 0.5 * (umat + umat.adjoint())
  if not np.allclose(hmat, hmat.adjoint()):
    raise AssertionError('Something is wrong, created non-Hermitian.')

  # Compute eigenvalues and vectors.
  #
  # Note the Python way to extract column c from
  # a matrix v as v[:, c]:
  #
  #     w       : array of eigenvalues
  #     v[:, i] : eigenvector corresponding to w[i]
  #
  w, v = np.linalg.eig(hmat)

  # Check that the eigenvalues are real.
  #
  for i in range(ndim):
    if not np.allclose(w[i].imag, 0.0):
      raise AssertionError('Found non-real eigenvalue.')

  # Check that the eigenvectors are orthogonal.
  #
  for i in range(ndim):
    for j in range(i + 1, ndim):
      dot = np.dot(v[:, i], v[:, j].adjoint())
      if not np.allclose(dot, 0.0, atol=1e-5):
        raise AssertionError('Invalid, non-orthogonal basis found')

  # Check that eigenvectors are orthonormal.
  for i in range(ndim):
    dot = np.dot(v[:, i], v[:, i].adjoint())
    if not np.allclose(dot, 1.0, atol=1e-5):
      raise AssertionError('Found non-orthonormal basis vectors')

  # Construct a matrix following the spectral theorem and
  # check equivalance.
  #
  #  A = sum_i {lambda_i * |u_i><u_i|}
  #
  x = np.matrix(np.zeros((ndim, ndim)))
  for i in range(ndim):
    x = x + w[i] * np.outer(v[:, i], v[:, i].adjoint())
  if not np.allclose(hmat, x, atol=1e-5):
    raise AssertionError("Spectral decomp doesn't seem to work.")

  # Can we use this elegant spectral decomposition to compute the
  # the inverse of a matrix?
  #
  # Let's compute the inverse by using only the inverse of the eigenvalues,
  # as (1 / eigenvalue), but keep the same eigenvectors otherwise:
  #
  x = np.matrix(np.zeros((ndim, ndim)))
  for i in range(ndim):
    x = x + 1 / w[i] * np.outer(v[:, i], v[:, i].adjoint())
  if not np.allclose(np.linalg.inv(hmat), x, atol=1e-5):
    raise AssertionError("Inverse computation doesn't seem to work.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  iterations = 100
  print(f'{iterations} Spectral decompositiona')
  for _ in range(iterations):
    spectral_decomp(32)
  print('Success')


if __name__ == '__main__':
  app.run(main)
