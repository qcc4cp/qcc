# python3
"""HHL algorithm."""

# This is WIP: Work in Progress

from absl import app
import numpy as np

from src.lib import ops
from src.lib import state


# In the derivation of the HHL algorithm, the inverse of an eigenvalue
# must be computed. This is being achieved with a controlled-Y rotation.
#
# In the derivation, this term appears (in pseudo-Latex):
#
#   \sqrt{1 - C^2/lam^2}        C / lam
#   -------------------- |0> +  -------- |1>
#       factor_0                factor_1
#
# This is similar to the rotation in Grover.
#
# It now can be shown that this expression corresponds to a y-rotation
# by angle theta being:
#
#   theta = 2 arcsin(1/lam).
#
# The code below is trying to verify the rotation itself.


def rotate_ry(lam: float):
  """Evaluate two ways to achieve a rotation about the y-axis."""

  if lam < 1.0:
    raise AssertionError('Incorrect input, lam must be >= 1.')

  # Compute the two factors as shown above.
  #   We set C=1. In general, C should be a normalization
  #   factor, with C < lam to avoid a negative sqrt.
  #
  #   Here, C=1 and lam must be >= 1.0 to avoid a negative sqrt.
  #
  factor_0 = np.sqrt(1.0 - 1.0 / (lam * lam))
  factor_1 = 1.0 / lam

  # Compute a y-rotation by theta:
  #
  theta = 2.0 * np.arcsin(1.0 / lam)
  psi = state.zeros(1)
  psi = ops.RotationY(theta)(psi)

  # Check and compare the results.
  #
  # print(f'controlled ry({lam:.2f}): theta: {theta*180.0:.4f}', end='')
  # print(f'-> {factor_0:.2f}|0> + {factor_1:.2f}|1>')

  if not np.isclose(factor_0, psi[0], atol=0.001):
    raise AssertionError('Invalid computation.)')

  if not np.isclose(factor_1, psi[1], atol=0.001):
    raise AssertionError('Invalid computation.)')


def run_experiment(A, b, verify: bool = False):
  """Run a single instance of HHL for Ax = b."""

  if not A.is_hermitian():
    raise AssertionError('Input A must be Hermitian.')

  # To check, we can solve Ax = b classically:
  x = np.array([3/8, 9/8])
  if not np.allclose(np.dot(A, x), b, atol=1e-5):
    raise AssertionError('Incorrect classical solution')

  # The ratio of |x0|^2 and |x1|^2 is:
  ratio_x = (x[1] * x[1].conj()) / (x[0] * x[0].conj())
  print(f'Solution^2 ratio: {ratio_x:.1f}')

  # For quantum, initial parameters.
  dim = A.shape[0]
  N = dim ** 2

  # Eigenvalue/vector computation.
  w, v = np.linalg.eig(A)

  # We sort the eigenvalues and eigenvectors to match the paper.
  idx = w.argsort()
  w = w[idx]
  v = v[:,idx]

  # From the experiments in 'spectral_decomp.py', we know that for
  # a Hermitian A:
  #    - Eigenvalues are real (that's why a Hamiltonian must be Hermitian)
  w = np.real(w)

  #    - Eigenvectors are orthogonal and orthonormal
  #    - We can construct the matrix via the spectral theorem as:
  #         A = sum_i {lambda_i * |u_i><u_i|}
  if verify:
    x = np.matrix(np.zeros((dim, dim)))
    for i in range(dim):
      x = x + w[i] * np.outer(v[:, i], v[:, i].adjoint())
    if not np.allclose(A, x, atol=1e-5):
      raise AssertionError('Spectral decomp doesn\'t seem to work.')

  # We want to use basis encoding to encode the eigenvalues. For this,
  # we have to map the float eigenvalues in w to integer values.
  # We do this by computing the ratio between w[0] and w[1] and then
  # mapping this to states |1> and |scaled up>.
  #
  # In this simple example, we make sure that the ratio is an integer
  # multiple, that makes it a little easier as the |1> state doesn't
  # need to be scaled up.
  #
  if verify:
    if w[1] < w[0]:
      raise AssertionError('w[0] must be larger than w[1], consider sorting.')

  ratio = w[1] / w[0]
  if ratio - np.round(ratio) > 0.01:
    raise AssertionError('We assume integer ratio between w[1] and w[0]')
  print(f'Ratio between Eigenvalues: {ratio:.1f}')

  # We also know that:
  #   lam_i = (N * w[j] * t) / (2 * np.pi)
  #
  # We want lam_i to be integers, so we compute 't' as:
  #   t = lam[0] / N / w[0] * 2 * np.pi
  #
  # Which, for the example, comes to 3/4 pi:
  t =  ratio / N / w[1] * 2 * np.pi
  if verify:
    if t - 3.0 * np.pi / 4.0 > 1e-5:
      raise AssertionError('Incorrect calculation of t')

  # With 't' we can now compute the integer eigenvalues:
  lam = [(N * np.real(w[i]) * t / (2 * np.pi)) for i in range(2)]
  print(f'Scaled Lambda\'s are: {lam[0]:.1f}, {lam[1]:.1f}')

  # We can compute A diagonalized from v as:
  a_diag = v.transpose().conj() @ A @ v
  if verify:
    if not np.allclose(a_diag, np.array([[2/3, 0],
                                         [0, 4/3]]), atol=1e-5):
      raise AssertionError('Incorrect computation of Adiag')


  # Compute the matrix U from A via:
  #   U = exp(i * A * t)
  #
  # Since U is diagonal:
  #
  u = ops.Operator(np.array([[np.exp(1j * w[0] * t), 0],
                             [0, np.exp(1j * w[1] * t)]]))
  if verify:
    if not np.allclose(u, np.array([[1j, 0],
                                    [0, -1]]), atol=1e-5):
      raise AssertionError('Incorrect computation of U')

  # Compute U^2
  #
  u2 = u@u
  if verify:
    if not np.allclose(u2, np.array([[-1, 0],
                                     [0, 1]]), atol=1e-5):
      raise AssertionError('Incorrect computation of U^2')
    if not u.is_unitary() or not u2.is_unitary():
      raise AssertionError('U\'s are not unitary')

  # Both U and U^2 are in the eigenvector basis of A. To convert these
  # operators to the computational basis we apply the similarity
  # transformations:
  #
  u = v @ u @ v.transpose().conj()
  u2 = v @ u2 @ v.transpose().conj()
  if verify:
    if not np.allclose(u, 0.5 * np.array([[-1+1j, 1+1j],
                                          [1+1j, -1+1j]]), atol=1e-5):
      raise AssertionError('Incorrect cobversion of U to comp. basis')                                          
    if not np.allclose(u2, np.array([[0, -1],
                                     [-1, 0]]), atol=1e-5):
      raise AssertionError('Incorrect conversion of U^2 to comp. basis')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('HHL Algorithm, based on "Step-by-Step HHL Algorithm Walkthrough..."')
  print('*** This is WIP ***')
  return

  # Check the rotation mechanism.
  rotate_ry(1.2)

  # The numerical example from:
  #   "Step-by-Step HHL Algorithm Walkthrough..." by
  #    Morrell, Zaman, Wong
  # (which is a 2x2 Hermitian matrix)
  #
  A = ops.Operator(np.array([[1.0, -1/3], [-1/3, 1]]))
  b = ops.Operator(np.array([0, 1]))
  run_experiment(A, b, True)


if __name__ == '__main__':
  np.set_printoptions(precision=3)
  app.run(main)
