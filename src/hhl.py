# python3
"""HHL algorithm."""

# This is WIP: Work in Progress

# To verify the implementation this code closely mirrors the
# numerical example in:
#   'Step-by-Step HHL Algorithm Walkthrough to Enhance the
#    Understanding of Critical Quantum Computing Concepts.'
# by Morrell, Zaman, and Wong.
#
# To make the implementation somewhat more general, the
# specific comparisons to the paper are guarded by a
# boolean 'verify'.


from absl import app
import numpy as np

from src.lib import circuit
from src.lib import ops
from src.lib import state


# In the derivation of the HHL algorithm, the inverse of an eigenvalue
# must be computed. This is being achieved with a Controlled-Y rotation.
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


def check_rotate_ry(lam: float):
  """Evaluate two ways to achieve a rotation about the y-axis."""

  if lam < 1.0:
    raise AssertionError('Incorrect input, lam must be >= 1.')

  # Compute the two factors as shown above.
  #   We set C=1. In general, C should be a normalization
  #   factor, with C < lam to avoid a negative sqrt.
  #
  #   Here, C=1 and lam must be >= 1.0 to avoid a negative sqrt.
  factor_0 = np.sqrt(1.0 - 1.0 / (lam * lam))
  factor_1 = 1.0 / lam

  # Compute a y-rotation by theta:
  theta = 2.0 * np.arcsin(1.0 / lam)
  psi = state.zeros(1)
  psi = ops.RotationY(theta)(psi)

  # Check and compare the results.
  if not np.isclose(factor_0, psi[0], atol=0.001):
    raise AssertionError('Invalid computation.)')

  if not np.isclose(factor_1, psi[1], atol=0.001):
    raise AssertionError('Invalid computation.)')


def check_classic_solution(a, b):
  """Check classic solution, based on paper values."""

  x = np.array([3/8, 9/8])
  if not np.allclose(np.dot(a, x), b, atol=1e-5):
    raise AssertionError('Incorrect classical solution')

  # The ratio of |x0|^2 and |x1|^2 is:
  ratio_x = (x[1] * x[1].conj()) / (x[0] * x[0].conj())
  print(f'Solution^2 ratio: {ratio_x:.1f}')
  return ratio_x


def compute_sorted_eigenvalues(a, verify: bool = True):
  """Compute and verify the sorted eigenvalues/vectors."""

  # Eigenvalue/vector computation.
  w, v = np.linalg.eig(a)

  # We sort the eigenvalues and eigenvectors (to match the paper).
  idx = w.argsort()
  w = w[idx]
  v = v[:, idx]

  # From the experiments in 'spectral_decomp.py', we know that for
  # a Hermitian A:
  #   Eigenvalues are real (that's why a Hamiltonian must be Hermitian)
  w = np.real(w)

  #   Eigenvectors are orthogonal and orthonormal
  #   We can construct the matrix via the spectral theorem as:
  #      A = sum_i {lambda_i * |u_i><u_i|}
  if verify:
    dim = a.shape[0]
    x = np.matrix(np.zeros((dim, dim)))
    for i in range(dim):
      x = x + w[i] * np.outer(v[:, i], v[:, i].adjoint())
    if not np.allclose(a, x, atol=1e-5):
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
      raise AssertionError('w[0] must be larger than w[1].')
    ratio = w[1] / w[0]
    if ratio - np.round(ratio) > 0.01:
      raise AssertionError('We assume integer ratio between w[1] and w[0]')

  return w, v


def compute_u_matrices(a, w, v, t, verify):
  """Compute the various U matrices and exponentiations."""


  # We can compute A diagonalized from v as:
  a_diag = v.transpose().conj() @ a @ v
  if verify:
    if not np.allclose(a_diag, np.array([[2/3, 0],
                                         [0, 4/3]]), atol=1e-5):
      raise AssertionError('Incorrect computation of Adiag')

  # Compute the matrices U an U^2 from A via:
  #   U = exp(i * A * t) (^2)
  #
  # Since U is diagonal:
  u = ops.Operator(np.array([[np.exp(1j * w[0] * t), 0],
                             [0, np.exp(1j * w[1] * t)]]))
  if verify:
    if not np.allclose(u, np.array([[1j, 0],
                                    [0, -1]]), atol=1e-5):
      raise AssertionError('Incorrect computation of U')

  u2 = u @ u
  if verify:
    if not np.allclose(u2, np.array([[-1, 0],
                                     [0, 1]]), atol=1e-5):
      raise AssertionError('Incorrect computation of U^2')
    if not u.is_unitary() or not u2.is_unitary():
      raise AssertionError('U\'s are not unitary')

  # Both U and U^2 are in the eigenvector basis of A. To convert these
  # operators to the computational basis we apply the similarity
  # transformations:
  u = v @ u @ v.transpose().conj()
  u2 = v @ u2 @ v.transpose().conj()
  if verify:
    if not np.allclose(u, 0.5 * np.array([[-1+1j, 1+1j],
                                          [1+1j, -1+1j]]), atol=1e-5):
      raise AssertionError('Incorrect conversion of U to comp. basis.')
    if not np.allclose(u2, np.array([[0, -1],
                                     [-1, 0]]), atol=1e-5):
      raise AssertionError('Incorrect conversion of U^2 to comp. basis.')

  # Compute the inverses U^-1 and U^-2:
  um1 = np.linalg.inv(u)
  um2 = um1 @ um1
  if verify:
    if not np.allclose(um1, 0.5 * np.array([[-1-1j, 1-1j],
                                            [1-1j, -1-1j]]), atol=1e-5):
      raise AssertionError('Something is wrong with U^-1.')
    if not np.allclose(u2, um2, atol=1e-5):
      raise AssertionError('Something is wrong with U^-2.')

  return u, u2, um1, um2


def construct_circuit(a, b, w, v, u, u2, um1, um2, C):
  """Construct a circuit for the given paramters."""

  qc = circuit.qc('hhl', eager=True)

  b = qc.reg(1, 0)
  clock = qc.reg(2, 0)
  anc = qc.reg(1, 0)

  qc.x(b)
  qc.psi.dump('psi 1')

  qc.h(clock)
  qc.psi.dump('psi 2')

  # State Preparation.
  qc.ctl_2x2(clock[0], b, u)
  qc.ctl_2x2(clock[1], b, u2)
  qc.psi.dump('psi 3')

  # Inverse QFT.
  qc.h(clock[1])
  qc.cu1(clock[1], clock[0], -np.pi/2)
  qc.h(clock[0])
  qc.swap(clock[0], clock[1])
  qc.psi.dump('psi 4')

  # From above we know that:
  #   theta = 2 arcsin(1 / 1am_j)
  #
  # We need a function that performs the rotation for
  # all lam's that are non-zero. In the verify example the
  # lam's are |1> and |2>:
  #
  #   theta(c) = theta(c_1 c_0) = 2 arcsin(1 / c)
  #
  # where c is the value of the clock qubits, c_1 c_0 are c
  # in binary.
  #
  # In the example, we must ensure that this function is correct
  # for the states |01> and |10>, corresponding to the lam's:
  #
  #   theta(1) = theta(01) = 2 arcsin(1 / 1) = pi
  #   theta(2) = theta(10) = 2 arcsin(1 / 2) = pi/3
  #
  # In general, this theta function must be computed (which is
  # trivial when lam's binary representations don't have matching 1's).
  # For the example, the solution is simple as no bits overlap:
  #
  #   theta(c) = theta(c_1 c_0) = pi/3 c_1 + pi c_0
  #
  # So we have to rotate the ancilla via qubit c_1 by pi/3
  # and via qubit c_0 by pi.
  qc.cry(clock[0], anc, np.pi)
  qc.cry(clock[1], anc, np.pi/3)
  qc.psi.dump('psi 5')

  p, psi = qc.measure_bit(anc[0], 1, collapse=True)
  qc.psi.dump('psi 6')

  # QFT
  qc.swap(clock[0], clock[1])
  qc.h(clock[0])
  qc.cu1(clock[1], clock[0], np.pi/2)
  qc.h(clock[1])
  qc.psi.dump('psi 5')

  # Uncompute state initialization.
  qc.ctl_2x2(clock[1], b, um2)
  qc.ctl_2x2(clock[0], b, um1)

  qc.h(clock[1])
  qc.h(clock[0])
  qc.psi.dump('psi 9')


  # p, psi = qc.measure_bit(clock[0], 1, collapse=True)
  # p, psi = qc.measure_bit(clock[1], 1, collapse=True)
  psi.dump('after anc measurement.')
  return qc


def run_experiment(a, b, verify: bool = False):
  """Run a single instance of HHL for Ax = b."""

  if not a.is_hermitian():
    raise AssertionError('Input A must be Hermitian.')

  # To check, we can solve Ax = b classically:
  if verify:
    ratio_x = check_classic_solution(a, b)

  # For quantum, initial parameters.
  dim = a.shape[0]

  # pylint: disable=invalid-name
  N = dim ** 2

  # Compute (and verify) eigenvalue/vectors.
  w, v = compute_sorted_eigenvalues(a, verify)

  # Compute and print the ratio. We will compare the results
  # against this value below.
  ratio = w[1] / w[0]
  print(f'Ratio between Eigenvalues: {ratio:.1f}')

  # We also know that:
  #   lam_i = (N * w[j] * t) / (2 * np.pi)
  #
  # We want lam_i to be integers, so we compute 't' as:
  #   t = lam[0] / N / w[0] * 2 * np.pi
  #
  # Which, for the example, comes to 3/4 pi:
  t = ratio / N / w[1] * 2 * np.pi
  if verify:
    if t - 3.0 * np.pi / 4.0 > 1e-5:
      raise AssertionError('Incorrect calculation of t')

  # With 't' we can now compute the integer eigenvalues:
  lam = [(N * np.real(w[i]) * t / (2 * np.pi)) for i in range(2)]
  print(f'Scaled Lambda\'s are: {lam[0]:.1f}, {lam[1]:.1f}')

  # Compute the U matrices.
  u, u2, um1, um2 = compute_u_matrices(a, w, v, t, verify)

  # On to computing the rotations.
  #
  # The factors to |0> and 1> of the ancilla will be:
  #   \sqrt{1 - C^2 / lam_j^2} and C / lam_j
  #
  # C must be smaller than the minimal lam. We set it to the minimum:
  C = np.min(lam)
  print(f'Set C to minimal Eigenvalue: {C:.1f}')

  # Now we have all the values and matrices. Let's construct a circuit.
  qc = construct_circuit(a, b, w, v, u, u2, um1, um2, C)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('HHL Algorithm, based on "Step-by-Step HHL Algorithm Walkthrough..."')
  print('*** This is WIP ***')

  # Check the rotation mechanism.
  check_rotate_ry(1.2)

  # The numerical example from:
  #   "Step-by-Step HHL Algorithm Walkthrough..." by
  #    Morrell, Zaman, Wong
  # (which is a 2x2 Hermitian matrix)
  a = ops.Operator(np.array([[1.0, -1/3], [-1/3, 1]]))
  b = ops.Operator(np.array([0, 1]))
  run_experiment(a, b, True)


if __name__ == '__main__':
  np.set_printoptions(precision=3)
  app.run(main)
