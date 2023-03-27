# python3
"""Example: HHL algorithm on 2x2 matrices."""

# HLL is an ADVANCED ALGORIHM. For study it is recommended
# to first become proficient with these key concepts:
#   basis changes
#   phase estimation
#   quantum Fourier transformation
#   Hamiltonian simulation
#
# To experiment with HHL the code here closely mirrors (and allows to
# verify) the numerical example in:
#   'Step-by-Step HHL Algorithm Walkthrough to Enhance the
#    Understanding of Critical Quantum Computing Concepts.'
# by Morrell, Zaman, and Wong.
#
# The specific comparisons to the paper are guarded by a
# boolean 'verify'. This code only works for 2x2 matrices with
# Eigenvalues that map onto 2 bits. as it only supports
# rotations with 2 clock bits.


from absl import app
import numpy as np

from src.lib import circuit
from src.lib import ops
from src.lib import state


def check_rotate_ry(lam: float):
  """Evaluate two ways to achieve a rotation about the y-axis."""

  if lam < 1.0:
    raise AssertionError('Incorrect input, lam must be >= 1.')

  # In the derivation of the HHL algorithm, the inverse of an eigenvalue
  # must be computed. This is being achieved with a Controlled-Y rotation.
  #
  # In the derivation, this term appears (in pseudo-Latex):
  #
  #   \sqrt{1 - C^2/lam^2}        C / lam
  #   -------------------- |0> +  -------- |1>
  #       factor_0                factor_1
  #
  # It can be shown that this expression corresponds to a y-rotation
  # by angle theta being:
  #
  #   theta = 2 arcsin(1/lam).
  #
  # As a preliminaty, the code below verifies this rotation itself.
  #
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

  # Check the results.
  if not np.isclose(factor_0, psi[0], atol=0.001):
    raise AssertionError('Invalid computation.)')
  if not np.isclose(factor_1, psi[1], atol=0.001):
    raise AssertionError('Invalid computation.)')


def check_classic_solution(a, b, verify):
  """Check classic solution, verify against paper values."""

  x = np.linalg.solve(a, b)
  if verify:
    y = np.array([3 / 8, 9 / 8])
    if not np.allclose(np.dot(a, y), b, atol=1e-5):
      raise AssertionError('Incorrect classical solution')
    if not np.allclose(x, y, atol=1e-5):
      raise AssertionError('Incorrect classical solution')

  # The ratio of |x0|^2 and |xi|^2 is:
  for i in range(1, 2**b.nbits):
    ratio_x = np.real((x[i] * x[i].conj()) / (x[0] * x[0].conj()))
    print(f'Classic solution^2 ratio: {ratio_x:.3f}')

  # For now, just return the last raio.
  return ratio_x


def check_results(qc, a, b, verify):
  """Check the results by inspecting the final state."""

  ratio_classical = check_classic_solution(a, b, verify)

  res = (qc.psi > 0.001).nonzero()[0]
  for j in range(1, b.size):
    ratio_quantum = np.real(qc.psi[res[j]] ** 2 / qc.psi[res[0]] ** 2)
    print(f'Quantum solution^2 ratio: {ratio_quantum:.3f}\n')
    if not np.allclose(ratio_classical, ratio_quantum, atol=1e-4):
      raise AssertionError('Incorrect result.')


def compute_sorted_eigenvalues(a, verify: bool = True):
  """Compute and verify the sorted eigenvalues/vectors."""

  # Eigenvalue/vector computation.
  w, v = np.linalg.eig(a)

  # We sort the eigenvalues and eigenvectors (also to match the paper).
  idx = w.argsort()
  w = w[idx]
  v = v[:, idx]

  # From the experiments in 'spectral_decomp.py', we know that for
  # a Hermitian A:
  #   Eigenvalues are real (that's why a Hamiltonian must be Hermitian)
  w = np.real(w)

  #   Eigenvectors are orthogonal and orthonormal.
  #   We can construct the matrix A via the spectral theorem as:
  #      A = sum_i {lambda_i * |u_i><u_i|}
  if verify:
    dim = a.shape[0]
    x = np.matrix(np.zeros((dim, dim)))
    for i in range(dim):
      x = x + w[i] * np.outer(v[:, i], v[:, i].adjoint())
    if not np.allclose(a, x, atol=1e-5):
      raise AssertionError('Spectral decomp doesn\'t seem to work.')

  # We want to use basis encoding to encode the eigenvalues. For this,
  # we have to map the float eigenvalues of w to integer values.
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


def compute_u_matrix(a, w, v, t, verify):
  """Compute the various U matrices and exponentiations."""

  # Compute the matrices U an U^2 from A via:
  #   U = exp(i * A * t) (^2)
  #
  # Since U is diagonal:
  u = ops.Operator(
      np.array([[np.exp(1j * w[0] * t), 0], [0, np.exp(1j * w[1] * t)]])
  )
  if verify:
    if not np.allclose(u, np.array([[1j, 0], [0, -1]]), atol=1e-5):
      raise AssertionError('Incorrect computation of U')

    u2 = u @ u
    if not np.allclose(u2, np.array([[-1, 0], [0, 1]]), atol=1e-5):
      raise AssertionError('Incorrect computation of U^2')
    if not u.is_unitary() or not u2.is_unitary():
      raise AssertionError('U matrices are not unitary')

  # Both U and U^2 are in the eigenvector basis of A. To convert these
  # operators to the computational basis we apply the similarity
  # transformations:
  u = v @ u @ v.transpose().conj()
  if verify:
    u2 = u @ u
    if not np.allclose(u, 0.5 * np.array([[-1 + 1j, 1 + 1j],
                                          [1 + 1j, -1 + 1j]]), atol=1e-5):
      raise AssertionError('Incorrect conversion of U to comp. basis.')
    if not np.allclose(u2, np.array([[0, -1], [-1, 0]]), atol=1e-5):
      raise AssertionError('Incorrect conversion of U^2 to comp. basis.')

    # Compute the inverses U^-1 and U^-2:
    um1 = np.linalg.inv(u)
    um2 = um1 @ um1
    if not np.allclose(um1, 0.5 * np.array([[-1 - 1j, 1 - 1j],
                                            [1 - 1j, -1 - 1j]]), atol=1e-5):
      raise AssertionError('Something is wrong with U^-1.')
    if not np.allclose(u2, um2, atol=1e-5):
      raise AssertionError('Something is wrong with U^-2.')

   # To verify, we can compute A diagonalized from v as:
    a_diag = v.transpose().conj() @ a @ v
    if not np.allclose(a_diag, np.array([[2 / 3, 0], [0, 4 / 3]]), atol=1e-5):
      raise AssertionError('Incorrect computation of Adiag')

  # Return u in the computational basis.
  return u


def construct_circuit(b, w, u, c, clock_bits=2):
  """Construct a circuit for the given paramters."""

  qc = circuit.qc('hhl', eager=True)
  breg = qc.reg(1, 0)
  clock = qc.reg(clock_bits, 0)
  anc = qc.reg(1, 0)

  # Initialize 'b' to (0, 1), if appropriate.
  if b[1] == 1:
    qc.x(breg)

  # State Preparation, which is basically phase estimation.
  qc.h(clock)
  u_phase = u
  u_phase_gates = []
  for idx in range(clock_bits):
    qc.cu(clock[idx], breg, u_phase)
    u_phase_gates.append(u_phase)
    u_phase = u_phase @ u_phase

  # Inverse QFT. After this, the eigenvalues will be
  # in the clock register.
  qc.inverse_qft(clock, True)

  # From above we know that:
  #   theta = 2 arcsin(1 / lam_j)
  #
  # We need a function that performs the rotation for
  # all lam's that are non-zero. In the verify example the
  # lam's are |1> and |2>:
  #
  #   theta(c) = theta(c_1 c_0) = 2 arcsin(C / c)
  #
  # where c is the value of the clock qubits, c_1 c_0 are c
  # in binary.
  #
  # In the example, we must ensure that this function is correct
  # for the states |01> and |10>, corresponding to the lam's:
  #
  #   theta(1) = theta(01) = 2 arcsin(C=1 / 1) = pi
  #   theta(2) = theta(10) = 2 arcsin(C=1 / 2) = pi/3
  #
  # In general, this theta function must be computed (which is
  # trivial when lam's binary representations don't have matching 1's).
  # For the verified example, the solution is simple as no bits overlap:
  #   theta(c) = theta(c_1 c_0) = pi/3 c_1 + pi c_0
  # So we have to rotate the ancilla via qubit c_1 by pi/3
  # and via qubit c_0 by pi.
  #
  # In general (for 2 lambda's):
  #   if bit 0 is set in the larger lamba, eg., |01> and |11>:
  #
  angle0 = 2 * np.arcsin(c / w[0])
  angle1 = 2 * np.arcsin(c / w[1])
  if int(np.round(w[1])) & 1 == 1:
    angle1 = angle1 - angle0
  qc.cry(clock[0], anc, angle0)
  qc.cry(clock[1], anc, angle1)

  # Measure (and force) ancilla to be |1>.
  _, _ = qc.measure_bit(anc[0], 1, collapse=True)

  # QFT
  qc.qft(clock, True)

  # Uncompute state initialization.
  for idx in range(clock_bits-1, -1, -1):
    qc.cu(clock[idx], breg, np.linalg.inv(u_phase_gates[idx]))

  # Move clock bits out of Hadamard basis.
  qc.h(clock)
  qc.psi.dump('Final state')
  return qc


def run_experiment(a, b, verify: bool = False):
  """Run a single instance of HHL for Ax = b."""

  if not a.is_hermitian():
    raise AssertionError('Input A must be Hermitian.')

  # For quantum, initial parameters.
  dim = a.shape[0]

  # pylint: disable=invalid-name
  N = dim**2

  # Compute (and verify) eigenvalue/vectors.
  w, v = compute_sorted_eigenvalues(a, verify)

  # Compute and print the ratio. We will compare the results
  # against this value below.
  ratio = w[1] / w[0]

  # We also know that:
  #   lam_i = (N * w[j] * t) / (2 * np.pi)
  # We want lam_i to be integers, so we compute 't' as:
  #   t = lam[0] / N / w[0] * 2 * np.pi
  # For the example this becomes 3/4 * pi:
  t = ratio / N / w[1] * 2 * np.pi
  if verify:
    if t - 3.0 * np.pi / 4.0 > 1e-5:
      raise AssertionError('Incorrect calculation of t')

  # With 't' we can now compute the integer eigenvalues:
  lam = [(N * np.real(w[i]) * t / (2 * np.pi)) for i in range(2)]
  print(f'Scaled Lambda\'s are: {lam[0]:.1f}, {lam[1]:.1f}. Ratio: {ratio:.1f}')

  # Compute the U matrices.
  u = compute_u_matrix(a, w, v, t, verify)

  # On to computing the rotations.
  #
  # The factors to |0> and 1> of the ancilla will be:
  #   \sqrt{1 - C^2 / lam_j^2} and C / lam_j
  #
  # C must be smaller than the minimal lam. We set it to the minimum:
  C = np.min(lam)
  print(f'Set C to minimal Eigenvalue: {C:.1f}')

  # Now we have all the values and matrices. Let's construct a circuit.
  qc = construct_circuit(b, lam, u, C, 2)
  check_results(qc, a, b, verify)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('HHL Algorithm...')

  # Preliminary: Check the rotation mechanism.
  check_rotate_ry(1.2)

  # The numerical 2x2 Hermitian example is from:
  #   "Step-by-Step HHL Algorithm Walkthrough..." by
  #    Morrell, Zaman, Wong
  #
  # Maps to Eigenvalues |01> and |10> interpreted as decimal 1 and 2
  a = ops.Operator([[1.0, -1/3], [-1/3, 1]])
  b = ops.Operator([0, 1])
  run_experiment(a, b, True)

  # Maps to Eigenvalues |01> and |11> interpreted as decimal 1 and 3
  a = ops.Operator([[1.0, -1/2], [-1/2, 1]])
  b = ops.Operator([0, 1])
  run_experiment(a, b, False)

  # Maps to Eigenvalues |01> and |10> interpreted as decimal 1/2 and 1/4
  a = ops.Operator([[1.0, -1/3], [-1/3, 1]])
  b = ops.Operator([1, 0])
  run_experiment(a, b, False)

  # Maps to Eigenvalues |01> and |11> interpreted as decimal 1/2 and 1/3
  a = ops.Operator([[1.0, -1/2], [-1/2, 1]])
  b = ops.Operator([1, 0])
  run_experiment(a, b, False)


if __name__ == '__main__':
  np.set_printoptions(precision=3)
  app.run(main)
