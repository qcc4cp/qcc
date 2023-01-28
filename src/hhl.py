# python3
"""HHL algorithm."""

# HLL is an ADVANCED ALGORIHM. For study it is recommended
# to first become proficient with these key concepts:
#   basis changes
#   phase estimation
#   quantum Fourier transformation
#   Hamiltonian simulation
#
# This version (compared to hhl_2x2.py is more general and
# will be extended to support 4x4 matrices as well. The
# numerical comparisons to a reference numerical example
# have been removed.

from absl import app
import numpy as np

from src.lib import circuit
from src.lib import ops
from src.lib import state


def check_classic_solution(a, b):
  """Check classic solution."""

  x = np.linalg.solve(a, b)
  for i in range(1, 2 ** b.nbits):
    ratio_x = np.real((x[i] * x[i].conj()) / (x[0] * x[0].conj()))
    print(f'Classic ratio: {ratio_x:.3f}')
  return ratio_x


def check_results(qc, a, b):
  """Check the results by inspecting the final state."""

  ratio_classical = check_classic_solution(a, b)
  res = (qc.psi > 0.001).nonzero()[0]
  for j in range(1, b.size):
    ratio_quantum = np.real(qc.psi[res[j]]**2 / qc.psi[res[0]]**2)
    print(f'Quantum ratio: {ratio_quantum:.3f}\n')
    if not np.allclose(ratio_classical, ratio_quantum, atol=1e-4):
      raise AssertionError('Incorrect result.')


def compute_sorted_eigenvalues(a):
  """Compute the sorted eigenvalues/vectors."""

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
  return w, v


def compute_u_matrix(a, w, v, t):
  """Compute the various U matrices and exponentiations."""

  # Compute the matrices U an U^2 from A via:
  #   U = exp(i * A * t) (^2)
  #
  # Since U is diagonal:
  u = ops.Operator(np.array([[np.exp(1j * w[0] * t), 0],
                             [0, np.exp(1j * w[1] * t)]]))

  # Both U and U^2 are in the eigenvector basis of A. To convert these
  # operators to the computational basis we apply the similarity
  # transformations:
  u = v @ u @ v.transpose().conj()
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
    qc.ctl_2x2(clock[idx], breg, u_phase)
    u_phase_gates.append(u_phase)
    u_phase = u_phase @ u_phase

  # Inverse QFT. After this, the eigenvalues will be
  # in the clock register.
  qc.inverse_qft(clock, True)

  # From above we know that:
  #   theta = 2 arcsin(1 / lam_j)
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
    qc.ctl_2x2(clock[idx], breg, np.linalg.inv(u_phase_gates[idx]))

  # Move clock bits out of Hadamard basis.
  qc.h(clock)
  qc.psi.dump('Final state')
  return qc


def run_experiment(a, b, clock_bits):
  """Run a single instance of HHL for Ax = b."""

  if not a.is_hermitian():
    raise AssertionError('Input A must be Hermitian.')
  print(f'Clock bits   : {clock_bits}')
  print(f'Dimensions A : {a.shape[0]}x{a.shape[1]}')

  # For quantum, initial parameters.
  dim = a.shape[0]

  # Compute eigenvalue/vectors.
  w, v = compute_sorted_eigenvalues(a)

  # Compute and print the ratio. We will compare the results
  # against this value below.
  ratio = w[1] / w[0]

  # We also know that:
  #   lam_i = (N * w[j] * t) / (2 * np.pi)
  # We want lam_i to be integers, so we compute 't' as:
  #   t = lam[0] / N / w[0] * 2 * np.pi
  n = 2 ** clock_bits
  t = ratio / n / w[1] * 2 * np.pi

  # With 't' we can now compute the integer eigenvalues:
  lam = [(n * np.real(w[i]) * t / (2 * np.pi)) for i in range(2)]
  print(f'int lambda\'s : {lam[0]:.1f}, {lam[1]:.1f}')
  # TODO: Print _all_ ratios here.

  # Compute the U matrices.
  u = compute_u_matrix(a, w, v, t)

  # On to computing the rotations.
  #
  # The factors to |0> and 1> of the ancilla will be:
  #   \sqrt{1 - C^2 / lam_j^2} and C / lam_j
  #
  # C must be smaller than the minimal lam. We set it to the minimum:
  C = np.min(lam)
  print(f'Set C to min : {C:.1f}')

  # Now we have all the values and matrices. Let's construct a circuit.
  qc = construct_circuit(b, lam, u, C, clock_bits)
  check_results(qc, a, b)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('General HHL Algorithm...')
  print('*** This is WIP, extending to larger A matrices. ***')

  
  a = ops.Operator(np.array([[3/5, -1/5], [-1/5, 3/5]]))
  b = ops.Operator(np.array([1, 0]))
  run_experiment(a, b,clock_bits = 4)


if __name__ == '__main__':
  np.set_printoptions(precision=4)
  app.run(main)
