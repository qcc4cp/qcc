# python3
"""Example: HHL algorithm."""

# HLL is an ADVANCED ALGORIHM. For study it is recommended
# to first become proficient with these key concepts:
#   Basis changes
#   Phase estimation
#   Quantum Fourier transformation
#   Hamiltonian simulation
#
# This version (compared to hhl_2x2.py) is more general and
# extended to support 4x4 matrices as well. The numerical
# comparisons to the reference numerical example have been removed.

from absl import app
import numpy as np
from src.lib import circuit
from src.lib import ops


def check_classic_solution(a, b):
  """Check classic solution."""

  x = np.linalg.solve(a, b)
  ratio = []
  for i in range(1, 2**b.nbits):
    ratio.append(np.real((x[i] * x[i].conj()) / (x[0] * x[0].conj())))
    print(f'Classic ratio: {ratio[-1]:6.3f}')
  return ratio


def check_results(qc, a, b):
  """Check the results by inspecting the final state."""

  ratio_classical = check_classic_solution(a, b)
  res = (np.abs(qc.psi) > 0.07).nonzero()[0]
  ratio_quantum = [np.real(qc.psi[res[j]] ** 2 / qc.psi[res[0]] ** 2)
                   for j in range(1, len(res))]

  for idx, ratio in enumerate(ratio_quantum):
    delta = ratio - ratio_classical[idx]
    print(f'Quantum ratio: {ratio:6.3f}, delta: {delta:+5.3f}')
    if abs(delta) > 0.2:
      raise AssertionError('Incorrect result.')


def compute_sorted_eigenvalues(a):
  """Compute the sorted eigenvalues/vectors."""

  # Eigenvalue/vector computation.
  w, v = np.linalg.eig(a)

  # Sort the eigenvalues and eigenvectors.
  idx = w.argsort()
  w = w[idx]
  v = v[:, idx]

  # From the experiments in 'spectral_decomp.py', we know that for
  # a Hermitian A:
  #   Eigenvalues are real (that's why a Hamiltonian must be Hermitian)
  w = np.real(w)
  return w, v


def compute_u_matrix(a, w, v, t):
  """Compute the U matrix."""

  # Compute the matrices U (and U^n) from A via:
  #   U = exp(i * A * t) (^n)
  # Since U is diagonal:
  u = ops.Operator(np.zeros((a.shape[0], a.shape[1]), dtype=np.complex64))
  for i in range(a.shape[0]):
    u[i][i] = np.exp(1j * w[i] * t)

  # Both U and U^2 are in the eigenvector basis of A. To convert these
  # operators to the computational basis we apply the similarity
  # transformation:
  u = v @ u @ v.transpose().conj()
  return u


def compute_angles(w, c):
  """Compute the angles for the conditional rotations."""

  # This method is not fully general (yet). It is minimal
  # and handles the simple cases used in the example,
  # such as eigenvalues of 1, 2, 3, 4, and 8.
  #
  # We know that theta = 2 arcsin(1/lam_j):
  unis = np.unique(w)
  angles = [2 * np.arcsin(c / eigen) for eigen in unis]
  if int(np.round(w[1])) & 1 == 1:
    angles[1] = angles[1] - angles[0]
  return angles


def construct_circuit(b, w, u, c, clock_bits):
  """Construct a circuit for the given paramters."""

  qc = circuit.qc('hhl')

  # State preparation - just initialize the b register.
  breg = qc.state(b)
  clock = qc.reg(clock_bits, 0)
  anc = qc.reg(1, 0)

  # Move clock bits into Hadamard basis.
  qc.h(clock)

  # Phase estimation to bring the eigenvalues into the clock register.
  u_phase = u
  u_phase_inv_gates = []
  for idx in range(clock_bits):
    op = ops.ControlledU(clock[idx], breg[breg.size - 1], u_phase)
    qc.unitary(op, breg[0])
    u_phase_inv_gates.append(np.linalg.inv(u_phase))
    u_phase = u_phase @ u_phase

  # Inverse QFT. After this, the eigenvalues will be in the clock register.
  qc.inverse_qft(clock, True)

  # Conditional rotations to compute inverse eigenvalues.
  angles = compute_angles(w, c)
  for idx, angle in enumerate(angles):
    qc.cry(clock[idx], anc, angle)

  # Measure (and force) ancilla to state |1>.
  qc.measure_bit(anc[0], 1, collapse=True)

  # QFT
  qc.qft(clock, True)

  # Uncompute.
  for idx in reversed(range(clock_bits)):
    op = ops.ControlledU(clock[idx], breg[breg.size - 1],
                         u_phase_inv_gates[idx])
    qc.unitary(op, breg[0])

  # Move clock bits out of Hadamard basis.
  qc.h(clock)

  return qc


def run_experiment(a, b, clock_bits):
  """Run a single instance of HHL for Ax = b."""

  if not a.is_hermitian():
    raise AssertionError('Input A must be Hermitian.')

  print(f'\nClock bits   : {clock_bits}')
  print(f'Dimensions A : {a.shape[0]}x{a.shape[1]}')

  # Compute eigenvalue/vectors.
  w, v = compute_sorted_eigenvalues(a)

  # We know that:
  #   lam_i = (N * w[j] * t) / (2 * np.pi)
  #
  # We want lam_i to be integers, so we compute 't' as:
  #   t = lam[0] / N / w[1] * 2 * np.pi
  n = 2**clock_bits
  t = 2 * np.pi / w[0] / n

  # With 't' we can now compute the integer eigenvalues:
  lam = [(n * np.real(w[i]) * t / (2 * np.pi)) for i in range(a.shape[0])]
  for i in range(a.shape[0]):
    print(f'  lambda[{i}]  : {lam[i]:.1f}')

  # Compute the U matrix.
  u = compute_u_matrix(a, w, v, t)

  # C must be <= than the minimal lam:
  c = np.min(np.abs(lam))
  print(f'Set C to min : {c:.1f}')

  # Now we have all the values and matrices. Let's construct a circuit.
  qc = construct_circuit(b, lam, u, c, clock_bits)
  check_results(qc, a, b)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('General HHL Algorithm...')

  a = ops.Operator([[3 / 5, -1 / 5], [-1 / 5, 3 / 5]])
  b = ops.Operator([1, 0])
  run_experiment(a, b, clock_bits=4)

  a = ops.Operator([[11, 5, -1, -1],
                    [5, 11, 1, 1],
                    [-1, 1, 11, -5],
                    [-1, 1, -5, 11]]) / 16
  b = ops.Operator([0, 0, 0, 1]).transpose()
  run_experiment(a, b, clock_bits=4)

  a = ops.Operator([[15, 9, 5, -3],
                    [9, 15, 3, -5],
                    [5, 3, 15, -9],
                    [-3, -5, -9, 15]]) / 4
  b = ops.Operator([0, 0, 0, 1]).transpose()
  run_experiment(a, b, clock_bits=4)


if __name__ == '__main__':
  np.set_printoptions(precision=4)
  app.run(main)
