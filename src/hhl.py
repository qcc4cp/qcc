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
import random

from src.lib import circuit
from src.lib import ops
from src.lib import state
from src.lib import tensor


def check_classic_solution(a, b):
  """Check classic solution."""

  x = np.linalg.solve(a, b)
  for i in range(1, 2**b.nbits):
    ratio_x = np.real((x[i] * x[i].conj()) / (x[0] * x[0].conj()))
    print(f'Classic ratio: {ratio_x:.3f}')
  return ratio_x


def check_results(qc, a, b):
  """Check the results by inspecting the final state."""

  ratio_classical = check_classic_solution(a, b)
  res = (np.abs(qc.psi) > 0.04).nonzero()[0]
  for j in range(1, len(res)):
    ratio_quantum = np.real(qc.psi[res[j]]**2 / qc.psi[res[0]]**2)
    print(f'Quantum ratio: {ratio_quantum:.3f}')
    if a.shape[0] == 2:
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
  # Since U is diagonal:
  u = ops.Operator(np.zeros((a.shape[0], a.shape[1]), dtype=np.complex64))
  for i in range(a.shape[0]):
    u[i][i] = np.exp(1j * w[i] * t)

  # Both U and U^2 are in the eigenvector basis of A. To convert these
  # operators to the computational basis we apply the similarity
  # transformations:
  u = v @ u @ v.transpose().conj()
  return u


def construct_circuit(b, w, u, c, clock_bits):
  """Construct a circuit for the given paramters."""

  qc = circuit.qc('hhl', eager=True)
  breg = qc.state(b)
  clock = qc.reg(clock_bits, 0)
  anc = qc.reg(1, 0)

  # State Preparation, which is basically phase estimation.
  qc.h(clock)
  u_phase = u
  u_phase_gates = []
  for idx in range(clock_bits):
    op = ops.ControlledU(clock[idx], breg[breg.size-1], u_phase)
    qc.unitary(op, breg[0])
    u_phase_gates.append(u_phase)
    u_phase = u_phase @ u_phase

  # Inverse QFT. After this, the eigenvalues will be
  # in the clock register.
  qc.inverse_qft(clock, True)

  # From above we know that:
  #   theta = 2 arcsin(1 / lam_j)
  angles = []
  for eigen in w:
    angles.append(2 * np.arcsin(c / eigen))
  if int(np.round(w[1])) & 1 == 1:
    angles[1] = angles[1] - angles[0]
  for idx in range(len(angles)):
    qc.cry(clock[idx], anc, angles[idx])

  # Measure (and force) ancilla to be |1>.
  _, _ = qc.measure_bit(anc[0], 1, collapse=True)

  # QFT
  qc.qft(clock, True)

  # Uncompute state initialization.
  for idx in range(clock_bits-1, -1, -1):
    op = ops.ControlledU(clock[idx], breg[breg.size-1],
                         np.linalg.inv(u_phase_gates[idx]))
    qc.unitary(op, breg[0])

  # Move clock bits out of Hadamard basis.
  qc.h(clock)
  qc.psi.dump('Final state')
  return qc


def run_experiment(a, b, clock_bits):
  """Run a single instance of HHL for Ax = b."""

  if not a.is_hermitian():
    raise AssertionError('Input A must be Hermitian.')
  print(f'\nClock bits   : {clock_bits}')
  print(f'Dimensions A : {a.shape[0]}x{a.shape[1]}')

  # For quantum, initial parameters.
  dim = a.shape[0]

  # Compute eigenvalue/vectors.
  w, v = compute_sorted_eigenvalues(a)

  # Compute the ratio. We will compare the results
  # against this value below.
  ratio = w[1] / w[0]

  # We also know that:
  #   lam_i = (N * w[j] * t) / (2 * np.pi)
  # We want lam_i to be integers, so we compute 't' as:
  #   t = lam[0] / N / w[1] * 2 * np.pi
  n = 2**clock_bits
  t = ratio / n / w[1] * 2 * np.pi

  # With 't' we can now compute the integer eigenvalues:
  lam = [(n * np.real(w[i]) * t / (2 * np.pi)) for i in range(dim)]
  for i in range(dim):
    print(f'  lambda[{i}]  : {lam[i]:.1f}')

  # Compute the U matrix.
  u = compute_u_matrix(a, w, v, t)

  # The factors to |0> and 1> of the ancilla will be:
  #   \sqrt{1 - C^2 / lam_j^2} and C / lam_j
  # C must be <= than the minimal lam. We set it to the minimum:
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
  run_experiment(a, b,clock_bits=2)

  a = ops.Operator(np.array([[3/5, -1/5], [-1/5, 3/5]]))
  b = ops.Operator(np.array([1, 0]))
  run_experiment(a, b,clock_bits=4)

  a = ops.Operator(np.array([[3/5, -1/5], [-1/5, 3/5]]))
  b = ops.Operator(np.array([1, 0]))
  run_experiment(a, b,clock_bits=6)

  a = ops.Operator(np.array([[11,  5, -1, -1],
                             [ 5, 11,  1,  1],
                             [-1,  1, 11, -5],
                             [-1,  1, -5, 11]])) / 16
  b = ops.Operator(np.array([0, 0, 0, 1]).transpose())
  run_experiment(a, b, clock_bits=4)


if __name__ == '__main__':
  np.set_printoptions(precision=4)
  app.run(main)
