# python3
"""Example: State preparation with Moettoenen's algorithm."""

from absl import app
import numpy as np
from src.lib import circuit

# Reference:
# [1] Transformation of quantum states using uniformly controlled rotations
# https://arxiv.org/pdf/quant-ph/0407010.pdf
#
# This implementation would not have been possible without looking
# at existing code references:
#
#   https://github.com/ravikumar1728/Mottonen-State-Preparation
#
#   https://docs.pennylane.ai/en/stable/_modules/pennylane/
#           templates/state_preparations/mottonen.html
#
# The reference takes an existing state and transforms it down to |0>.
# Since we want to prepare a specific state starting from |0> we
# reverse the procedure in this code snippet.


def gray_code(i: int) -> int:
  """Return Gray code at index i."""

  return i ^ (i >> 1)


def compute_alpha_y(vec, k: int, j: int):
  """Compute the angles alpha_k for the y rotations."""

  # This is the implementation of Equation (8) in the reference.
  # Note the off-by-1 issues (the paper is 1-based).
  m = 2 ** (k - 1)
  enumerator = sum(vec[(2 * (j + 1) - 1) * m + l] ** 2 for l in range(m))
  m = 2**k
  divisor = sum(vec[j * m + l] ** 2 for l in range(m))
  if divisor != 0:
    return 2 * np.arcsin(np.sqrt(enumerator / divisor))
  return 0.0


def compute_alpha_z(omega, k: int, j: int):
  """Compute the angles alpha_k for the z rotations."""

  # This is the implementation of Equation (5) in the reference.
  # Note the off-by-1 issues (the paper is 1-based).
  m = 2 ** (k - 1)
  ind1 = [(2 * (j + 1) - 1) * m + l for l in range(m)]
  ind2 = [(2 * (j + 1) - 2) * m + l for l in range(m)]
  diff = (omega[ind1] - omega[ind2]) / m
  return sum(diff)


def compute_m(k: int):
  """Compute matrix M which takes alpha -> theta."""

  # This computation of M follows Equation (3) in the reference.
  n = 2**k
  m = np.zeros([n, n])
  for i in range(n):
    for j in range(n):
      # Note: bit_count() only supported from Python 3.10.
      m[i, j] = (-1) ** bin(j & gray_code(i)).count('1') * 2 ** (-k)
  return m


def compute_ctl(idx: int):
  """Compute control indices for the cx gates."""

  # This code implements the control qubit indices following
  # Fig 2 in the reference in a recursive manner. The secret
  # to success is to 'kill' the last token in the recursive call.
  if idx == 0:
    return []
  side = compute_ctl(idx - 1)[:-1]
  return side + [idx - 1] + side + [idx - 1]


def controlled_rotation(qc, alpha_k, control, target, gate):
  """Implement the controlled rotations."""

  k = len(control)
  thetas = compute_m(k) @ alpha_k
  ctl = compute_ctl(k)
  for i in range(2**k):
    gate(target, thetas[i])
    if k > 0:
      qc.cx(control[k - 1 - ctl[i]], target)


def prepare_state_mottonen(qc, qb, vector, nbits: int = 3):
  """Construct the Mottonen circuit based on input vector."""

  # Ry gates for the absolute amplitudes.
  avec = abs(vector)
  for k in range(nbits):
    alpha_k = [compute_alpha_y(avec, nbits - k, j) for j in range(2**k)]
    controlled_rotation(qc, alpha_k, qb[:k], qb[k], qc.ry)

  # Rz gates to normalize up to a global phase. This is only
  # needed for complex values.
  omega = np.angle(vector)
  if np.allclose(omega, 0.0):
    return

  for k in range(0, nbits):
    alpha_z = [compute_alpha_z(omega, nbits - k, j) for j in range(2**k)]
    controlled_rotation(qc, alpha_z, qb[:k], qb[k], qc.rz)


def run_experiment(nbits: int = 3):
  """Prepare a random state with nbits qubits."""

  vector = np.random.random([2**nbits]) + 1j * np.random.random([2**nbits])
  vector = vector / np.linalg.norm(vector)
  print(f'  Qubits: {nbits:2d}, vector: {vector[:6]}...')

  qc = circuit.qc()
  qb = qc.reg(nbits)
  prepare_state_mottonen(qc, qb, vector, nbits)

  # For complex numbers, this algorithm introduces a global phase
  # which we can account for (and ignore) here:
  phase = vector[0] / qc.psi[0]
  if not np.allclose(vector, qc.psi * phase, atol=1e-5):
    raise AssertionError('Invalid State initialization.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print("State Preparation with Moettoenen's Algorithm...")

  for nbits in range(1, 11):
    run_experiment(nbits)


if __name__ == '__main__':
  np.set_printoptions(precision=2)
  app.run(main)
