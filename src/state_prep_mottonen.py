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
#   https://github.com/ravikumar1728/Mottonen-State-Preparation
#   https://docs.pennylane.ai/en/stable/_modules/pennylane/
#           templates/state_preparations/mottonen.html


def gray_code(i: int) -> int:
  """Return Gray code at index i."""

  return i ^ (i >> 1)


def compute_alpha(vec, k: int, j: int):
  """Compute the angles alpha_k."""

  # This is a faithful implementation of Equation (8) in the reference.
  # Note the off-by-1 issues (the paper is 1-based).
  m = 2 ** (k - 1)
  enumerator = 0
  for l in range(m):
    enumerator += vec[(2 * (j + 1) - 1) * m + l] ** 2

  m = 2**k
  divisor = 0
  for l in range(m):
    divisor += vec[j * m + l] ** 2

  if divisor != 0:
    return 2 * np.arcsin(np.sqrt(enumerator / divisor))
  return 0.0


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

  # This code is tricky. It implements the control
  # qubit indices following Fig 2 in the reference, in a recursive
  # manner. The secret to success is to 'kill' the last token in
  # the recurive call.
  if idx == 0:
    return []
  side = compute_ctl(idx - 1)[:-1]
  return side + [idx - 1] + side + [idx - 1]


def controlled_ry(qc, alpha_k, control, target):
  """Implement the controlled-ry rotations."""

  k = len(control)

  # This is Equation (3) in the reference.
  thetas = compute_m(k) @ alpha_k

  if k == 0:
    qc.ry(target, thetas[0])
    return

  ctl = compute_ctl(k)
  for i in range(2**k):
    qc.ry(target, thetas[i])
    qc.cx(control[k - 1 - ctl[i]], target)


def prepare_state(nbits: int = 3):
  """Prepare a random state with nbits qubits."""

  # Input should be a real (!) and non-negative (!) array of floating
  # point values. To allow negatives, another pass of cz gates
  # must be added to front and back of the circuit. For simplicity,
  # we omit this in this implementation.
  vector = np.random.random([2**nbits])
  vector = vector / np.linalg.norm(vector)

  # This is a little tricky. We need a circuit because below we
  # compute the angles and corresponding Ry and CX gates. However
  # we do _not_ want to generate a quantum register yet, as we
  # want to run the generated circuit on a state |0>.
  #
  # So we generate the circuit, but we only shim the quantum register
  # with a simple array of integer indices.
  qc = circuit.qc('mottonen', eager=False)
  qb = range(nbits)

  for k in range(nbits):
    alpha_k = [compute_alpha(vector, nbits - k, j) for j in range(2**k)]
    controlled_ry(qc, alpha_k, qb[:k], qb[k])

  # At this point we can actually allocate the register and
  # run the generated circuit.
  qc.reg(nbits)
  qc.run()

  if not np.allclose(vector, qc.psi, atol=1e-5):
    raise AssertionError('Invalid State initialization.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print("State Preparation with Moettoenen's Algorithm...")

  for nbits in range(1, 11):
    print(f'{nbits} qubits...')
    for _ in range(5):
      prepare_state(nbits)


if __name__ == '__main__':
  np.set_printoptions(precision=2)
  app.run(main)
