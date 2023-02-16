# python3
"""Example: Quantum Mean Computation."""

import random
from absl import app
import numpy as np

from src.lib import circuit
from src.lib import ops


def run_experiment(nbits: int):
  """Run a single mean computation."""

  x = [random.randint(0, 10) for _ in range(2 ** nbits)]
  x_norm = np.linalg.norm(x)
  xn = x / x_norm

  # Define a unitary which does:
  # $$
  #  U_a:|0\rangle|0\rangle\mapsto
  # \frac{1}{2^{n/2}}
  #     \sum_x|x\rangle(\sqrt{1-F(x)}|0\rangle+\sqrt{F(x)}|1\rangle).
  # $$
  # In other words:
  #   |00> -> np.sqrt(1 - a) |000>
  # Which can be done with a controlled rotations
  #
  qc = circuit.qc('mean calculator')
  inp = qc.reg(2, 0)
  ext = qc.reg(1, 0)

  if nbits > 2:
    raise AssertionError('Currently only 2 qubits supported.')

  # This can be extended to more qubits quite easily.
  # The trick is to control the rotations with the bit
  # patterns of the indices (encoded via qubits) into x.
  qc.h(inp)
  qc.ccu([0], [1], ext, ops.RotationY(2 * np.arcsin(xn[0])))
  qc.ccu([0], 1, ext, ops.RotationY(2 * np.arcsin(xn[1])))
  qc.ccu(0, [1], ext, ops.RotationY(2 * np.arcsin(xn[2])))
  qc.ccu(0, 1, ext, ops.RotationY(2 * np.arcsin(xn[3])))
  qc.h(inp)

  # Index 1 may have to change if other qubits are added.
  qmean = np.real(qc.psi[1])
  qclas = np.real(qc.psi[1] * x_norm)

  if not np.allclose(np.mean(xn), qmean, atol=0.001):
    raise AssertionError('Incorrect quantum mean computation.')
  if not np.allclose(np.mean(x), qclas, atol=0.001):
    raise AssertionError('Incorrect quantum scaled mean computation.')
  print(f'  Mean: classic: {np.mean(x):.3f}, quantum: {qclas:.3f}')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('WIP: Quantum Mean Computation.')

  for _ in range(10):
    run_experiment(2)


if __name__ == '__main__':
  np.set_printoptions(precision=4)
  app.run(main)
