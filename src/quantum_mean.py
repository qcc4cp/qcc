# python3
"""Example: Quantum Mean Computation."""

import itertools
import random
from absl import app
import numpy as np

from src.lib import circuit
from src.lib import helper
from src.lib import ops


def run_experiment(nbits: int):
  """Run a single mean computation."""

  # Random numbers, positive and negative.
  x = np.array([random.randint(0, 10) - 5 for _ in range(2**nbits)])
  xn = x / np.linalg.norm(x)

  # Define a unitary which does:
  # $$
  #  U_a:|0\rangle|0\rangle\mapsto
  # \frac{1}{2^{n/2}}
  #     \sum_x|x\rangle(\sqrt{1-F(x)}|0\rangle+\sqrt{F(x)}|1\rangle).
  # $$
  # In other words:
  #   |000> -> np.sqrt(1 - a) |000>
  #   |000> -> np.sqrt(a) |001>
  #
  # Which can be done for x_i with a controlled rotations about y by
  #   2 arcsin(x_i)
  #
  # Measuring the state |001| should give the mean. This can be done by
  # repeated experiments, but we could also use amplitude
  # estimation (as suggested in the book by Moscha).
  #
  qc = circuit.qc('mean calculator')
  inp = qc.reg(nbits, 0)  # State input
  aux = qc.reg(nbits - 1, 0)  # Aux qubits for the multi-controlled gates
  ext = qc.reg(1, 0)  # Target 'extra' qubit

  # The trick is to control the rotations with the bit
  # patterns of the indices (encoded via qubits) into x.
  qc.h(inp)
  for bits in itertools.product([0, 1], repeat=nbits):
    idx = helper.bits2val(bits)
    # Control-by-zero is indicated with a single-element list.
    ctl = [i if bit == 1 else [i] for i, bit in enumerate(bits)]
    qc.multi_control(ctl, ext, aux,
                     ops.RotationY(2 * np.arcsin(xn[idx])), 'multi-ry')
  qc.h(inp)

  # We 'measure' via peak-a-boo of state |00...001>
  qmean = np.real(qc.psi[1])
  qclas = np.real(qc.psi[1] * np.linalg.norm(x))

  # Check results.
  if not np.allclose(np.mean(xn), qmean, atol=0.001):
    raise AssertionError('Incorrect quantum mean computation.')
  if not np.allclose(np.mean(x), qclas, atol=0.001):
    raise AssertionError('Incorrect quantum scaled mean computation.')
  print(f'  Mean ({nbits} qb): classic: {np.mean(x):.3f}, quantum: {qclas:.3f}')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('Quantum Mean Computation.')

  for nbits in range(2, 8):
    run_experiment(nbits)

if __name__ == '__main__':
  np.set_printoptions(precision=4)
  app.run(main)
