# python3
"""Quantum Mean Computation."""

#!!!!!!!!!!!!!!!!!!!!!!!!!
# WIP - work in progress
#!!!!!!!!!!!!!!!!!!!!!!!!!

import random
from absl import app
import numpy as np

from src.lib import circuit
from src.lib import ops
from src.lib import state


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Quantum Mean Computation via AE.')
  x = [1.0, 2.0, 3.0, 2.0]
  x_norm = np.linalg.norm(x)
  x = x / x_norm
  print(np.mean(x) * np.mean(x), np.mean(x), np.sqrt(np.mean(x)))
  # The normalized vector is:
  #  [0.0928 0.2785 0.4642 0.8356]

  # Define a unitary which does:
  # $$
  #  U_a:|0\rangle|0\rangle\mapsto
  # \frac{1}{2^{n/2}}
  #     \sum_x|x\rangle(\sqrt{1-F(x)}|0\rangle+\sqrt{F(x)}|1\rangle).
  # $$
  # In other words:
  #   |00> -> np.sqrt(1 - 0.093) |000>
  # Which can be done with a controlled rotation:
  #
  qc = circuit.qc('mean calculator')
  input = qc.reg(2, 0)
  ext = qc.reg(1)
  anc = qc.reg(1, 0)
  qc.h(input)

  for val in x:
    angle = 2 * np.arcsin(val)
    qc.ry(anc, angle)
    qc.multi_control([[0], [1]], ext,
    qc.cx(anc, ext)
    qc.ry(anc, -angle)
  qc.psi.dump()

if __name__ == '__main__':
  np.set_printoptions(precision=4)
  app.run(main)
