# python3
"""Example: Entanglement Swapping."""

import math

from absl import app
import numpy as np

from src.lib import circuit


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  qc = circuit.qc('Entanglement Swap')
  qc.reg(4, 0)

  # Alice has qubits 0, 1
  # Bob   has qubits 2, 3
  #
  # Entangle 0, 2 and 1, 3
  # ----------------------
  # Alice  0        1
  #        |        |
  # Bob    2        3
  #
  qc.h(0)
  qc.cx(0, 2)
  qc.h(1)
  qc.cx(1, 3)

  # Now Alice and Bob physically separate.
  # Alice keeps qubits 1 and 4 on Earth.
  # Bob takes qubits 2 and 3 to the Moon.
  #          ... travel, space ships, etc...
  # Alice performs a Bell measurement between her qubits 0 and 1,
  # which means to apply a reverse entangler circuit:
  #
  # Alice  0~~~BM~~~1
  #        |        |
  # Bob    2        3
  #
  qc.cx(0, 1)
  qc.h(0)

  # At this point, Alice will physically measure her qubits 0, 1.
  # There are four possible outcomes, |00>, |01>, |10>, and |11>,
  #
  # Depending on that outcome, now qubits 2 and 3 will also be
  # in a corresponding Bell state! The entanglement has been
  # teleported to qubits 2 and 3, even though they never
  # interacted before!
  #
  # Alice  0        1
  #
  # Bob    2 ~~~~~~ 3
  #
  # To check the results:
  # Iterate over the four possibilities for measurement
  # The table covers all expected results. There will only be
  # two states with probability >0.0. For example, the
  # first line below represents:
  #    0 0 0 0  p=0.707...
  #    0 0 1 1  p=0.707...
  # Ignore the first two qubits. This shows a
  # superposition of state 00 and 11 in a Bell state.
  #
  # Qubits
  #    0  1  2  3    0  1  2  3   factor   Bell
  # ---------------------------------------------
  cases = [
      [0, 0, 0, 0,   0, 0, 1, 1,  1.0],  # b00
      [0, 1, 0, 1,   0, 1, 1, 0,  1.0],  # b01
      [1, 0, 0, 0,   1, 0, 1, 1, -1.0],  # b10
      [1, 1, 0, 1,   1, 1, 1, 0, -1.0],  # b11
  ]

  c07 = 1 / math.sqrt(2)
  psi = qc.psi

  for c in cases:
    qc.psi = psi
    qc.measure_bit(0, c[0], collapse=True)
    qc.measure_bit(1, c[1], collapse=True)
    qc.psi.dump(f'after measuring |{c[0]}..{c[3]}>')

    if not np.allclose(
        np.real(qc.psi.ampl(c[0], c[1], c[2], c[3])), c07
    ):
      raise AssertionError('Invalid measurement results')
    if not np.allclose(
        np.real(qc.psi.ampl(c[4], c[5], c[6], c[7])), c07 * c[8]
    ):
      raise AssertionError('Invalid measurement results')


if __name__ == '__main__':
  app.run(main)
