# python3
"""Example: Entanglement Swapping."""

import math

from absl import app
import numpy as np

from src.lib import circuit


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Alice has qubits 1 and 4
  # Bob   has qubits 2 and 3
  qc = circuit.qc('swap the entanglement', eager=True)
  qc.reg(4, 0)

  # Qubits 1 and 2 are entangled, and so are qubits 3 and 4:
  #
  # Alice  1        4
  #        |        |
  # Bob    2        3
  #
  qc.h(0)
  qc.cx(0, 1)
  qc.h(2)
  qc.cx(2, 3)

  # Now Alice and Bob physically separate.
  # Alice keeps qubits 1 and 4 on Earth.
  # Bob takes qubits 2 and 3 to the Moon.

  # ... travel, space ships, etc...

  # Alice performs a Bell measurement between qubits 1 and 4,
  # which means to apply a reverse entangler circuit:
  #
  # Alice  1~~~BM~~~4
  #        |        |
  # Bob    2        3
  #
  qc.cx(0, 3)
  qc.h(0)

  # At this point, Alice will physically measure her qubits 1 and 4.
  # There are four possible outcomes, |00>, |01>, |10>, and |11>,
  #
  # Depending on that outcome, now qubits 2 and 3 will also be
  # in a corresponding Bell state! The entanglement has been
  # teleported to qubits 2 and 3, even though they never
  # interacted before!
  #
  # Alice  1        4
  #        |        |
  # Bob    2 ~~~~~~ 3
  #
  #
  # To check the results:
  #
  # Iterate over the four possibilities
  # This array 'cases' represents 2 cases in each entry, either
  #   qubit entries 0, 1, 2, 3  or
  #   qubit entries 0, 5, 6, 3 (multiplied by c[7])
  #
  # This covers all expected results.
  #
  # qubits   0  1  2  3       1  2   factor
  # -----------------------------------------
  cases = [
      [0, 0, 0, 0, 1.0, 1, 1, 1.0],
      [0, 0, 1, 1, 1.0, 1, 0, 1.0],
      [1, 0, 0, 0, 1.0, 1, 1, -1.0],
      [1, 0, 1, 1, 1.0, 1, 0, -1.0],
  ]

  c07 = 1 / math.sqrt(2)
  psi = qc.psi
  for c in cases:
    qc.psi = psi

    qc.measure_bit(0, c[0], collapse=True)
    qc.measure_bit(3, c[3], collapse=True)
    qc.psi.dump(f'after measuring |{c[0]}..{c[3]}>')

    if not math.isclose(
        np.real(qc.psi.ampl(c[0], c[1], c[2], c[3])), c07, abs_tol=1e-5
    ):
      raise AssertionError('Invalid measurement results')
    if not math.isclose(
        np.real(qc.psi.ampl(c[0], c[5], c[6], c[3])), c07 * c[7], abs_tol=1e-5
    ):
      raise AssertionError('Invalid measurement results')


if __name__ == '__main__':
  app.run(main)
