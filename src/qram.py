# python3
"""Example: QRAM."""

from absl import app
from src.lib import circuit
from src.lib import ops


# This is a very simple and simplified implementation of
# simple QRAM's, mostly to show the principles of a specific
# way to materialize a QRAM.
#
# Currently, only 1 address qubit -> 2 values and 2 address
# qubits -> 1 value are implemented. It should be easy to see
# that the principles can be easily extended to arbitrary
# n -> m mappings. The corresponding circuits, however, can
# become quite unwieldy, hence we're not showing those here.


def qram_1_to_2():
  """Map a single qubit to 2 data bits."""

  # This example is taken from:
  #   https://youtu.be/eBN5MXMirYs
  # A single address qubit is |0> or |1> and maps to:
  #  |0> -> |01>
  #  |1> -> |11>
  #
  print('1 address bit, 2 data bits.')
  qc = circuit.qc('test_1_2')
  a = qc.reg(1)
  d = qc.reg(2)
  qc.h(a)
  qc.cx0(a, d[1])
  qc.cx(a, d[0])
  qc.cx(a, d[1])

  if qc.psi[0b011] < 0.7 or qc.psi[0b101] < 0.7:
    raise AssertionError('Incorrect results')


def qram_2_to_1():
  """Map two address qubits to 0 or 1."""

  # Two address bits contain either a 1 or 0.
  #  |00> -> |0>
  #  |01> -> |1>
  #  |10> -> |1>
  #  |11> -> |0>
  print('2 address bits, 1 data bit.')
  qc = circuit.qc('test')
  a = qc.reg(2)
  _ = qc.reg(1)

  qc.h(a)
  qc.multi_control([[0], 1], 2, None, ops.PauliX(), 'ccx')
  qc.multi_control([0, [1]], 2, None, ops.PauliX(), 'ccx')

  if qc.psi[0b011] < 0.4 or qc.psi[0b101] < 0.4:
    raise AssertionError('Incorrect results')
  if qc.psi[0b000] < 0.4 or qc.psi[0b110] < 0.4:
    raise AssertionError('Incorrect results')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('QRAM...')

  # These simple examples show how to binary encode address
  # bits and value bits and how to multi-control the value
  # bits from the address qubits. Extending this to longer
  # addresses and values is straight-forward (but quite
  # unwieldy).
  #
  qram_1_to_2()
  qram_2_to_1()


if __name__ == '__main__':
  app.run(main)
