# python3
"""Example: Simon's General Algorithm."""

from absl import app
import numpy as np

from src.lib import helper
from src.lib import ops
from src.lib import state

# A general Simon Oracle can be constructed the following way:
#
#    Assume a secret string 'c' and the msb set to 1
#
#    For each input value x:
#        if msb(x) == 0:
#           return x
#        if msb(x) == 1:
#           return x % c
#
# In quantum gates, that's easy to accomplish.
# First copy the input gates to the output gates:
#
# ----o--------
#     |
# ----|--o-----
#     |  |
# ----X--|-----
#        |
# -------X-----
#
# Now, for each bit i in c that is 1, cxor the output gate i
# controlled by bit 0.
#
# So for example, for the string c = 10:
#
# ----o------o-
#     |      |
# ----|--o---|-
#     |  |   |
# ----X--|---X-
#        |
# -------X-----
#
# So for example, for the string c = 11:
#
# ----o------o--o-
#     |      |  |
# ----|--o---|--|-
#     |  |   |  |
# ----X--|---X--|-
#        |      |
# -------X------X-


def make_c(nbits):
  """Make a random constant c from {0,1}. This is the c we try to find."""

  constant_c = [0] * nbits
  for idx in range(nbits):
    constant_c[idx] = int(np.random.random() < 0.5)
  print('Magic constant: {}'.format(constant_c))
  return constant_c


def make_u(nbits, constant_c):
  """Make general Simon's Oracle."""

  # Copy bits.
  op = ops.Identity(nbits*2)
  for idx in range(nbits):
    op = (ops.Identity(idx) *
          ops.Cnot(idx, idx+nbits) *
          ops.Identity(nbits - idx - 1)) @ op

  # Connect the xor's controlled by the msb(x).
  for idx in range(nbits):
    if constant_c[idx] == 1:
      op = (ops.Cnot(0, idx+nbits) * ops.Identity(nbits-idx-1)) @ op

  if not op.is_unitary():
    raise AssertionError('Produced non-unitary Uf')
  return op


def dot2(bits, nbits):
  """Compute dot module 2."""

  accum = 0
  for idx in range(nbits):
    accum = accum + bits[idx] * bits[idx + nbits]
  return accum % 2


def run_experiment(nbits):
  """Run single, defined experiment for secret 11."""

  psi = state.zeros(nbits * 2)
  c = make_c(nbits)
  u = make_u(nbits, c)

  psi = ops.Hadamard(nbits)(psi)
  psi = u(psi)
  psi = ops.Hadamard(nbits)(psi)

  # Because of the xor patterns (Yanofski 6.64)
  # measurement will only find those qubit strings where
  # the scalar product of z (lower bits) and secret string:
  #    <z, c> = 0
  #
  print('Measure likely states:')
  for bits in helper.bitprod(nbits*2):
    if psi.prob(*bits) > 0.0 and dot2(bits, nbits) < 1.0:
      print('|{}> = 0 : {:.2f}  dot % 2: {:.2f}'.
            format(bits, psi.prob(*bits),
                   dot2(bits, nbits)))

  # Multiple rounds (nbits) are necessary to find
  # the system of equations that allows finding of 'c'.
  #   ... (not implemented here)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Note: Running with 5 (10 total) qubits takes about 5 minutes.
  run_experiment(3)


if __name__ == '__main__':
  app.run(main)
