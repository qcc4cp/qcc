# python3
"""Example: Simon's General Algorithm."""

import contextlib
import io

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
#           return x XOR c
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

  # The oracle construction in make_u() requires the msb of c to be 1 (see
  # the comment at the top of the file). Otherwise f(x) != f(x XOR c), the
  # oracle is one-to-one instead of two-to-one, and the measurements violate
  # <z, c> = 0. This also guarantees c != 0.
  constant_c[0] = 1
  print('Magic constant: {}'.format(constant_c))
  return constant_c


def make_u(nbits, constant_c):
  """Make general Simon's Oracle."""

  # The construction below only has the period c if the msb of c is 1.
  # The all-zero c, a one-to-one function, is the only exception.
  if constant_c[0] != 1 and any(constant_c):
    raise ValueError(f'The msb of the secret must be 1, got {constant_c}')

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


def violation_probability(psi, nbits, constant_c):
  """Probability of measuring an input register z with <z, c> = 1 (mod 2)."""

  violation = 0.0
  for bits in helper.bitprod(nbits * 2):
    if sum(z * c for z, c in zip(bits[:nbits], constant_c)) % 2 == 1:
      violation += psi.prob(*bits)
  return violation


def run_experiment(nbits):
  """Run single, defined experiment for secret 11."""

  psi = state.zeros(nbits * 2)
  c = make_c(nbits)
  u = make_u(nbits, c)

  psi = ops.Hadamard(nbits)(psi)
  psi = u(psi)
  psi = ops.Hadamard(nbits)(psi)

  # Simon's promise: the input register can only be measured as a z with
  # <z, c> = 0. The simulator computes in single precision, hence 1e-4.
  if violation_probability(psi, nbits, c) > 1e-4:
    raise AssertionError('Measurements violate <z, c> = 0')

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


def test_oracle():
  """Check make_c() and make_u() for all secrets of 2 and 3 bits."""

  # make_c() must always return a secret with msb 1.
  for nbits in range(1, 7):
    for seed in range(50):
      np.random.seed(seed)
      with contextlib.redirect_stdout(io.StringIO()):
        c = make_c(nbits)
      if len(c) != nbits or c[0] != 1:
        raise AssertionError(f'make_c({nbits}) returned {c}')

  for nbits in (2, 3):
    for bits in helper.bitprod(nbits):
      c = list(bits)
      if c[0] == 0 and any(c):
        # Secrets with msb 0 can't be built, and must be rejected.
        try:
          make_u(nbits, c)
        except ValueError:
          continue
        raise AssertionError(f'make_u accepted invalid secret {c}')

      # For all valid secrets, f(x) == f(x XOR c) and f is two-to-one
      # (or one-to-one for c == 0). Column x << nbits of the operator
      # is the oracle applied to |x>|0>, from which we read f(x).
      u = make_u(nbits, c)
      matrix = np.asarray(u)
      c_val = helper.bits2val(c)
      outputs = [int(np.argmax(np.abs(matrix[:, x << nbits]))) &
                 ((1 << nbits) - 1) for x in range(1 << nbits)]
      for x in range(1 << nbits):
        if outputs[x] != outputs[x ^ c_val]:
          raise AssertionError(f'c={c}: f({x}) != f({x ^ c_val})')
      want = (1 << nbits) // 2 if c_val else (1 << nbits)
      if len(set(outputs)) != want:
        raise AssertionError(f'c={c}: {len(set(outputs))} outputs, want {want}')

      # And the measurements must satisfy <z, c> = 0.
      psi = ops.Hadamard(nbits)(state.zeros(nbits * 2))
      psi = ops.Hadamard(nbits)(u(psi))
      if violation_probability(psi, nbits, c) > 1e-4:
        raise AssertionError(f'c={c}: measurements violate <z, c> = 0')
  print('Oracle tests passed.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  test_oracle()

  # Note: Running with 5 (10 total) qubits takes about 5 minutes.
  run_experiment(3)


if __name__ == '__main__':
  app.run(main)
