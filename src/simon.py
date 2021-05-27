# python3
"""Example: Simon's Algorithm."""

from absl import app

from src.lib import helper
from src.lib import ops
from src.lib import state


# This is a common Oracle description for a 2-qubit (4-qubit total)
# Uf operator that produces even f(x) == f(x ^ c)
#
# ----o--o--------
#     |  |
# ----|--|--o--o---
#     |  |  |  |
# ----X--|--X--|---
#        |     |
# -------X-----X---
#
# Truth Table will be (for secret string 11):
#
#   f(0, 0) = 00 = f(0^1, 0^1) = f(1, 1)
#   f(0, 1) = 11 = f(0^1, 1^1) = f(1, 0)
#   f(1, 0) = 11 = ...
#   f(1, 1) = 00
#
# Thus, f(x) = f(x + c) with c being the secret string 11
# (Note that module 2 xor, +, and -, are all the same).
#
# Reference:
# https://qiskit.org/textbook/ch-algorithms/simon.html#implementation


def make_u():
  """Make Simon's 2 (total 4) qubit Oracle."""

  # We have to properly 'pad' the various gates to 4 qubits.
  #
  ident = ops.Identity()
  cnot0 = ops.Cnot(0, 2) * ident
  cnot1 = ops.Cnot(0, 3)
  cnot2 = ident * ops.Cnot(0, 1) * ident
  cnot3 = ident * ops.Cnot(0, 2)

  return cnot3 @ cnot2 @ cnot1 @ cnot0


def dot2(bits):
  """Compute dot module 2."""

  return (bits[0] * bits[2] + bits[1] * bits[3]) % 2


def run_experiment():
  """Run single, defined experiment for secret 11."""

  psi = state.zeros(4)
  u = make_u()

  psi = ops.Hadamard(2)(psi)
  psi = u(psi)
  psi = ops.Hadamard(2)(psi)

  # Because of the xor patterns (Yanofski 6.64)
  # measurement will only find those qubit strings where
  # the scalar product of z (lower bits) and secret string:
  #    <z, c> = 0
  #
  # This should measure |00> and |11> with equal probability.
  # If true, than we can derive the secret string as being 11
  # because f(00) = f(11) and because f(00) = f(00 ^ c) -> c = 11
  #
  print('Measure likely states (want: pairs of 00 or 11):')
  for bits in helper.bitprod(4):
    if psi.prob(*bits) > 0.01:
      if (bits[0] == 0 and bits[1] == 1) or (bits[0] == 1 and bits[1] == 0):
        raise AssertionError('Invalid Results')
      print('|{}{} {}{}> = 0 : {:.2f}  dot % 2: {:.2f}'.
            format(bits[0], bits[1],
                   bits[2], bits[3], psi.prob(*bits),
                   dot2(bits)))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run_experiment()


if __name__ == '__main__':
  app.run(main)
