# python3
"""Example: Order Finding - the precursor to Shor."""

# This code could not have been completed and debugged without looking
# and comparing to this working implementation (based on Qiskit):
#
#    https://github.com/ttlion/ShorAlgQiskit/blob/master/Shor_Normal_QFT.py
#
# Funny enough - even Qiskit mentions this implementation as a reference.
#
import fractions
import math
from typing import List, Tuple

from absl import app
from absl import flags

from src.lib import circuit
from src.lib import helper


# Good values to try and factor are:
#   N=15, a=4    (18 qubits)
#   N=21, a=11   (22 qubits)
#   N=25, a=11   (22 qubits, only finds 1 factor)
#   N=33, a=13   (26 qubits, only finds 1 factor)
#   N=35, a=9    (26 qubit)
#
flags.DEFINE_integer('N', 15, 'Number to factor.')
flags.DEFINE_integer('a', 4, 'Start search with this number.')


def modular_inverse(a: int, m: int) -> int:
  """Compute Modular Inverse."""

  def egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidian Algorithm."""

    # Explained here:
    # https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    if a == 0:
      return (b, 0, 1)
    g, y, x = egcd(b % a, a)
    return (g, x - (b // a) * y, y)

  # Modular inverse of x mod m is the number x^-1 such that
  #   x * x^-1 = 1 mod m
  g, x, _ = egcd(a, m)
  assert g == 1, f'Modular inverse ({a}, {m}) does not exist.'
  return x % m


def precompute_angles(a: int, n: int) -> List[float]:
  """Pre-compute angles used in the Fourier Transform, for a."""

  angles = [0.0] * n
  for i in range(n):
    for j in range(i, n):
      if (a & (1 << n - j - 1)):
        angles[n - i - 1] += 2 ** (-(j - i))
    angles[n - i - 1] *= math.pi
  return angles


def add(qc, q, a: int, n: int, factor: float) -> None:
  """Un-controlled add in fourier space."""

  for idx, angle in enumerate(precompute_angles(a, n)):
    qc.u1(q[idx], factor * angle)


def cadd(qc, q, ctl, a: int, n: int, factor: float) -> None:
  """Controlled add in fourier space."""

  for idx, angle in enumerate(precompute_angles(a, n)):
    qc.cu1(ctl, q[idx], factor * angle)


def ccadd(qc, q, ctl1: int, ctl2: int, a: int, n: int,
          factor: float) -> None:
  """Controlled-controlled add in fourier space."""

  for idx, angle in enumerate(precompute_angles(a, n)):
    qc.ccu1(ctl1, ctl2, q[idx], factor * angle)


def qft(qc, up_reg, n: int, with_swaps: bool = False) -> None:
  qc.qft(up_reg[:n], with_swaps)


def inverse_qft(qc, up_reg, n: int, with_swaps: bool = False) -> None:
  qc.inverse_qft(up_reg[:n], with_swaps)


def cc_add_mod_n(qc, q, ctl1, ctl2, aux, a, number, n):
  """Circuit that implements double controlled modular addition by a."""

  ccadd(qc, q, ctl1, ctl2, a, n, factor=1.0)
  add(qc, q, number, n, factor=-1.0)
  inverse_qft(qc, q, n)
  qc.cx(q[n - 1], aux)
  qft(qc, q, n)
  cadd(qc, q, aux, number, n, factor=1.0)

  ccadd(qc, q, ctl1, ctl2, a, n, factor=-1.0)
  inverse_qft(qc, q, n)
  qc.cx0(q[n - 1], aux)
  qft(qc, q, n)
  ccadd(qc, q, ctl1, ctl2, a, n, factor=1.0)


def cc_add_mod_n_inverse(qc, q, ctl1, ctl2, aux, a, number, n):
  """Inverse of the double controlled modular addition."""

  ccadd(qc, q, ctl1, ctl2, a, n, factor=-1.0)
  inverse_qft(qc, q, n)
  qc.cx0(q[n - 1], aux)
  qft(qc, q, n)
  ccadd(qc, q, ctl1, ctl2, a, n, factor=1.0)

  cadd(qc, q, aux, number, n, factor=-1.0)
  inverse_qft(qc, q, n)
  qc.cx(q[n - 1], aux)
  qft(qc, q, n)
  add(qc, q, number, n, factor=1.0)
  ccadd(qc, q, ctl1, ctl2, a, n, factor=-1.0)


def cmultmodn(qc, ctl, q, aux, a, number, n):
  """Controlled Multiplies modulo N."""

  print('Compute... ')
  qft(qc, aux, n + 1)
  for i in range(n):
    cc_add_mod_n(qc, aux, q[i], ctl, aux[n + 1],
                 ((2**i) * a) % number, number, n + 1)
  inverse_qft(qc, aux, n + 1)

  print('Swap')
  for i in range(n):
    qc.cswap(ctl, q[i], aux[i])
  a_inv = modular_inverse(a, number)

  print('Uncompute...')
  qft(qc, aux, n + 1)
  for i in reversed(range(n)):
    cc_add_mod_n_inverse(qc, aux, q[i], ctl, aux[n+1],
                         ((2**i) * a_inv) % number, number, n + 1)
  inverse_qft(qc, aux, n + 1)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('Order finding.')

  number = flags.FLAGS.N
  a = flags.FLAGS.a

  # The classical part are handled in 'shor_classic.py'
  nbits = number.bit_length()
  print('Shor: N = {}, a = {}, n = {} -> qubits: {}, iterations: {}'
        .format(number, a, nbits, nbits * 4 + 2, nbits * 2))
  qc = circuit.qc('order_finding')

  # Aux register for addition and multiplication.
  aux = qc.reg(nbits + 2)

  # Register for QFT. This reg will hold the resulting x-value.
  up = qc.reg(nbits * 2)

  # Register for multiplications.
  down = qc.reg(nbits)

  # Check early to see whether modular inverse even exists.
  _ = modular_inverse(int(a), number)

  # The main phase estimation loop over the multiplication mod N.
  qc.h(up)
  qc.x(down[0])
  for i in range(nbits * 2):
    cmultmodn(qc, up[i], down, aux, int(a ** (2**i)), number, nbits)
  inverse_qft(qc, up, 2 * nbits, with_swaps=True)

  print('Measurement...')
  total_prob = 0.0
  for bits in helper.bitprod(nbits * 4 + 2):
    prob = qc.psi.prob(*bits)
    if prob > 0.01:
      bitslice = bits[nbits + 2 : nbits + 2 + nbits * 2][::-1]
      intval = helper.bits2val(bitslice)
      phase = helper.bits2frac(bitslice)

      r = fractions.Fraction(phase).limit_denominator(8).denominator
      guesses = [math.gcd(a ** (r // 2) - 1, number),
                 math.gcd(a ** (r // 2) + 1, number)]
      print('Final x: {:3d} phase: {:3f} prob: {:.3f} factors: {}'.
            format(intval, phase, prob.real, guesses))

    total_prob += prob
    if total_prob > 0.999:
      break

  print(qc.stats())


if __name__ == '__main__':
  app.run(main)
