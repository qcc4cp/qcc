# python3
"""Example: Arithmetic with Quantum Circuits doing arithmetic via QFT."""

# ACHTUNG: This version did not work with a given Python environment.
# Updating the environment fixed it, but the real reasons are unknown.


import math

from absl import app

from src.lib import circuit
from src.lib import helper

# Note: The QFT was constructed when the bit order in the state
#    initializers was wrongly inverted. After I corrected the bit order,
#    this code here could change. However, the index expressions appear
#    to be simpler in this order, so I decided to keep it in this (wrong)
#    order and instead just invert the in/out bits.

# Quantum addition / subtraction / multiplication based on QFT.
#
# A nice description can be found here:
#  https://medium.com/@sashwat.anagolum/qftaddition-ce0a0b2bc4f4
#
# The following code is interesting in that the QFT here uses
# a different initial power of two (ops.qft starts with 2 pi,
# this one just with pi).
#
# This qft would work in phase estimation, even though with slightly
# worse accuracy, while the opposite is not true.
#
# TODO(rhundt): Analyze why exactly.


def check_result(psi, a, b, nbits, factor=1.0):
  """Find most likely result, dump it, compare against expected."""

  maxbits, _ = psi.maxprob()
  result = helper.bits2val(maxbits[0:nbits][::-1])
  if result != a + factor * b:
    print(f'{a} + ({factor} * {b}) = {result}')
    raise AssertionError('incorrect addition')


def qft(qc, reg, n):
  """qft."""

  qc.h(reg[n])
  for i in range(0, n):
    qc.cu1(reg[n-(i+1)], reg[n], math.pi/float(2**(i+1)))


def evolve(qc, reg_a, reg_b, n, factor):
  """Rotations."""

  for i in range(0, n+1):
    qc.cu1(reg_b[n-i], reg_a[n], factor * math.pi/float(2**(i)))


def inverse_qft(qc, reg, n):
  """Inverse qft."""

  for i in range(0, n):
    qc.cu1(reg[i], reg[n], -1*math.pi/float(2**(n-i)))
  qc.h(reg[n])


def arith_quantum(n, init_a, init_b, factor=1.0, dumpit=False):
  """Run a quantum add experiment."""

  qc = circuit.qc('qadd')
  a = qc.reg(n+1, helper.val2bits(init_a, n)[::-1], name='a')
  b = qc.reg(n+1, helper.val2bits(init_b, n)[::-1], name='b')
  for i in  range(0, n+1):
    qft(qc, a, n-i)
  for i in range(0, n+1):
    evolve(qc, a, b, n-i, factor)
  for i in range(0, n+1):
    inverse_qft(qc, a, i)
  if dumpit:
    qc.dump_to_file()
  check_result(qc.psi, init_a, init_b, n+1, factor)


# If we know which specific constant 'a' to add to a quantum register,
# we can just apply the rotations, no need for the b register in the
# general case. We just have to precompute the angles, as done here.
def precompute_angles(a, n):
  """Pre-compute angles used in the Fourier Transform, for fixed a."""

  # Convert 'a' to a string of 0's and 1's.
  s = bin(int(a))[2:].zfill(n)

  angles = [0.] * n
  for i in range(0, n):
    for j in range(i, n):
      if s[j] == '1':
        angles[n-i-1] += 2**(-(j-i))
    angles[n-i-1] *= math.pi
  return angles


def arith_quantum_constant(n, init_a, c):
  """Run a quantum add-constant experiment."""

  qc = circuit.qc('qadd')
  a = qc.reg(n+1, helper.val2bits(init_a, n)[::-1], name='a')
  for i in  range(0, n+1):
    qft(qc, a, n-i)

  angles = precompute_angles(c, n)
  for i in range(0, n):
    qc.u1(a[i], angles[i])

  for i in range(0, n+1):
    inverse_qft(qc, a, i)

  maxbits, _ = qc.psi.maxprob()
  result = helper.bits2val(maxbits[0:n][::-1])
  if result != init_a + c:
    print(f'{init_a} + {c} = {result}')
    raise AssertionError('incorrect addition')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Check large addition (18 qubits)...')
  arith_quantum(8, 1, 2, 1.0, True)

  print('Check quantum addition...')
  for i in range(7):
    for j in range(7):
      arith_quantum(6, i, j)
      arith_quantum_constant(6, i, j)

  print('Check quantum subtraction...')
  for i in range(8):
    for j in range(0, i):  # Note: Results are 'correct' as 2nd complements.
      arith_quantum(6, i, j, -1.0)

  print('Check quantum (pseudo) multiplication...')
  for i in range(0, 6):
    for j in range(0, 6):
      arith_quantum(6, 0, i, j)

if __name__ == '__main__':
  app.run(main)
