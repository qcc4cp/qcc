# python3
"""Example: Arithmetic with Quantum Circuits doing arithmetic via QFT."""

# ACHTUNG: This version did not work with a given Python environment.
# Updating the environment fixed it, but the real reasons are unknown.


import math
from typing import List

from absl import app

from src.lib import circuit
from src.lib import helper
from src.lib import state


# Note: The QFT was constructed when the bit order in the state
#    initializers was wrongly inverted. After I corrected the bit
#    order, this code here could change. However, the index
#    expressions appear to be simpler in this order, so I decided
#    to keep it in this order and instead just invert the in/out bits.

# Quantum addition / subtraction / multiplication based on QFT.
#
# The following code is interesting in that the QFT here uses
# a different initial power of two (ops.qft starts with 2 pi,
# this one just with pi).


def check_result(psi: state.State, a, b,
                 nbits: int, factor: float = 1.0) -> None:
  """Find most likely result, dump it, compare against expected."""

  maxbits, _ = psi.maxprob()
  result = helper.bits2val(maxbits[0:nbits][::-1])
  if result != a + factor * b:
    print(f'{a} + ({factor} * {b}) != {result}')
    raise AssertionError('Incorrect addition.')


def qft(qc: circuit.qc, reg: state.Reg, n: int) -> None:
  qc.h(reg[n])
  for i in range(n):
    qc.cu1(reg[n - (i + 1)], reg[n], math.pi / float(2 ** (i + 1)))


def evolve(qc: circuit.qc, reg_a: state.Reg, reg_b: state.Reg,
           n: int, factor: float) -> None:
  for i in range(n + 1):
    qc.cu1(reg_b[n - i], reg_a[n], factor * math.pi / float(2**i))


def inverse_qft(qc: circuit.qc, reg: state.Reg, n: int) -> None:
  for i in range(n):
    qc.cu1(reg[i], reg[n], -1 * math.pi / float(2 ** (n - i)))
  qc.h(reg[n])


def arith_quantum(n: int, init_a: int, init_b: int,
                  factor: float = 1.0, dumpit: bool = False) -> None:
  """Run a quantum add experiment."""

  qc = circuit.qc('qadd')
  a = qc.reg(n + 1, helper.val2bits(init_a, n)[::-1], name='a')
  b = qc.reg(n + 1, helper.val2bits(init_b, n)[::-1], name='b')
  for i in range(n + 1):
    qft(qc, a, n - i)
  for i in range(n + 1):
    evolve(qc, a, b, n - i, factor)
  for i in range(n + 1):
    inverse_qft(qc, a, i)
  if dumpit:
    qc.dump_to_file()
  check_result(qc.psi, init_a, init_b, n + 1, factor)


# If we know which specific constant 'a' to add to a quantum register,
# we can just apply the rotations, no need for the b register in the
# general case. We just have to precompute the angles, as done here.
def precompute_angles(a: int, n: int) -> List[float]:
  """Pre-compute angles for the Fourier Transform, for fixed a."""

  # Convert 'a' to a string of 0's and 1's.
  s = bin(int(a))[2:].zfill(n)

  angles = [0.0] * n
  for i in range(n):
    for j in range(i, n):
      if s[j] == '1':
        angles[n - i - 1] += 2 ** (-(j - i))
    angles[n - i - 1] *= math.pi
  return angles


def arith_quantum_constant(n: int, init_a: int, c: int) -> None:
  """Run a quantum add-constant experiment."""

  qc = circuit.qc('qadd')
  a = qc.reg(n + 1, helper.val2bits(init_a, n)[::-1], name='a')
  for i in range(n + 1):
    qft(qc, a, n - i)

  angles = precompute_angles(c, n)
  for i in range(n):
    qc.u1(a[i], angles[i])

  for i in range(n + 1):
    inverse_qft(qc, a, i)

  maxbits, _ = qc.psi.maxprob()
  result = helper.bits2val(maxbits[0:n][::-1])
  if result != init_a + c:
    print(f'{init_a} + {c} = {result}')
    raise AssertionError('incorrect addition')


def arith_quantum_mult(nbits_a: int, init_a: int,
                       nbits_b: int, init_b: int) -> None:
  """Run a true quantum multiplication, c = a * b."""

  def add_src_to_targ(qc, nbits: int, src, targ, factor: float = 1.0):
    """Add the value of (src * factor) to targ, as shown above."""

    for i in range(nbits):
      qft(qc, targ, nbits - i)
    for i in range(nbits):
      evolve(qc, targ, src, nbits - i, factor)
    for i in range(nbits):
      inverse_qft(qc, targ, i)

  # Compute  c = a * b
  #
  # We have learned about how to add a term (factor * b) to c. We
  # use the same technique but control the execution of factors being
  # powers of two by the individual qubits of 'a'. As we go through
  # the qubits of 'a', we multiply 'factor' by 2 to model the binary
  # values of 'a' individual qubit positions.
  #
  qc = circuit.qc('qmult', eager=False)
  a = qc.reg(nbits_a, helper.val2bits(init_a, nbits_a)[::-1], name='a')
  b = qc.reg(nbits_b * 2 + 1, helper.val2bits(init_b, nbits_b)[::-1], name='b')
  c = qc.reg(nbits_b * 2 + 1, 0, name='c')

  factor = 1.0
  for idx in range(nbits_a):
    sc = qc.sub()
    add_src_to_targ(sc, nbits_b * 2, b, c, factor)
    sc.control_by(a[idx])
    qc.qc(sc)
    factor *= 2
  qc.run()

  maxbits, _ = qc.psi.maxprob()
  result = helper.bits2val(maxbits[c[0] : c[0 + nbits_b * 2]][::-1])
  if result != init_a * init_b:
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
    for j in range(i):  # Note: Results can be 2nd complements.
      arith_quantum(6, i, j, -1.0)

  print('Check quantum (pseudo) multiplication...')
  for i in range(7):
    for j in range(7):
      arith_quantum(6, 0, i, j)

  print('Check quantum true multiplication...')
  for b in range(4):
    for a in range(3):
      arith_quantum_mult(2, a, 3, b)


if __name__ == '__main__':
  app.run(main)
