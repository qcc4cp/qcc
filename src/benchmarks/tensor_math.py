# python3
"""Example: Experiments with Tensor Math."""

import random
import timeit

from absl import app

from src.lib import circuit
from src.lib import helper
from src.lib import ops
from src.lib import state


def operator_order():
  """Evaluate order of operations and corresponding matmuls."""

  hi = ops.Hadamard() * ops.Identity()
  cx = ops.Cnot(0, 1)

  # Make sure that the order of evaluation is correct. For example,
  # this simple circuit:
  #
  # |0> --- H --- o ---
  #               |
  # |0> ----------X ---
  #
  #  p0   p1     p2
  #
  # Can be evaluated step wise, applying each gate to psi:
  psi_0 = state.zeros(2)
  psi_1 = hi(psi_0)
  psi_2 = cx(psi_1)

  # Or via a combined operator. Yet, the order ot the ops
  # has to be reversed from above picture:
  combined_op = (cx @ hi)
  combined_psi = state.zeros(2)
  combined_psi_2 = combined_op(combined_psi)
  if not psi_2.is_close(combined_psi_2):
    raise AssertionError('Invalid order of operators from matmul')

  # This can also be expressed via the function call construct:
  combined_f = hi(cx)
  combined_f_psi = state.zeros(2)
  combined_f_psi_2 = combined_f(combined_f_psi)
  if not psi_2.is_close(combined_f_psi_2):
    raise AssertionError('Invalid order of operators from call construct')


def operator_complexity():
  """Demonstrate the combined operators are much more expensive than single ops."""

  # A value of 9 or higher shows a 10x or more advantage.
  n = 8

  def with_matmul():
    psi = state.zeros(n)
    ident = ops.Identity(n)
    h = ops.Hadamard(n)

    psi = (ident @ h)(psi)
    return psi

  def by_op():
    psi = state.zeros(n)
    ident = ops.Identity(n)
    h = ops.Hadamard(n)

    psi = ident(psi)
    psi = h(psi)
    return psi

  psim = with_matmul()
  psio = by_op()
  if not psim.is_close(psio):
    raise AssertionError('Invalid operator invocation')

  print('Time with combined op: {:.2f} secs'
        .format(timeit.timeit(with_matmul, number=1)))
  print('Time with op by op: {:.2f} secs'
        .format(timeit.timeit(by_op, number=10)))


def operator_per_state():
  """First foray into more effective math for 1-qubit operators."""

  # Make product state
  q0 = state.qubit(alpha=0.5)
  q1 = state.qubit(alpha=0.9)
  psi = q0 * q1

  # Compute via combined operator.
  op = ops.PauliX() * ops.Identity(1)
  psi1 = op(psi)

  # Combine via 1-qubit on q0 * q1
  psi2 = ops.PauliX()(q0) * q1

  if not psi1.is_close(psi2):
    raise AssertionError('Wrong tensor math and application of 1-qubit gate.')

  # Same thing, but apply to qubit 1.
  op = ops.Identity() * ops.PauliX()
  psi1 = op(psi)
  psi2 = q0 * ops.PauliX()(q1)
  if not psi1.is_close(psi2):
    raise AssertionError('Wrong tensor math and application of 1-qubit gate.')


# Single gate application reduces complexty from O(n*n) to O(n).
# TODO(rhundt): Evaluate 2 strategies for controlled qubits:
#    1) check if c is |1> and only apply t in that case.
#    2) Figure out how to apply gates of sizes other than 2x2
#    3) Other smarts ?!


def apply_single_gate(gate, qubit, psi):
  """Apply a single-qubit gate via explicit indexing."""

  # To maintain qubit ordering in this infrastructure,
  # index needs to be reversed.
  #
  qubit = psi.nbits - qubit - 1
  two_q = 2**qubit
  for g in range(0, 2**psi.nbits, 2**(qubit+1)):
    for i in range(g, g + two_q):
      t1 = gate[0, 0] * psi[i] + gate[0, 1] * psi[i + two_q]
      t2 = gate[1, 0] * psi[i] + gate[1, 1] * psi[i + two_q]
      psi[i] = t1
      psi[i + two_q] = t2
  return psi


def apply_controlled_gate(gate, control, target, psi):
  """Apply a controlled 2-qubit gate via explicit indexing."""

  # To maintain qubit ordering in this infrastructure,
  # index needs to be reversed.
  #
  qubit = psi.nbits - target - 1
  two_q = 2**qubit
  control = psi.nbits - control - 1
  for g in range(0, 2**psi.nbits, 2**(qubit+1)):

    # Note that this could be further specialized.
    #
    #    if control < target and  control bit == 0
    #
    # then all control bits will be 0 and the i loop below can be skipped.

    for i in range(g, g + two_q):
      idx = g * 2**psi.nbits + i
      if idx & (1 << control):
        t1 = gate[0, 0] * psi[i] + gate[0, 1] * psi[i + two_q]
        t2 = gate[1, 0] * psi[i] + gate[1, 1] * psi[i + two_q]
        psi[i] = t1
        psi[i + two_q] = t2
  return psi


def hipster_single():
  """Single-qubit Hipster Technique."""

  # This is a nice trick, outlined in this paper on "Hipster":
  #    https://arxiv.org/pdf/1601.07195.pdf
  #
  # The observation is that to apply a single-qubit gate to a
  # gubit with index i, take the binary representation of inidices and
  # apply the transformation matrix to the elements according
  # to the power of 2 index. Generally:
  # "Performing a single-qubit gate on qubit k of n-qubit quantum
  # register applies G to pairs of amplitudes whose indices differ in
  # k-th bits of their binary index".
  #
  # For example, for a 2-qubit system, to apply a gate to qubit 0:
  #    apply G to
  #    q11, q12   psi[0], psi[1]
  #    q21, q22   psi[2], psi[3]
  #
  # To apply to qubit 1:
  #    q11, q12   psi[0], psi[2]
  #    q21, q22   psi[1], psi[3]
  #
  # 'Outer loop' jumps by 2**(nbits+1)
  # 'Inner loop' jumps by 2**k
  #
  # To maintain the qubit / index ordering of this infrastructure,
  # the qubit index in the paper is reversed to the qubit index here.
  # (Hence the (nbits - qubit - 1) above)
  #

  # Make sure that for sample gates and all states the transformations
  # are identical.
  #
  for gate in (ops.PauliX(), ops.PauliZ(), ops.Hadamard(), ops.RotationX(0.5)):
    nbits = 5
    for bits in helper.bitprod(nbits):
      psi = state.bitstring(*bits)
      qubit = random.randint(0, nbits-1)

      # Full matrix (O(n*n).
      op = ops.Identity(qubit) * gate * ops.Identity(nbits - qubit - 1)
      psi1 = op(psi)

      # Single Qubit (O(n))
      psi = apply_single_gate(gate, qubit, psi)

      if not psi.is_close(psi1):
        raise AssertionError('Invalid Single Gate Application.')


def single_gate_complexity():
  """Compare times for full matmul vs single-gate."""

  nbits = 12
  qubit = random.randint(0, nbits-1)
  gate = ops.PauliX()

  def with_matmul():
    psi = state.zeros(nbits)
    op = ops.Identity(qubit) * gate * ops.Identity(nbits - qubit - 1)
    psi = op(psi)

  def apply_single():
    psi = state.zeros(nbits)
    psi = apply_single_gate(gate, qubit, psi)

  print('Time with full matmul: {:.3f} secs'
        .format(timeit.timeit(with_matmul, number=1)))
  print('Time with single gate: {:.3f} secs'
        .format(timeit.timeit(apply_single, number=1)))


def hipster_multi():
  """Multi-qubit, optimized application."""

  nbits = 7
  for bits in helper.bitprod(nbits):
    psi = state.bitstring(*bits)
    for target in range(1, nbits):
      # Full matrix (O(n*n).
      op = (ops.Identity(target-1) * ops.Cnot(target-1, target) *
            ops.Identity(nbits - target - 1))
      psi1 = op(psi)

      # Single Qubit (O(n))
      psi = apply_controlled_gate(ops.PauliX(), target-1, target, psi)
      if not psi.is_close(psi1):
        raise AssertionError('Invalid Single Gate Application.')

  psi = state.bitstring(1, 1, 0, 0, 1)
  pn = ops.Cnot(1, 4)(psi, 1)
  if not pn.is_close(apply_controlled_gate(ops.PauliX(), 1, 4, psi)):
    raise AssertionError('Invalid Cnot')
  pn = ops.Cnot(4, 1)(psi, 1)
  if not pn.is_close(apply_controlled_gate(ops.PauliX(), 4, 1, psi)):
    raise AssertionError('Invalid Cnot')
  pn = ops.ControlledU(0, 1, ops.ControlledU(1, 4, ops.PauliX()))(psi)

  psi = state.qubit(alpha=0.6) * state.ones(2)
  pn = ops.Cnot(0, 2)(psi)
  if not pn.is_close(apply_controlled_gate(ops.PauliX(), 0, 2, psi)):
    raise AssertionError('Invalid Cnot')


def time_gate_application(nbits):
  """Benchmark single gate application."""

  def with_matrix():
    psi = state.zeros(nbits)
    psi = ops.Hadamard(nbits)(psi)

  def individual():
    psi = state.zeros(nbits)
    h = ops.Hadamard()
    for i in range(nbits):
      psi = apply_single_gate(h, i, psi)

  print('Time with full matrix: {:.3f} secs'
        .format(timeit.timeit(with_matrix, number=1)))
  print('Time via  single gates: {:.3f} secs'
        .format(timeit.timeit(individual, number=1)))


def time_series(limit):
  """Simple time series for gate application."""

  def bench():
    psi = state.zeros(nbits)
    for i in range(nbits):
      psi = apply_single_gate(h, i, psi)

  h = ops.Hadamard()
  for i in range(10, limit):
    nbits = i
    print('qubit: {}, muls: {}, mem: {}k, time: {:.3f} secs'
          .format(nbits, 2**nbits*4, 2**nbits*16 / 1024,
                  timeit.timeit(bench, number=1)))


def time_series_qc(limit):
  """Simple time series for gate application."""

  def bench():
    qc = circuit.qc()
    qc.rand(nbits)
    for i in range(nbits):
      qc.h(i)

  for i in range(10, limit):
    nbits = i
    print('qubit: {}, muls: {}, mem: {}k, time: {:.3f} secs'
          .format(nbits, 2**nbits*4, 2**nbits*16 / 1024,
                  timeit.timeit(bench, number=1)))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  operator_order()
  operator_complexity()
  operator_per_state()
  hipster_single()
  single_gate_complexity()
  hipster_multi()
  time_gate_application(12)
  time_series(17)
  time_series_qc(25)


if __name__ == '__main__':
  app.run(main)
