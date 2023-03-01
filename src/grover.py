# python3
"""Example: Grover Algorithm."""

import math
import random
from typing import List

from absl import app
import numpy as np

from src.lib import circuit
from src.lib import helper
from src.lib import ops
from src.lib import state


#
# This function can be used if there is only 1 solution,
# which is the simplest case. We keep it here, as it is
# referenced in the book. It is no longer used in this code.
#
def make_f1(d: int = 3):
  """Construct function that will return 1 for only one bit string."""

  answers = np.zeros(1 << d, dtype=np.int8)
  answer_true = np.random.randint(0, 1 << d)
  answers[answer_true] = 1
  return (lambda bits: answers[helper.bits2val(bits)],
          helper.val2bits(answer_true, d))


def make_f(d: int = 3, nsolutions: int = 1):
  """Construct function that will return 1 for 'solutions' bits."""

  answers = np.zeros(1 << d, dtype=np.int8)
  solutions = random.sample(range(1 << d), nsolutions)
  answers[solutions] = 1
  return lambda bits: answers[helper.bits2val(bits)]


def run_experiment(nbits: int, solutions: int) -> None:
  """Run oracle-based experiment."""

  # Note that op_zero multiplies the diagonal elements of the operator by -1,
  # except for element [0][0]. This can be interpreted as "rotating around
  # the |00..)>" state. More pragmatically, multiplying this op_zero with
  # a Hadamard from the left and right gives a matrix of this form:
  #
  # 2/N-1   2/N    2/N    ...   2/N
  # 2/N     2N-1   2/N    ...   2/N
  # 2/N     2/N    2/N-1  ...   2/N
  # 2/N     2/N    2/N    ...   2/N-1
  #
  # Multiplying this matrix with a state vector computes exactly:
  #     2u - c_x
  # for every vector element c_x, with u being the mean over the
  # state vector. This is the defintion of inversion about the mean.
  #
  op_zero = ops.ZeroProjector(nbits)
  reflection = op_zero * 2.0 - ops.Identity(nbits)

  # Make f and Uf. Note:
  # We reserve space for an ancilla 'y'. This allows reuse of
  # the Deutsch Uf builder.
  #
  # We use the Oracle construction for convenience. It is rather
  # slow (full matrix) for larger qubit counts. Once can construct
  # a 'regular' circuit for the grover search algorithms, but this
  # circuit is different for each bitstring. Circuit versions are
  # below.
  #
  f = make_f(nbits, solutions)
  uf = ops.OracleUf(nbits + 1, f)

  # Build state with 1 ancilla of |1>.
  #
  psi = state.zeros(nbits) * state.ones(1)
  for i in range(nbits + 1):
    psi.apply1(ops.Hadamard(), i)

  # Build Grover operator, note Id() for the ancilla.
  # The Grover operator is the combination of:
  #    - phase inversion via the u unitary
  #    - inversion about the mean (see matrix above)
  #
  hn = ops.Hadamard(nbits)
  inversion = hn(reflection(hn)) * ops.Identity()
  grover = inversion(uf)

  # Number of Grover iterations
  #
  # There are at least 2 ways to compute the required number of iterations.
  #
  # 1) Most text books, eg., page 157 of Kaye, Laflamme and Mosca, find
  #    that the highest probability of finding the right results occurs
  #    after pi/4 sqrt(n) rotations.
  #
  # 2) For grover specifically, with 1 solution the iteration count is indeed:
  #        int(math.pi / 4 * math.sqrt(n))
  #
  # 3) For grover with multiple solutions:
  #        int(math.pi / 4 * math.sqrt(n / solutions))
  #
  # 4) For amplitude amplification, it's the probability of good
  #    solutions, which is trivial with the Grover equal
  #    superposition here:
  #        int(math.sqrt(n / solutions))
  #
  iterations = int(math.pi / 4 * math.sqrt(2**nbits / solutions))
  for _ in range(iterations):
    psi = grover(psi)

  # Measurement - pick element with higher probability.
  #
  # Note: We constructed the Oracle with n+1 qubits, to allow
  # for the 'xor-ancillary'. To check the result, we need to
  # ignore this ancilla.
  #
  maxbits, maxprob = psi.maxprob()
  result = f(maxbits[:-1])
  print('Matrix : Got f({}) = {}, want: 1, #: {:2d}, p: {:6.4f}'
        .format(maxbits[:-1], result, solutions, maxprob))
  if result != 1:
    raise AssertionError('something went wrong, measured invalid state')


def run_experiment_circuit(nbits: int) -> None:
  """Run circuit-based experiment."""

  # pylint disable=g-bare-generic
  def multi(qc: circuit.qc, gate: ops.Operator, idx: List[int]):
    for i in idx:
      qc.apply1(gate, i, 'multi')

  # pylint disable=g-bare-generic
  def multi_masked(qc: circuit.qc, gate: ops.Operator, idx: List[int],
                   mask, allow: int):
    for i in idx:
      if mask[i] == allow:
        qc.apply1(gate, i, 'multi-mask')

  # This implementation uses the 'trivial' multi-controlled gates,
  # which introduce _a lot_ of ancilla gates. As a result, there
  # are practically no benefits over the matrix-based implementation.
  # Once an optimized multi-controlled implementation is available,
  # then the circuit-based performance for many qubits will do
  # much better!.
  #
  qc = circuit.qc('Grover', eager=False)
  reg = qc.reg(nbits, 0)
  qc.reg(1, 1)
  aux = qc.reg(nbits - 1, 0)
  f, bits = make_f1(nbits)

  multi(qc, ops.Hadamard(), [i for i in range(nbits + 1)])

  iterations = int(math.pi / 4 * math.sqrt(2**nbits))
  idx = [i for i in range(nbits)]

  for _ in range(iterations):
    # Phase Inversion
    multi_masked(qc, ops.PauliX(), idx, bits, 0)
    qc.multi_control(reg, nbits, aux, ops.PauliX(), 'Phase Inversion')
    multi_masked(qc, ops.PauliX(), idx, bits, 0)

    # Mean Inversion
    multi(qc, ops.Hadamard(), idx)
    multi(qc, ops.PauliX(), idx)
    qc.multi_control(reg, nbits, aux, ops.PauliZ(), 'Mean Inversion')
    multi(qc, ops.PauliX(), idx)
    multi(qc, ops.Hadamard(), idx)

  qc.run()
  maxbits, maxprob = qc.psi.maxprob()
  result = f(maxbits[:nbits])
  print('Circuit: Got f({}) = {}, want: 1, p: {:6.4f}'
        .format(maxbits[:nbits], result, maxprob))
  if result != 1:
    raise AssertionError('something went wrong, measured invalid state')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for nbits in range(3, 10):
    run_experiment(nbits, 1)
  for solutions in range(1, 9):
    run_experiment(7, solutions)
  for nbits in range(8, 10):
    run_experiment_circuit(nbits)


if __name__ == '__main__':
  app.run(main)
