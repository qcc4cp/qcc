# python3
"""Example: 3-Sat solver via Grover's Algorithm."""

import itertools
import math
import random

from absl import app
import numpy as np

from src.lib import circuit
from src.lib import helper
from src.lib import state
from src.lib import ops


# 3-Sat Satisfyability Decision Problem:
#
# Given a Boolean formula in Conjunctive Normal Form (CNF), does
# this formula have a satisfying True assignment.
#
# What is CNF?
#   A 'literal' is a Boolean variable or its negation,
#       written as x or -x.
#   A 'clause' is a disjunction (logical OR) of literals.
#       for example (x0 or -x1 or -x2)
#   A 'formula' is a conjunction (logical AND) of clauses.
#       for example (x0 or -x1 or -x2) and (-x0 or -x2)
#
# A 3SAT problem is a CNF where each clause has 3 literals, each
# involving a different variable.
#       for example (x0 or -x1 or -x2) and (-x0 or -x1 or x2)
#
# The task is to find an assignment of True/False to the variables
# x0, x1, ..., xn-1 such that the formula returns True.
#
# Classically, this problem is known to be NP-complete. As a matter
# of fact, this problem helped frame the whole class of NP-complete
# problems.
#
# The idea here is - can we use Grover's algorithm to find a positive
# solution in sqrt(N) time.

# For the solution using circuits, we ONLY evaluate a single clause.
# This can be extended quite easily, of course.

# In C/C++, a return value of 0 is considered False. However,
# in Quantum computing, states are initialized as |0> and
# applying an X-gate negates the state to |1>. So it is
# the reverse, with |0> being True. Here, we use the C/C++
# convention.
FALSE = 0
TRUE = 1


def print_formula(clauses):
  """Convert formula, a list of clauses, to a string."""

  expstr = []
  for expr in clauses:
    # convert each clause to string.
    substr = []
    for j in range(len(expr)):
      # An int of 0 means negation.
      res = '-' if expr[j] == FALSE else ' '
      substr.append(res + 'x' + str(j))
    expstr.append('(' + ' '.join(substr) + ')')

  # produce final formula.
  res = '&'.join(expstr)
  return res


def match_bit(b, val):
  """Match bit (0==False, 1==True) with possible negation (val==0)."""

  if b == FALSE:
    return True if val == FALSE else False
  if b == TRUE:
    return False if val == FALSE else True


def eval_formula(bits, clauses: int):
  """Evaluate a formula."""

  res = True
  for clause in clauses:
    # Compute result of a clause (logical or)
    value = False
    for b in range(len(bits)):
      value = value or match_bit(bits[b], clause[b])
    # Compute conjuction (logical and)
    res = res and value
  return res


def make_clause(variables: int):
  """Make an individual clause from 'variables' variables."""

  # Here, clauses are just lists of int's. Position in the list
  # indicates the corresponding variable to use (x0, x1, ...).
  #
  # We represent negation as int 0, unmodified as int 1.
  #   Ex: A list of [1 0 1] corresponds to (x0 or -x1 or x2)
  #
  clause = []
  for _ in range(variables):
    clause.append(random.randint(0, 1))
  return clause


def make_formula(variables: int, clauses: int):
  """Make a formula."""

  formula = []
  for _ in range(clauses):
    formula.append(make_clause(variables))
  return formula


def make_f(variables: int, formula):
  """Construct function that evaluates formula."""

  # This is the simplest approach where we construct
  # an operator. However, this construction requires
  # that we evaluate the formula for each input, so
  # we don't gain anything. Nevertheless, let's use this
  # to first prove out that Grover would work on this
  # kind of input function (of course it does!).
  #
  num_inputs = 1 << variables
  answers = np.zeros(num_inputs, dtype=np.int8)

  for bits in itertools.product([0, 1], repeat=variables):
    res = eval_formula(bits, formula)
    # Note: We negate the result, as for small numbers
    # of clauses there are more positives than negatives.
    #
    answers[helper.bits2val(bits)] = not res

  # pylint: disable=no-value-for-parameter
  def func(*bits) -> int:
    return answers[helper.bits2val(*bits)]

  return func


def find_solutions(variables: int, formula):
  """Find number of (negative) solutions."""

  solutions = []
  for bits in itertools.product([0, 1], repeat=variables):
    res = eval_formula(bits, formula)

    # Note again the negation here.
    if not res:
      solutions.append(bits)
  return solutions


def grover_with_oracle(variables: int, clauses: int, solutions: int):
  """Oracle-based Grover."""

  formula = make_formula(variables, clauses)

  nbits = variables
  f = make_f(variables, formula)
  uf = ops.OracleUf(nbits+1, f)

  zero_projector = np.zeros((2**nbits, 2**nbits))
  zero_projector[0, 0] = 1
  op_zero = ops.Operator(zero_projector)

  psi = state.zeros(nbits) * state.ones(1)
  for i in range(nbits + 1):
    psi.apply1(ops.Hadamard(), i)

  hn = ops.Hadamard(nbits)
  reflection = op_zero * 2.0 - ops.Identity(nbits)
  inversion = hn(reflection(hn)) * ops.Identity()
  grover = inversion(uf)

  iterations = int(math.pi / 4 * math.sqrt(2**nbits / solutions))
  for _ in range(iterations):
    psi = grover(psi)

  maxbits, maxprob = psi.maxprob()
  result = f(maxbits[:-1])
  print('Oracle: Got f({}) = {}, want: 1, #: {:2d}, p: {:6.4f}'
        .format(maxbits[:-1], result, solutions, maxprob))
  if result != 1:
    raise AssertionError('Wrong result in Grover.')


def diffuser(qc: circuit.qc, reg, checker, aux):
  """Simple diffuser gate. Input qubits are in a register."""

  qc.h(reg)
  qc.x(reg)
  qc.multi_control(reg, checker,
                   aux, ops.PauliX(), 'Diffuser Gate')
  qc.x(reg)
  qc.h(reg)


def test_2sat_1():
  """Test (x or y) and not y."""

  # Note: De Morgan allows us to rewrite this as:
  #       not (not x and not y) and not y.
  #
  qc = circuit.qc('2Sat Circuit')

  reg = qc.reg(2, 0)
  x = reg[0]
  y = reg[1]
  w1 = qc.reg(1, 0)[0]
  w2 = qc.reg(1, 0)[0]
  chk = qc.reg(1, 0)[0]
  aux = qc.reg(1, 0)

  # Equal superposition of inputs.
  qc.h(reg)

  # Make a subcircuit.
  cc = circuit.qc('Gates', eager=False)

  # not x and not y -> w1.
  cc.x(reg)
  cc.toffoli(x, y, w1)
  cc.x(reg)

  # not y -> w2.
  cc.x(y)
  cc.cx(y, w2)
  cc.x(y)

  # Execute this subcircuit.
  qc.qc(cc)

  # w1 and w2 -> checker.
  cc.x(w1)
  cc.toffoli(w1, w2, chk)
  cc.x(w1)

  # Uncompute the subcircuit.
  qc.qc(cc.inverse())

  # Diffuser
  diffuser(qc, reg, chk, aux)

  maxbits, maxprob = qc.psi.maxprob()
  print('Test: (x and y) and (not y) ', end='')
  print('Want:', [1, 0], 'Got:', maxbits[:2], 'p:', maxprob)
  if maxbits[:2] != [1, 0]:
    raise AssertionError('Incorrect Result.')


def grover_with_circuit_3_1():
  """Circuit-based Grover for single 3-variable clause."""

  # Step 1: Make a single clause of 3 literals
  #         that are all OR'ed together.
  variables = 3
  formula = make_formula(variables, 1)
  clause = formula[0]

  # For single OR-clauses, there is only 1 negative solution,
  # which is the one where every single literatal is False.
  # Let's verify (solutions will just be the formula inverted).
  solution = find_solutions(variables, formula)

  # Let's compute the number of iterations we will need.
  iterations = int(math.pi / 4 * math.sqrt(2**variables))

  # We have clauses of 3 OR'ed together literals:
  #    x0 or x1 or x2
  # where each literal can also be negated.
  #
  # Since Grover works best if only 1 solution is present,
  # we want to find the 1 assignment where the clause becomes
  # False. For a single clause, that the assignment where
  # each single literal is False.
  #
  # De-Morgan tells us:
  #    not (x or y) == (not x and not y)
  #
  #    not (x or y or z) =
  #    not (x or (y or z)) =
  #    not x and not (y or z) =
  #    not x and not y and not z
  #
  # In terms of circuit construction, we have to negate each
  # literal. If the literal is already negated in the clause
  # we don't have to do anything!
  #
  # We are very generous spending ancillae, this can be totally
  # optimized.
  #
  qc = circuit.qc('Outer')
  reg = qc.reg(variables, 0)
  x = reg[0]
  y = reg[1]
  z = reg[2]
  aux = qc.reg(variables + 2, 0)
  ancx = aux[0]
  ancy = aux[1]
  ancz = aux[2]
  w0 = aux[3]
  w1 = aux[4]
  aux2 = qc.reg(variables-1)
  chk = qc.reg(1, 0)[0]

  # Equal superposition.
  qc.h(reg)

  # Construct the circuit.
  for iter in range(iterations):
    cc = circuit.qc('Gates', eager=False)
    if clause[0] == 1:
      cc.x(x)
      cc.cx(x, ancx)
      cc.x(x)
    else:
      cc.cx(x, ancx)
    if clause[1] == 1:
      cc.x(y)
      cc.cx(y, ancy)
      cc.x(y)
    else:
      cc.cx(y, ancy)
    if clause[2] == 1:
      cc.x(z)
      cc.cx(z, ancz)
      cc.x(z)
    else:
      cc.cx(z, ancz)
    cc.toffoli(ancx, ancy, w0)
    cc.toffoli(ancz, w0, w1)

    # Add sub=circuit.
    qc.qc(cc)

    # Phase inversion.
    qc.cx(w1, chk)

    # Uncompute.
    qc.qc(cc.inverse())

    # Mean inversion.
    diffuser(qc, reg, w1, aux2)

  maxbits, maxprob = qc.psi.maxprob()
  print(f'Circuit: Want: {list(solution[0])}, ', end='')
  print(f'Got: {list(maxbits[:3])}, p: {maxprob:.2f}')
  if list(solution[0]) != maxbits[:3]:
    raise AssertionError('Incorrect Result')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Quantum 3-SAT Solver.')

  # Quick simple tests.
  test_2sat_1()

  # Oracle-based.
  for _ in range(2):
    grover_with_oracle(3, 1, 1)
    grover_with_oracle(3, 2, 1)
    grover_with_oracle(3, 3, 1)

  # Single 3-variable clause.
  for _ in range(10):
    grover_with_circuit_3_1()


if __name__ == '__main__':
  app.run(main)
