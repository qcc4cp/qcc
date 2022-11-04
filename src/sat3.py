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


# ! ! ! This is Work In Progress (WIP) and not complete yet. ! ! !
#
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

  # Here, clauses are just lists of int's, position in the list
  # indicates the corresponding variable to use.
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


def make_or(qc, op0, op1, ancillary):
  """Make a single OR gate (ancillary initialized as |0>)."""

  # print(f'OR  from {op0} or {op1} -> {ancillary}')
  qc.x(op0)
  qc.x(op1)
  qc.x(ancillary)
  qc.toffoli(op0, op1, ancillary)
  qc.x(op1)
  qc.x(op0)


def make_and(qc, op0, op1, ancillary):
  """Make a single AND gate (ancillary initialized as |0>)."""

  # print(f'AND from {op0} to {op1} -> {ancillary}')
  qc.toffoli(op0, op1, ancillary)


def make_cnf_circuit(variables: int, clauses: int, formula):
  """Make the 'inside' of the CFG formula."""

  qubits = num_qubits(variables, clauses)
  print(f'Making {qubits:2d}-qubits circuit for: {print_formula(formula)}')

  qc = circuit.qc('3SAT Inner', eager=False)
  qc.reg(qubits)

  # First create all the OR's for each clause.
  anc = variables
  for cidx in range(len(formula)):
    clause = formula[cidx]

    # Negate gates, if necessary.
    for i in range(len(clause)):
      if clause[i] == FALSE:
        qc.x(i)

    # TODO(rhundt): Generalize for more than 3 variables.
    make_or(qc, 0, 1, anc)
    make_or(qc, 2, anc, anc + 1)
    anc += 2

    # Undo the negation, if necessary.
    for i in range(len(clause)):
      if clause[i] == FALSE:
        qc.x(i)

  # Now add the connecting AND gates.
  if len(formula) > 1:
    base = variables + 1
    make_and(qc, base, base+2, anc)
    base += 4

    for cidx in range(1, len(formula) - 1):
      make_and(qc, base, anc, anc+1)
      base += 2
      anc += 1
  else:
    # If there is no AND gate, we have to correct the last ancilla by 1.
    anc -= 1

  return qc


def num_qubits(variables: int, clauses: int) -> int:
  """Compute the number of gates needed for the inner circuit."""

  # An OR gate is a toffoli gate with all inputs [X]'ed.
  # Each OR needs 1 ancillary.
  # There are 3 variables - 2 OR's per clause.
  # Clauses are connected with AND, which also needs 1 ancillary.
  qubits = variables + clauses * (variables - 1) + (clauses - 1)
  return qubits


def make_full_circuit(qubits: int, bits, qc_inner):
  """Make the full circuit, connect input bits."""

  # Quantum Evaluation.
  qc = circuit.qc('Outer', eager=False)
  qc.reg(qubits, 0)
  for b in range(len(bits)):
    if bits[b] == 1:
      qc.x(b)
  qc.qc(qc_inner)

  # Let's add a final X-gate to reverse the output.
  # (There are more true's than false's with low numbers
  # of clauses)
  qc.x(qc.nbits - 1)
  return qc


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
  # answer_true = np.random.randint(0, num_inputs)

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
  print('Got f({}) = {}, want: 1, #: {:2d}, p: {:6.4f}'
        .format(maxbits[:-1], result, solutions, maxprob))
  if result != 1:
    raise AssertionError('Wrong result in Grover.')


def grover_with_circuit(variables: int, clauses: int):
  """Oracle-based Grover."""

  def multi(qc: circuit.qc, gate: ops.Operator, idx: list):
    for i in idx:
      qc.apply1(gate, i, 'multi')

  def multi_masked(qc: circuit.qc, gate: ops.Operator, idx: list,
                   mask, allow: int):
    for i in idx:
      if mask[i] == allow:
        qc.apply1(gate, i, 'multi-mask')

  formula = make_formula(variables, clauses)
  solutions = find_solutions(variables, formula)
  print('Solutions', solutions)

  nbits = num_qubits(variables, clauses)
  qc_inner = make_cnf_circuit(variables, clauses, formula)

  qc = circuit.qc('Outer', eager=False)
  qc.reg(nbits, 0)
  qc.reg(1, 1)
  aux = qc.reg(nbits - 1, 0)
  iterations = int(math.pi / 4 * math.sqrt(2**nbits))

  multi(qc, ops.Hadamard(), [i for i in range(nbits + 1)])
  print('nbits', nbits, 'iterations', iterations)

  for i in range(iterations):
    qc.qc(qc_inner)
    qc.x(qc.nbits - 1)

    idx = [i for i in range(nbits)]
    idx = [nbits - 1]

    # Phase Inversion
    qc.multi_control([nbits - 1], nbits,
                     aux, ops.PauliX(), 'Phase Inversion')

    # Mean Inversion
    multi(qc, ops.Hadamard(), idx)
    multi(qc, ops.PauliX(), idx)
    qc.multi_control(idx, nbits, aux, ops.PauliZ(), 'Mean Inversion')
    multi(qc, ops.PauliX(), idx)
    multi(qc, ops.Hadamard(), idx)

  qc.run()
  # qc.psi.dump()
  maxbits, maxprob = qc.psi.maxprob()
  print('->', formula, maxbits[:3], maxprob)


def run_tests(variables: int, clauses: int) -> None:
  """Run a few tests to ensure correct circuit construction."""

  qubits = num_qubits(variables, clauses)
  formula = make_formula(variables, clauses)
  qc_inner = make_cnf_circuit(variables, clauses, formula)

  for bits in itertools.product([0, 1], repeat=variables):
    # Classic Evaluation.
    res = eval_formula(bits, formula)

    # Quantum Evaluation.
    qc = make_full_circuit(qubits, bits, qc_inner)
    qc.run()

    # Note that 0/1 and 1/0 have the opposite meaning in
    # Classic vs Quantum. This measurement would assert
    # if an unexpected results appeared. We added an
    # X-gate above to map classical 0 to quantum |0>.
    #
    tostate = 0 if res == TRUE else 1
    prob = qc.measure_bit_iterative(qc_inner.nbits-1, tostate)
    if prob < 0.99:
      raise AssertionError('Incorrect result.')

    # print(f'Bits: {bits} -> C: {res} ', end='')
    # print(f'Q: {"False" if tostate == 0 else "True"}, prop: {prob:.1f}')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Quantum 3-SAT Solver. WIP, Not complete yet!')

  grover_with_circuit(3, 1)
  return

  # Oracle-based.
  for _ in range(2):
    grover_with_oracle(3, 1, 1)
    grover_with_oracle(3, 2, 1)
    grover_with_oracle(3, 3, 1)

  # Construct and check quantum circuits.
  for _ in range(2):
    run_tests(3, 1)
    run_tests(3, 2)
    run_tests(3, 3)
    run_tests(3, 4)

  # Now with full circuit construction.
  # TODO(rhundt)


if __name__ == '__main__':
  app.run(main)
