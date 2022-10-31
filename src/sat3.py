# python3
"""Example: 3-Sat solver via Grover's Algorithm."""

import itertools
import random

from absl import app
from src.lib import circuit


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
  """Compute the number of gates needed for the circuit."""

  # An OR gate is a toffoli gate with all inputs [X]'ed.
  # Hence each OR needs 1 ancillary.
  # There are variables - 1 OR's per clause.
  # Similar for AND, which also needs 1 ancillary.
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
  return qc


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
    # if an unexpected results appeared.
    tostate = 1 if res == TRUE else 0
    prob = qc.measure_bit_iterative(qc_inner.nbits-1, tostate)
    if prob < 0.99:
      raise AssertionError('Incorrect result.')

    # print(f'Bits: {bits} -> C: {res} ', end='')
    # print(f'Q: {"False" if tostate == 0 else "True"}, prop: {prob:.1f}')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Quantum 3-SAT Solver. WIP, Not complete yet!')
  for _ in range(5):
    run_tests(3, 1)
    run_tests(3, 2)
    run_tests(3, 3)
    run_tests(3, 4)


if __name__ == '__main__':
  app.run(main)
