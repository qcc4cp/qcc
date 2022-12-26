# python3
"""Example: Arithmetic with Quantum Circuits used classically."""

import math

from absl import app

from src.lib import circuit
from src.lib import ops
from src.lib import state


# We want to control this (non-quantum-effect-exploiting) circuit:
#
# a    ----o-----o---o-------
# b    ----|--o--o---|--o----
# cin  ----|--|--|---o--o--o-
# sum  ----X--X--|---|--|--X-
# cout ----------X---X--X----
#
# For educational purposes, we construct this circuit using the
# basic Operators in matrix form as well as the more compact
# circuit form.


def fulladder_qc(qc: circuit.qc):
  """Non-quantum-exploiting, classic full adder."""

  qc.cx(0, 3)
  qc.cx(1, 3)
  qc.ccx(0, 1, 4)
  qc.ccx(0, 2, 4)
  qc.ccx(1, 2, 4)
  qc.cx(2, 3)


def fulladder_matrix(psi: state.State):
  """Non-quantum-exploiting, classic full adder."""

  psi = ops.Cnot(0, 3)(psi, 0)
  psi = ops.Cnot(1, 3)(psi, 1)
  psi = ops.ControlledU(0, 1, ops.Cnot(1, 4))(psi, 0)
  psi = ops.ControlledU(0, 2, ops.Cnot(2, 4))(psi, 0)
  psi = ops.ControlledU(1, 2, ops.Cnot(2, 4))(psi, 1)
  psi = ops.Cnot(2, 3)(psi, 2)
  return psi


def experiment_qc(a: int, b: int, cin: int,
                  expected_sum: int, expected_cout: int):
  """Run a simple classic experiment, check results."""

  qc = circuit.qc('classic')

  qc.bitstring(a, b, cin, 0, 0)
  fulladder_qc(qc)

  bsum, _ = qc.measure_bit(3, tostate=1, collapse=False)
  bout, _ = qc.measure_bit(4, tostate=1, collapse=False)
  print(f'a: {a} b: {b} cin: {cin} sum: {bsum:.0f} cout: {bout:.0f}')
  if (not math.isclose(bsum, expected_sum, abs_tol=1e-5) or
      not math.isclose(bout, expected_cout, abs_tol=1e-5)):
    raise AssertionError('invalid results')


def experiment_matrix(a: int, b: int, cin: int,
                      expected_sum: int, expected_cout: int):
  """Run a simple classic experiment, check results."""

  psi = state.bitstring(a, b, cin, 0, 0)
  psi = fulladder_matrix(psi)

  bsum, _ = ops.Measure(psi, 3, tostate=1, collapse=False)
  bout, _ = ops.Measure(psi, 4, tostate=1, collapse=False)
  print(f'a: {a} b: {b} cin: {cin} sum: {bsum:.0f} cout: {bout:.0f}')
  if (not math.isclose(bsum, expected_sum) or
      not math.isclose(bout, expected_cout)):
    raise AssertionError('invalid results')


def add_classic():
  """Full eval of the full adder."""

  for exp_function in [experiment_matrix, experiment_qc]:
    exp_function(0, 0, 0, 0, 0)
    exp_function(0, 1, 0, 1, 0)
    exp_function(1, 0, 0, 1, 0)
    exp_function(1, 1, 0, 0, 1)
    exp_function(0, 0, 1, 1, 0)
    exp_function(0, 1, 1, 0, 1)
    exp_function(1, 0, 1, 0, 1)
    exp_function(1, 1, 1, 1, 1)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  add_classic()


if __name__ == '__main__':
  app.run(main)
