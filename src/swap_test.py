# python3
"""Example: Swap Test."""


# The Swap test is a circuit to measure how close to each other 2 states are.
# It doesn't tell what those states are, only if they are close. For example,
# the states |0> and |1> are maximally different. The pairs |0>, |0> and
# |1>, |1> are maximally similar. This applies also to other states than the
# basis states. For example, the two qubits (states):
#
#    psi = 0.2|0> + x|1>  # 0.2^2 + x^2 = 1.0
#    phi = 0.2|0> + x|1>
#
# are also maximally equal. These states are somewhere inbetween (80%):
#
#    psi = 0.2|0> + x|1>  # 0.2^2 + x^2 = 1.0
#    phi = 0.8|0> + y|1>  # 0.8^2 + y^2 = 1.0
#
# The measurement probability of qubit to be in state |0> is
#    1/2 + 1/2 * |<psi|phi>|^2
#
# Because of the dot product:
#   The probability found for maximally equal     is 1.0
#   The probability found for maximally different is 0.5
#
# A good overview can be found here:
#   https://en.wikipedia.org/wiki/Swap_test

import numpy as np

from absl import app

from src.lib import state
from src.lib import ops


def run_experiment(a1:np.complexfloating, a2:np.complexfloating,
                   target: float) -> None:
  """Construct swap test circuit and measure."""

  # The circuit is quite simple:
  #
  # |0> --- H --- o --- H --- Measure
  #               |
  # a1  --------- x ---------
  #               |
  # a2  ----------x ---------

  psi = state.bitstring(0) * state.qubit(a1) * state.qubit(a2)
  psi = ops.Hadamard()(psi, 0)
  psi = ops.ControlledU(0, 1, ops.Swap(1, 2))(psi)
  psi = ops.Hadamard()(psi, 0)

  # Measure once.
  p0, _ = ops.Measure(psi, 0)
  if abs(p0 - target) > 0.05:
    raise AssertionError(
        'Probability {:.2f} off more than 5% from target {:.2f}'
        .format(p0, target))
  print('Similarity of a1: {:.2f}, a2: {:.2f} ==>  %: {:.2f}'
        .format(a1, a2, 100.0 * p0))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Swap test. 0.5 means different, 1.0 means similar')
  run_experiment(1.0, 0.0, 0.5)
  run_experiment(0.0, 1.0, 0.5)
  run_experiment(1.0, 1.0, 1.0)
  run_experiment(0.0, 0.0, 1.0)
  run_experiment(0.1, 0.9, 0.65)
  run_experiment(0.2, 0.8, 0.8)
  run_experiment(0.3, 0.7, 0.9)
  run_experiment(0.4, 0.6, 0.95)
  run_experiment(0.5, 0.5, 0.97)
  run_experiment(0.1, 0.1, 1.0)
  run_experiment(0.8, 0.8, 1.0)


if __name__ == '__main__':
  app.run(main)
