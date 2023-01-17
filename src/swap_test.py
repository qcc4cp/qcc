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
# The measurement probability of the ancillary qubit to be in state |0> is
#    1/2 + 1/2 * |<psi|phi>|^2
#
# Because of the dot product:
#   The probability found for maximally equal     is 1.0
#   The probability found for maximally different is 0.5
#
# A good overview can be found here:
#   https://en.wikipedia.org/wiki/Swap_test

from absl import app
import numpy as np

from src.lib import ops
from src.lib import state


def run_experiment_single(a1: np.complexfloating, a2: np.complexfloating,
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
        f'Probability {p0:.2f} off more than 5% from target {target:.2f}')
  print(f'Similarity of a1: {a1:.2f}, a2: {a2:.2f} => %: {100 * p0:.2f}')


def run_experiment_double(a0: np.complexfloating, a1: np.complexfloating,
                          b0: np.complexfloating, b1: np.complexfloating,
                          target: float) -> None:
  """Construct multi-qubit swap test circuit and measure."""

  # The circuit is quite simple:
  #
  # |0> --- H --- o --- o --- H --- Measure
  #               |     |
  # a0  --------- x --- | ----
  #               |     |
  # a1  ----------| --- x ----
  #               |     |
  # b0  --------- x --- | ----
  #                     |
  # b1  ----------------x ----

  psi_a = state.qubit(a0) * state.qubit(a1)
  psi_a = ops.Cnot(0, 1)(psi_a)
  psi_b = state.qubit(b0) * state.qubit(b1)
  psi_b = ops.Cnot(0, 1)(psi_b)

  psi = state.bitstring(0) * psi_a * psi_b

  psi = ops.Hadamard()(psi, 0)
  psi = ops.ControlledU(0, 1, ops.Swap(1, 3))(psi)
  psi = ops.ControlledU(0, 2, ops.Swap(2, 4))(psi)
  psi = ops.Hadamard()(psi, 0)

  # Measure once.
  p0, _ = ops.Measure(psi, 0)
  print(f'Sim of (a0: {a0:.2f}, a1: {a1:.2f}) ' +
        f'(b0: {b0:.2f}, b1: {b1:.2f}) => %: {100 * p0:.2f}')
  if abs(p0 - target) > 0.05:
    raise AssertionError(
        'Probability {:.2f} off more than 5% from target {:.2f}'
        .format(p0, target))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Swap test. 0.5 means different, 1.0 means similar')
  run_experiment_single(1.0, 0.0, 0.5)
  run_experiment_single(0.0, 1.0, 0.5)
  run_experiment_single(1.0, 1.0, 1.0)
  run_experiment_single(0.0, 0.0, 1.0)
  run_experiment_single(0.1, 0.9, 0.65)
  run_experiment_single(0.2, 0.8, 0.8)
  run_experiment_single(0.3, 0.7, 0.9)
  run_experiment_single(0.4, 0.6, 0.95)
  run_experiment_single(0.5, 0.5, 0.97)
  run_experiment_single(0.1, 0.1, 1.0)
  run_experiment_single(0.8, 0.8, 1.0)

  # 2 qubits:
  probs = [0.5, 0.5, 0.5, 0.52, 0.55, 0.59, 0.65, 0.72, 0.80, 0.90]
  for i in range(10):
    run_experiment_double(1.0, 0.0, 0.0 + i * 0.1, 1.0 - i * 0.1, probs[i])

  # An experiment with superposition, as mentioned in the literature:
  psi = state.bitstring(0, 0, 0, 0, 0)

  psi = ops.Hadamard()(psi, 0)
  psi = ops.Hadamard()(psi, 1)
  psi = ops.Hadamard()(psi, 2)
  psi = ops.ControlledU(0, 1, ops.Swap(1, 3))(psi)
  psi = ops.ControlledU(0, 2, ops.Swap(2, 4))(psi)
  psi = ops.Hadamard()(psi, 0)
  p0, _ = ops.Measure(psi, 0)

  # P(|0>) = 1/2 + 1/2<a|b>^2.
  # Hence (dot product)^2 = 2 * (p0 - 0.5)
  #
  if abs(p0 - 0.624999) > 0.5:
    raise AssertionError('Incorrect math on example.')
  print(f'Similarity from literature: p={p0:.3f}, dot={2*(p0-0.5):.3f} (ok)')


if __name__ == '__main__':
  app.run(main)
