# python3
"""Example: Euclidean Distance."""


import random
from absl import app
import numpy as np

from src.lib import ops
from src.lib import state


def run_experiment(a, b):
  """Compute Euclidean Distance between vectors a and b."""

  print(f'Quantum Euclidean Distance between a={a} b={b}')

  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  if norm_a == 0 or norm_b == 0:
    return
  normed_a = a / norm_a
  normed_b = b / norm_b
  z = (norm_a**2) + (norm_b**2)

  # Create state phi:
  #  |phi> = 1 / sqrt(z) (||a|| |0> - ||b|| |1>)
  #
  phi = state.State(1 / np.sqrt(z) * np.array([norm_a, -norm_b]))

  # Create state psi:
  #  |psi> = 1 / sqrt(2) |0>|a> + |1>|b>)
  #
  psi = (state.bitstring(0) * state.State(normed_a) +
         state.bitstring(1) * state.State(normed_b)) / np.sqrt(2)

  # Make a combined state with an ancilla (|0>), phi, and psi:
  #
  combo = state.bitstring(0) * phi * psi

  # Construct a swap test and find the measurement probability
  # of the ancilla.
  #
  combo = ops.Hadamard()(combo, 0)
  combo = ops.ControlledU(0, 1, ops.Swap(1, 2))(combo)
  combo = ops.Hadamard()(combo, 0)

  p0, _ = ops.Measure(combo, 0)

  # Now compute the euclidian norm from the probability.
  #
  eucl_dist_q = (4 * z * (p0 - 0.5)) ** 0.5

  # We can also compute the euclidian distance classically.
  #
  eucl_dist_c = np.linalg.norm(a - b)

  if not np.allclose(eucl_dist_q, eucl_dist_c, atol=1e-4):
    raise AssertionError('Incorrect computation')
  print(f'  Classic: {eucl_dist_c:.2f}, quantum: {eucl_dist_q:.2f}, Correct')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Compute Quantum Euclidean Distance.')

  for _ in range(10):
    a = np.array(random.choices(range(10), k=4))
    b = np.array(random.choices(range(10), k=4))
    run_experiment(a, b)

  for _ in range(10):
    a = np.array(random.choices(range(100), k=8))
    b = np.array(random.choices(range(100), k=8))
    run_experiment(a, b)


if __name__ == '__main__':
  app.run(main)
