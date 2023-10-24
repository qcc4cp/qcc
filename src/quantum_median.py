# python3
"""Example: Quantum Median Computation, simulated classically."""

import random
from absl import app
import numpy as np


# This algorithm utilizes two quantum techniques:
#    Quantum Mean Estimation (in quantum_mean.py)
#    Minimum Finding (in minimum_finding.py)
#
# We do _not_ reimplement those here, we just check classically whether
# the overall approach using these techniques would work.
#
# What is this approach?
#
# According to [https://arxiv.org/abs/1106.4267], the median z is the
# value that minimizes
#
#   \sum_x |f(x) - f(z)|
#
# For each value z in the input array, We compute the difference
# vector [x[0] - z, x[1] - z, ..., x[n-1] - z].
#
# Then we compute the mean of this vector with the quantum
# algorithm and finally find the minimum element z that would
# minimize the mean. That will have to be done with the quantum
# algorithm.
#
# Note: Computing the mean and minimum is clear. It is not clear
# how to compute the difference vector quantum'ly. It may just be
# one of those cases where we assume that that's easily doable.


def run_experiment(nbits: int):
  """Run a single median computation."""

  # Create the random state vector.
  x = np.array([random.randint(0, 2**nbits) for _ in range(2**nbits)])
  xn = x / np.linalg.norm(x)

  # Compute the mean(s) for each difference vector.
  median = min_mean = 1000
  for idx, z in enumerate(xn):
    # Make the difference vector. Note that this vector
    # may not be normalized!
    diff = [abs(xval - z) for xval in xn]

    # Normalization (required for quantum, also improves
    # accuracy by an order of magnitude).
    diffnorm = np.linalg.norm(diff)

    # Compute the mean (which we know how to do quantumly).
    mean = np.mean(diffnorm)

    # Find the minimum (which we would also know how to do quantumly).
    if mean < min_mean:
      min_mean = mean
      median = idx

  # Print and check results, we allow for a 1% deviation.
  print(
      f' Median ({nbits} qb): Classic: {np.mean(x):.3f},'
      f' Quantum: {x[median]:.3f}'
  )
  if max(np.mean(x), x[median]) / min(np.mean(x), x[median]) > 1.02:
    raise AssertionError('Incorrect median computation.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  print('Classic Sim of Quantum Median Computation.')

  for _ in range(10):
    run_experiment(nbits=10)


if __name__ == '__main__':
  np.set_printoptions(precision=4)
  app.run(main)
