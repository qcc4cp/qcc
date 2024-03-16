# python3
"""Example: Decompose states in Bell basis."""

import math
import random

from absl import app
import numpy as np

from src.lib import bell
from src.lib import state


# Decompose a state into the Bell basis:
#   psi = c_0 * b_00 + c1 * b_01 + c2 * b_10 + c3 * b_11.
#
# We produce a random state 'psi' first.
#
# Then we compute the inner product between psi and
# all of the four Bell states (the Bell basis) to compute
# the factors c_i.
#
# Finally, we reconstruct the state by multiplying and
# adding the factor with the respective Bell states.
#
# This new state and the original state must be identical.
#
def run_experiment():
  """Run an single state decomposition."""

  psi = np.random.random([4]) + np.random.random([4]) * 1j
  psi = state.State(psi / np.linalg.norm(psi))

  bells = [bell.bell_state(0, 0),
           bell.bell_state(0, 1),
           bell.bell_state(1, 0),
           bell.bell_state(1, 1)]

  c = [0] * 4
  for idx, b in enumerate(bells):
    c[idx] = np.inner(psi, b)

  new_psi = [0] * 4
  for idx in range(4):
    new_psi = new_psi + c[idx] * bells[idx]
    
  assert np.allclose(psi, new_psi), 'Incorrect result.'


def main(argv):
  print('Express 1000 random states in the Bell basis.')

  for _ in range(1000):
    run_experiment()


if __name__ == '__main__':
  app.run(main)
