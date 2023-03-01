# python3
"""Example: CHSH implementation and measurement."""

import random

from absl import app
import numpy as np
from src.lib import bell
from src.lib import helper
from src.lib import ops
from src.lib import state


# The CHSH game, named after Clauser, Horne, Shimony, and Holt,
#   https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.23.880
# is a simplified form of the Bell equation and is used to
# demonstrate the power of entanglement.
#
# As Prof. Vazirani says, we should think of entanglement as a
# resource that allows us to compute certain things better or
# faster than with classical resources.
#
# In the CHSH game, both Alice and Bob receive random bits
# $x$ and $y$ from a referee Charlie. Based on the bit values and
# a strategy discussed between Alice and Bob beforehand they will
# respond with bit values $a$ and $b$. During the game, Alice and
# Bob cannot communicate. The goal of the game is to produce matching
# bits $a$ and $b$, except when both $x = y = 1$. In this case
# $a$ and $b$ must differ.
#
# In closed form, the winning condition can be written as:
#      if x * y == (a + b) % 2:
#         wins += 1
#
# The best possible classical strategy for both Alice and Bob is
# to always respond with a 0, which leads to a $3/4$ success probability.
# Just using entanglement also doesn't work - on measurement, both Alice's
# and Bob's qubit will produce a matching value and this also represents
# a winning percentage of 3/4 (this strategy doesn't handle the [1, 1]
# case).
#
# In the quantum case, Alice and Bob share an entangled qubit in the state
#    $\psi = 1/\sqrt{2}(|0_A0_B\rangle + |1_A1_B\rangle)$.
# When Alice receives a bit $x = 0$ she measures in the
#    $|0\rangle, |1\rangle$ basis
# and if she gets $x = 1$ she measures in the Hadamard
#    $|+\rangle, |-\rangle$ basis.
# Correspondingly, if Bob receives $y = 0$ he measure in
#    $|a_0\rangle, |a_1\rangle$, where
#    $a_0 =  \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle$
#    $a_1 = -\sin(\pi/8)|0\rangle + \cos(\pi/8)|1\rangle$
# If he receives $y = 1$ he measures in
#    $|b_0\rangle, |b_1\rangle$, where
#    $b_0 = \cos(\pi/8)|0\rangle - \sin(\pi/8)|1\rangle$
#    $b_1 = \sin(\pi/8)|0\rangle + \cos(\pi/8)|1\rangle$
#
# All of the measurement bases are rotated bt $\pi/8$ from each other.
# With this, it can be shown that the success probability increases from
# 3/4 to cos^2 pi/8 (= ~0.86). Let's try this out here with simulated
# random measurements in the various bases.


def measure(psi: state.State):
  """Simulated, probabilistic measurement."""

  # We assume that random numbers are evenly distributed,
  # which will ensure that states are selected weighted
  # by their probabilities.
  #
  r = random.random() - 0.001
  total = 0
  for i in range(len(psi)):
    total += psi[i] * psi[i].conj()
    if r < total:
      psi = helper.val2bits(i, 2)
      return psi[0], psi[1]


def run_experiments(experiments: int, alpha: float) -> float:
  """Run CHSH experiments for a given angle."""

  wins = 0
  for _ in range(experiments):
    x = random.randint(0, 1)
    y = random.randint(0, 1)
    psi = bell.bell_state(0, 0)

    if x == 0:
      pass
    if x == 1:
      psi = ops.RotationY(2.0 * alpha)(psi, 0)
    if y == 0:
      psi = ops.RotationY(alpha)(psi, 1)
    if y == 1:
      psi = ops.RotationY(-alpha)(psi, 1)

    a, b = measure(psi)
    if x * y == (a + b) % 2:
      wins += 1

  return wins / experiments * 100.0


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Quantum CHSH evaluation.')

  # Compute results for optimal value.
  #
  percent = run_experiments(1000, 2.0 * np.pi / 8)
  print(f'Optimal Angle 2 pi / 8, winning: {percent:.1f}%')
  if percent < 80.0:
    raise AssertionError('Incorrect result, should reach above 80%')

  # Run a few incrementals and see how the results change.
  #
  steps = 32
  inc_angle = (2.0 * np.pi / 8) / (steps / 2)
  for i in range(0, 66, 2):
    percent = run_experiments(500, inc_angle * i)
    s = '(opt)' if i == 16 else ''
    print(
        f'{i:2d} * Pi/64 = {inc_angle * i:.2f}: winning: {percent:5.2f}% '
        f'{"#" * int(percent/3)}{s}'
    )


if __name__ == '__main__':
  app.run(main)
