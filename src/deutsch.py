# python3
"""Example: Deutsch's Algorithm."""

import math

from typing import Callable
from absl import app
import numpy as np

from src.lib import ops
from src.lib import state


def make_f(flavor: int) -> Callable[[int], int]:
  """Return a 1-bit constant or balanced function f. 4 flavors."""

  # The 4 versions are:
  #    f(0) -> 0, f(1) -> 0  constant
  #    f(0) -> 0, f(1) -> 1  balanced
  #    f(0) -> 1, f(1) -> 0  balanced
  #    f(0) -> 1, f(1) -> 1  constant
  flavors = [[0, 0], [0, 1], [1, 0], [1, 1]]

  def f(bit: int) -> int:
    """Return f(bit) for one of the 4 possible function types."""
    return flavors[flavor][bit]

  return f


def make_uf(f: Callable[[int], int]) -> ops.Operator:
  """Simple way to generate the 2-qubit, 4x4 Deutsch Oracle."""

  # This is how the Deutsch Oracle is being constructed.
  #
  #  The input state is one of these 2 qubit tensor products:
  #
  #  |00> = [1, 0, 0, 0].T
  #  |01> = [0, 1, 0, 0].T
  #  |10> = [0, 0, 1, 0].T
  #  |11> = [0, 0, 0, 1].T
  #
  #  Only the 2nd qubit is being modified by f (note that f
  #  is a function of x):
  #    |x, y> -> |x, y ^ f(x)>  (xor)
  #
  #  For f(0)=0, f(1)=0 (^ being add modulo 2, or xor):
  #
  #  x  y   ^ f(x)  =  new state
  #  ---------------------------
  #  0, 0   ^ 0     0    0, 0
  #  0, 1   ^ 0     1    0, 1
  #  1, 0   ^ 0     0    1, 0
  #  1, 1   ^ 0     1    1, 1
  #
  #  Which is being achieved by the identity matrix:
  #    1 0 0 0
  #    0 1 0 0
  #    0 0 1 0
  #    0 0 0 1
  #
  #  For f(0)=0, f(1)=1:
  #
  #  x  y   ^ f(x)  =  new state
  #  ---------------------------
  #  0, 0   ^ 0     0    0, 0
  #  0, 1   ^ 0     1    0, 1
  #  1, 0   ^ 1     1    1, 1
  #  1, 1   ^ 1     0    1, 0
  #
  #  Which is being achieved by the identity matrix:
  #    1 0 0 0
  #    0 1 0 0
  #    0 0 0 1
  #    0 0 1 0
  #
  #  The unitary matrices for the other 2 cases can be computed
  #  the exact same, mechanical way. Interpret the (x,y) as bits
  #  indexing the 4x4 operator array, then this code will build Uf:
  #
  u = np.zeros(16).reshape(4, 4)
  for col in range(4):
    y = col & 1
    x = col & 2
    fx = f(x >> 1)
    xor = y ^ fx
    u[col][x + xor] = 1.0

  op = ops.Operator(u)
  if not op.is_unitary():
    raise AssertionError('Produced non-unitary operator.')
  return op


def run_experiment(flavor: int) -> None:
  """Run full experiment for a given flavor of f()."""

  f = make_f(flavor)
  u = make_uf(f)
  h = ops.Hadamard()

  psi = h(state.zeros(1)) * h(state.ones(1))
  psi = u(psi)
  psi = (h * ops.Identity())(psi)

  p0, _ = ops.Measure(psi, 0, tostate=0, collapse=False)

  print('f(0) = {:.0f} f(1) = {:.0f}'.format(f(0), f(1)), end='')
  if math.isclose(p0, 0.0, abs_tol=1e-5):
    print('  balanced')
    if flavor == 0 or flavor == 3:
      raise AssertionError('Invalid result, expected balanced.')
  else:
    print('  constant')
    if flavor == 1 or flavor == 2:
      raise AssertionError('Invalid result, expected constant.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run_experiment(0)
  run_experiment(1)
  run_experiment(2)
  run_experiment(3)


if __name__ == '__main__':
  app.run(main)
