# python3
"""Generators for various entangled states, eg., the Bell states."""

import numpy as np

from src.lib import ops
from src.lib import state


def bell_state(a: int, b: int) -> state.State:
  """Make one of the four bell states with a, b from {0,1}."""

  if a not in [0, 1] or b not in [0, 1]:
    raise ValueError('Bell state arguments are bits and must be 0 or 1.')

  psi = state.bitstring(a, b)
  psi = ops.Hadamard()(psi)
  return ops.Cnot()(psi)


def ghz_state(nbits: int) -> state.State:
  """Make a maximally entangled nbits state (GHZ State)."""

  # Simple construction via:
  #
  # |0> --- H --- o ---------
  # |0> ----------X --- o ---
  # |0> ----------------X ---  ...
  #
  psi = state.zeros(nbits)
  psi = ops.Hadamard()(psi)
  for offset in range(nbits - 1):
    psi = ops.Cnot(0, 1)(psi, offset)
  return psi


def w_state() -> state.State:
  """Make a 3-qubit |W> state)."""

  # A 3-qubit |W> state (named after Wolfgang Duerr (2002)) is this state:
  #   1/sqrt(3)(|001> + |010> + |100>)
  #
  # This construction follows https://en.wikipedia.org/wiki/W_state:
  #
  # |0> -- Ry(phi3) - o ------o - X --
  #                   |       |
  # |0> ------------- H - o - X ------
  #                       |
  # |0> ----------------- X ----------
  #
  psi = state.zeros(3)
  phi3 = 2 * np.arccos(1 / np.sqrt(3))
  psi = ops.RotationY(phi3)(psi, 0)
  psi = ops.ControlledU(0, 1, ops.Hadamard())(psi, 0)
  psi = ops.Cnot(1, 2)(psi, 1)
  psi = ops.Cnot(0, 1)(psi, 0)
  psi = ops.PauliX()(psi, 0)
  return psi
