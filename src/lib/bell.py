#!/usr/bin/env python3
# Copyright 2023 Robert Hundt.
#
# This file is part of the source code accompanying the book
# "Quantum Computing for Programmers" by Robert Hundt,
# Cambridge University Press. See www.cambridge.org/9781009548533
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Generators for various entangled states, eg., the Bell states."""

import numpy as np

from src.lib import ops
from src.lib import state


def bell_state(a: int, b: int) -> state.State:
  """Make one of the four Bell states with a, b from {0, 1}.

  Args:
    a: The first classical input bit, must be 0 or 1. It selects the
      relative phase (via the leading Hadamard qubit).
    b: The second classical input bit, must be 0 or 1. It selects the
      parity of the entangled pair (via the target qubit).

  Returns:
    The 2-qubit entangled Bell state corresponding to (a, b).

  Raises:
    ValueError: If a or b is not 0 or 1.
  """
  if a not in (0, 1) or b not in (0, 1):
    raise ValueError('Bits a and b must be 0 or 1.')
  psi = state.bitstring(a, b)
  psi = ops.Hadamard()(psi)
  return ops.Cnot()(psi)


def ghz_state(nbits: int) -> state.State:
  """Make a maximally entangled nbits state (GHZ state).

  Args:
    nbits: The number of qubits in the state. Must be at least 1.

  Returns:
    The nbits GHZ state 1/sqrt(2) (|0...0> + |1...1>).

  Simple construction via:

    |0> --- H --- o ---------
    |0> ----------X --- o ---
    |0> ----------------X ---  ...
  """
  psi = state.zeros(nbits)
  psi = ops.Hadamard()(psi)
  for offset in range(nbits - 1):
    psi = ops.Cnot(0, 1)(psi, offset)
  return psi


def w_state() -> state.State:
  """Make a 3-qubit |W> state.

  Returns:
    The 3-qubit |W> state 1/sqrt(3) (|001> + |010> + |100>).

  The |W> state is named after Wolfgang Duerr (2002). This
  construction follows https://en.wikipedia.org/wiki/W_state:

    |0> -- Ry(phi3) - o ------o - X --
                      |       |
    |0> ------------- H - o - X ------
                          |
    |0> ----------------- X ----------
  """
  psi = state.zeros(3)
  phi3 = 2 * np.arccos(1 / np.sqrt(3))
  psi = ops.RotationY(phi3)(psi, 0)
  psi = ops.ControlledU(0, 1, ops.Hadamard())(psi, 0)
  psi = ops.Cnot(1, 2)(psi, 1)
  psi = ops.Cnot(0, 1)(psi, 0)
  psi = ops.PauliX()(psi, 0)
  return psi
