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
"""Helper functions."""

import itertools
import math
from collections.abc import Iterator, Sequence

import numpy as np


def bitprod(nbits: int) -> Iterator[tuple[int, ...]]:
  """Produce the iterable cartesian product of nbits over {0, 1}.

  Args:
    nbits: The number of bits in each generated tuple.

  Yields:
    Tuples of length nbits enumerating all 2**nbits combinations of
    0 and 1, in lexicographic order.
  """
  for bits in itertools.product([0, 1], repeat=nbits):
    yield bits


def bits2val(bits: Sequence[int]) -> int:
  """For given bits, compute the decimal integer.

  Args:
    bits: A sequence of 0/1 values in high-to-low bit order. For
      example, [1, 1, 0] represents the value 6.

  Returns:
    The non-negative integer encoded by the bits.
  """
  return sum(v * (1 << (len(bits) - i - 1)) for i, v in enumerate(bits))


def val2bits(val: int, nbits: int) -> list[int]:
  """Convert decimal integer to list of {0, 1}.

  Args:
    val: The non-negative integer to convert.
    nbits: The number of bits to produce; the result is zero-padded
      on the left to this width.

  Returns:
    A list of nbits 0/1 values in high-to-low bit order. For example,
    the value 6 is returned as [1, 1, 0].
  """
  return [int(c) for c in format(val, f'0{nbits}b')]


def bits2frac(bits: Sequence[int]) -> float:
  """For given bits, compute the binary fraction.

  Args:
    bits: A sequence of 0/1 values interpreted as the bits after the
      binary point, most significant first. For example, (0, 1)
      represents 0.01b = 0.25.

  Returns:
    The value in [0, 1) represented by the binary fraction.
  """
  return sum(bit * 2 ** (-idx - 1) for idx, bit in enumerate(bits))


def frac2bits(val: float, nbits: int) -> list[int]:
  """Approximate a float with n binary fractions.

  Args:
    val: The value to approximate. Must be strictly less than 1.0.
    nbits: The number of binary-fraction bits to produce.

  Returns:
    A list of nbits 0/1 values, most significant first, whose binary
    fraction approximates val.

  Raises:
    ValueError: If val is not strictly less than 1.0.
  """
  if val >= 1.0:
    raise ValueError('frac2bits: value must be strictly < 1.0')
  res = []
  while nbits:
    nbits -= 1
    val *= 2
    res.append(int(val))
    val -= int(val)
  return res


def density_to_cartesian(rho: np.ndarray) -> tuple[float, float, float]:
  """Compute Bloch sphere coordinates from 2x2 density matrix.

  Args:
    rho: A 2x2 single-qubit density matrix.

  Returns:
    A tuple (x, y, z) of the real Bloch-sphere coordinates.
  """
  a = rho[0, 0]
  b = rho[1, 0]
  x = 2.0 * b.real
  y = 2.0 * b.imag
  z = 2.0 * a - 1.0

  return np.real(x), np.real(y), np.real(z)


def qubit_to_bloch(psi: np.ndarray) -> tuple[float, float, float]:
  """Compute Bloch sphere coordinates from 2x1 state vector/qubit.

  Args:
    psi: A single-qubit state vector (2 amplitudes).

  Returns:
    A tuple (x, y, z) of the real Bloch-sphere coordinates.
  """
  return density_to_cartesian(np.outer(psi, psi.conj()))


def dump_bloch(x: float, y: float, z: float) -> None:
  """Textual output for Bloch sphere coordinates.

  Args:
    x: The Bloch-sphere x coordinate.
    y: The Bloch-sphere y coordinate.
    z: The Bloch-sphere z coordinate.
  """
  print(f'x: {x:.2f}, y: {y:.2f}, z: {z:.2f}')


def qubit_dump_bloch(psi: np.ndarray) -> None:
  """Print Bloch coordinates for state psi.

  Args:
    psi: A single-qubit state vector (2 amplitudes).
  """
  x, y, z = qubit_to_bloch(psi)
  dump_bloch(x, y, z)


def pi_fractions(val: float, pi: str = 'pi') -> str:
  """Convert a value into a string as a fraction of pi.

  Args:
    val: The value to express as a fraction of pi, or None.
    pi: The symbol to use for pi in the output string.

  Returns:
    A string such as '3*pi/2' or '-pi/2' if val closely matches a
    small fraction of pi, an empty string if val is None, '0' if val
    is 0, or the plain string representation of val otherwise.
  """
  if val is None:
    return ''
  if val == 0:
    return '0'
  for pi_multiplier in range(1, 4):
    for denom in range(-128, 128):
      if denom and math.isclose(val, pi_multiplier * math.pi / denom):
        pi_str = ''
        if pi_multiplier != 1:
          pi_str = f'{abs(pi_multiplier)}*'
        if denom == -1:
          return f'-{pi_str}{pi}'
        if denom < 0:
          return f'-{pi_str}{pi}/{-denom}'
        if denom == 1:
          return f'{pi_str}{pi}'
        return f'{pi_str}{pi}/{denom}'

  # couldn't find fractional, just return original value.
  return f'{val}'
