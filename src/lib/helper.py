# python3
"""Helper functions."""

import itertools
import math
import numpy as np

def to_rad(angle) -> float:
  """Convert angle (in degree) to radiants."""

  return math.pi / 180 * angle


def to_deg(rad) -> float:
  """Convert radiants to angle."""

  return rad / math.pi * 180.0


def bitprod(nbits):
  """Produce the iterable cartesian of nbits {0, 1}."""

  for bits in itertools.product([0, 1], repeat=nbits):
    yield bits


def bits2val(bits):
  """For a given enumeratable bits, compute the corresponding decimal integer."""

  # We assume bits are given in high to low order. For example,
  # the bits [1, 1, 0] will produce the value 6.
  return sum(v * (1 << (len(bits)-i-1)) for i, v in enumerate(bits))


def val2bits(val, nbits):
  """Convert decimal integer to list of {0, 1}."""

  # We return the bits in order high to low. For example,
  # the value 6 is being returned as [1, 1, 0].
  return [int(c) for c in format(val, '0{}b'.format(nbits))]


def bits2frac(bits, length):
  """For a given enumeratable bits, compute the binary fraction."""

  return sum(bits[i] * 2**(-i-1) for i in range(length))


def density_to_cartesian(rho):
  """Compute Bloch sphere coordinates from 2x2 density matrix."""

  a = rho[0, 0]
  b = rho[1, 0]
  x = 2.0 * b.real
  y = 2.0 * b.imag
  z = 2.0 * a - 1.0

  return np.real(x), np.real(y), np.real(z)
