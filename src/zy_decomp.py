# python3
"""Example: Z-Y dcomposition of a unitary U."""

import cmath

from absl import app
import numpy as np
import scipy.stats

from src.lib import ops

# The "Z-Y decomposition for a single qubit" shows that
# any unitary U can be decomposed into Rz and Ry rotations
# with 4 parameters alpha, beta, gamma, delta as follows:
#
#   U = e^(i*alpha) * Rz(beta) * Ry(gamma) * Rz(delta)
#
# The question is how to find the 4 parameters for
# a given U?
#
# One answer was provided here - it solves correctly for alpha and gamma:
#    https://threeplusone.com/pubs/on_gates.pdf
#
# The approach from this paper can be found in code, here:
#    https://github.com/gecrooks/quantumflow-dev/blob/master/quantumflow/decompositions.py
#
# Another publication solves correctly for beta and delta:
#    https://quantumcomputing.stackexchange.com/questions/\
#    16256/what-is-the-procedure-of-finding-z-y-decomposition-of-unitary-matrices
#
# Let's try and implement this here!


def make_u_zy(alpha, beta, gamma, delta):
  """Construct unitary via Z-Y from the 4 parameters."""

  return (
      ops.RotationZ(beta) @ ops.RotationY(gamma) @ ops.RotationZ(delta)
  ) * cmath.exp(1.0j * alpha)


def make_u_xy(alpha, beta, gamma, delta):
  """Construct unitary via X-Y from the 4 parameters."""

  return (
      ops.RotationX(beta) @ ops.RotationY(gamma) @ ops.RotationX(delta)
  ) * cmath.exp(1.0j * alpha)


def zy_decompose(umat):
  """Perform Z-Y decomposition of unitary operator in SU(2)."""

  a = umat[0][0]
  b = umat[0][1]
  c = umat[1][0]

  det = np.linalg.det(umat)
  alpha = 0.5 * np.arctan2(det.imag, det.real)

  if a >= b:
    gamma = 2 * np.arccos(abs(a))
  else:
    gamma = 2 * np.arcsin(abs(b))

  # TODO(rhundt): Handle cases with gamma very close to 0 or Pi

  beta = cmath.phase(c) - cmath.phase(a)
  delta = cmath.phase(-b) - cmath.phase(a)

  return alpha, beta, gamma, delta


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  iterations = 1000
  print(f'Perform {iterations} random Z-Y and X-Y decompositions.')

  for i in range(iterations):
    #
    # Construct a random unitary operator and put into SU(2).
    #
    u = scipy.stats.unitary_group.rvs(2)
    umat = np.sqrt(1 / np.linalg.det(u)) * u

    # Now decompose this operator and find the four parameters.
    #
    alpha, beta, gamma, delta = zy_decompose(umat)

    # Construct another operator from these newly found
    # parameters and make sure that the resulting
    # operator matches the one from above.
    #
    unew = make_u_zy(alpha, beta, gamma, delta)

    if not np.allclose(umat, unew, atol=1e-4):
      print(f'decomp : {i:2d}: {alpha:.3f} {beta:.3f} {gamma:.3f} {delta:.3f}')
      raise AssertionError('Z-Y decomposition failed')

    # According to Problem 4.10 in Nielsen/Chuang, we can also derive
    # an XY-decomposition. How? See:
    #
    #  https://quantumcomputing.stackexchange.com/a/32088/11582
    #
    # In essence, we change the axes by rotating U to U' = HUH.
    # We compute the Y-Z decomposition for U' and note that:
    #    U = HU'H
    # and correspondingly:
    #    H Rz(beta')  H -> Rx(beta)
    #    H Rz(delta') H -> Rx(delta)
    # and
    #    H Ry(gamma') H -> Ry (-gamma)
    #
    udash = ops.Hadamard() @ umat @ ops.Hadamard()

    alpha, beta, gamma, delta = zy_decompose(udash)
    unew = make_u_xy(alpha, beta, -gamma, delta)

    if not np.allclose(umat, unew, atol=1e-4):
      print(f'decomp : {i:2d}: {alpha:.3f} {beta:.3f} {gamma:.3f} {delta:.3f}')
      raise AssertionError('X-Y decomposition failed')

  print('Success')


if __name__ == '__main__':
  np.set_printoptions(precision=3)
  app.run(main)
