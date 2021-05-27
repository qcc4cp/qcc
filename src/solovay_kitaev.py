# python3
"""Example: Solovay-Kitaev Algorithm for gate approximation."""

import random

from absl import app
import numpy as np

from src.lib import helper
from src.lib import ops
from src.lib import state


def to_su2(U):
  """Convert a 2x2 unitary to a unitary with determinant 1.0."""

  return np.sqrt(1 / np.linalg.det(U)) * U


def trace_dist(U, V):
  """Compute trace distance between two 2x2 matrices."""

  return np.real(0.5 * np.trace(np.sqrt((U - V).adjoint() @ (U - V))))


def create_unitaries(base, limit):
  """Create all combinations of all base gates, up to length 'limit'."""

  # Create bitstrings up to bitstring length limit-1:
  #  0, 1, 00, 01, 10, 11, 000, 001, 010, ...
  #
  # Multiply together the 2 base operators, according to their index.
  # Note: This can be optimized, by remembering the last 2^x results
  # and multiplying them with base gets 0, 1.
  #
  gate_list = []
  for width in range(limit):
    for bits in helper.bitprod(width):
      U = ops.Identity()
      for bit in bits:
        U = U @ base[bit]
      gate_list.append(U)
  return gate_list


def find_closest_u(gate_list, u):
  """Find the one gate in the list closest to u."""

  # Linear search over list of gates - is _very_ slow.
  # This can be optimized by using kd-trees.
  #
  min_dist, min_u = 10, ops.Identity()
  for gate in gate_list:
    tr_dist = trace_dist(gate, u)
    if tr_dist < min_dist:
      min_dist, min_u = tr_dist, gate
  return min_u


def u_to_bloch(U):
  """Compute angle and axis for a unitary."""

  angle = np.real(np.arccos((U[0, 0] + U[1, 1])/2))
  sin = np.sin(angle)
  if sin < 1e-10:
    axis = [0, 0, 1]
  else:
    nx = (U[0, 1] + U[1, 0]) / (2j * sin)
    ny = (U[0, 1] - U[1, 0]) / (2 * sin)
    nz = (U[0, 0] - U[1, 1]) / (2j * sin)
    axis = [nx, ny, nz]
  return axis, 2 * angle


def gc_decomp(U):
  """Group commutator decomposition."""

  def diagonalize(U):
    _, V = np.linalg.eig(U)
    return ops.Operator(V)

  # Get axis and theta for the operator.
  axis, theta = u_to_bloch(U)

  # The angle phi comes from eq 10 in 'The Solovay-Kitaev Algorithm' by
  # Dawson, Nielsen. It is fully derived in the book section on the
  # theorem and algorithm.
  phi = 2.0 * np.arcsin(np.sqrt(
      np.sqrt((0.5 - 0.5 * np.cos(theta / 2)))))

  V = ops.RotationX(phi)
  if axis[2] > 0:
    W = ops.RotationY(2 * np.pi - phi)
  else:
    W = ops.RotationY(phi)

  Ud = diagonalize(U)
  VWVdWdd = diagonalize(V @ W @ V.adjoint() @ W.adjoint())
  S = Ud @ VWVdWdd.adjoint()

  V_hat = S @ V @ S.adjoint()
  W_hat = S @ W @ S.adjoint()
  return V_hat, W_hat


def sk_algo(U, gates, n):
  """Solovay-Kitaev Algorithm."""

  if n == 0:
    return find_closest_u(gates, U)
  else:
    U_next = sk_algo(U, gates, n-1)
    V, W = gc_decomp(U @ U_next.adjoint())
    V_next = sk_algo(V, gates, n-1)
    W_next = sk_algo(W, gates, n-1)
    return V_next @ W_next @ V_next.adjoint() @ W_next.adjoint() @ U_next


def random_gates(min_length, max_length, num_experiments):
  """Just create random sequences, find the best."""

  base = [to_su2(ops.Hadamard()), to_su2(ops.Tgate())]

  U = (ops.RotationX(2.0 * np.pi * random.random()) @
       ops.RotationY(2.0 * np.pi * random.random()) @
       ops.RotationZ(2.0 * np.pi * random.random()))

  min_dist = 1000
  for _ in range(num_experiments):
    seq_length = min_length + random.randint(0, max_length)
    U_approx = ops.Identity()

    for _ in range(seq_length):
      g = random.randint(0, 1)
      U_approx = U_approx @ base[g]

    dist = trace_dist(U, U_approx)
    min_dist = min(dist, min_dist)

  phi1 = U(state.zero)
  phi2 = U_approx(state.zero)
  print('Trace Dist: {:.4f} State: {:6.4f}%'.
        format(min_dist,
               100.0 * (1.0 - np.real(np.dot(phi1, phi2.conj())))))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  num_experiments = 10
  depth = 8
  recursion = 3
  print('SK algorithm - depth: {}, recursion: {}, experiments: {}'.
        format(depth, recursion, num_experiments))

  base = [to_su2(ops.Hadamard()), to_su2(ops.Tgate())]

  gates = create_unitaries(base, depth)
  sum_dist = 0.0
  for i in range(num_experiments):

      U = (ops.RotationX(2.0 * np.pi * random.random()) @
           ops.RotationY(2.0 * np.pi * random.random()) @
           ops.RotationZ(2.0 * np.pi * random.random()))

      U_approx = sk_algo(U, gates, recursion)

      dist = trace_dist(U, U_approx)
      sum_dist += dist

      phi1 = U(state.zero)
      phi2 = U_approx(state.zero)
      print('[{:2d}]: Trace Dist: {:.4f} State: {:6.4f}%'.
            format(i, dist,
                   100.0 * (1.0 - np.real(np.dot(phi1, phi2.conj())))))

  print('Gates: {}, Mean Trace Dist:: {:.4f}'.
        format(len(gates), sum_dist / num_experiments))

  min_length = 10
  max_delta = 50
  max_tries = 100
  print('Random Experiment, seq length: {} - {}, tries: {}'
        .format(min_length, max_delta, max_tries))
  for i in range(num_experiments):
    random_gates(min_length, max_delta, max_tries)


if __name__ == '__main__':
  app.run(main)
