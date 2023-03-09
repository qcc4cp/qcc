# python3
"""Example: Solovay-Kitaev Algorithm for gate approximation."""

import random

from absl import app
import numpy as np

from src.lib import helper
from src.lib import ops
from src.lib import state


def to_su2(u):
  """Convert a 2x2 unitary to a unitary with determinant 1.0."""

  return np.sqrt(1 / np.linalg.det(u)) * u


def trace_dist(u, v):
  """Compute trace distance between two 2x2 matrices."""

  return np.real(0.5 * np.trace(np.sqrt((u - v).adjoint() @ (u - v))))


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
      u = ops.Identity()
      for bit in bits:
        u = u @ base[bit]
      gate_list.append(u)
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


def u_to_bloch(u):
  """Compute angle and axis for a unitary."""

  angle = np.real(np.arccos((u[0, 0] + u[1, 1]) / 2))
  sin = np.sin(angle)
  if sin < 1e-10:
    axis = [0, 0, 1]
  else:
    nx = (u[0, 1] + u[1, 0]) / (2j * sin)
    ny = (u[0, 1] - u[1, 0]) / (2 * sin)
    nz = (u[0, 0] - u[1, 1]) / (2j * sin)
    axis = [nx, ny, nz]
  return axis, 2 * angle


def gc_decomp(u):
  """Group commutator decomposition."""

  def diagonalize(u):
    _, v = np.linalg.eig(u)
    return ops.Operator(v)

  # Get axis and theta for the operator.
  axis, theta = u_to_bloch(u)

  # The angle phi comes from eq 10 in 'The Solovay-Kitaev Algorithm' by
  # Dawson, Nielsen. It is fully derived in the book section on the
  # theorem and algorithm.
  phi = 2.0 * np.arcsin(np.sqrt(np.sqrt((0.5 - 0.5 * np.cos(theta / 2)))))

  v = ops.RotationX(phi)
  if axis[2] > 0:
    w = ops.RotationY(2 * np.pi - phi)
  else:
    w = ops.RotationY(phi)

  ud = diagonalize(u)
  vwvdwd = diagonalize(v @ w @ v.adjoint() @ w.adjoint())
  s = ud @ vwvdwd.adjoint()

  v_hat = s @ v @ s.adjoint()
  w_hat = s @ w @ s.adjoint()
  return v_hat, w_hat


def sk_algo(u, gates, n):
  """Solovay-Kitaev Algorithm."""

  if n == 0:
    return find_closest_u(gates, u)
  else:
    u_next = sk_algo(u, gates, n - 1)
    v, w = gc_decomp(u @ u_next.adjoint())
    v_next = sk_algo(v, gates, n - 1)
    w_next = sk_algo(w, gates, n - 1)
    return v_next @ w_next @ v_next.adjoint() @ w_next.adjoint() @ u_next


def random_gates(min_length, max_length, num_experiments):
  """Just create random sequences, find the best."""

  base = [to_su2(ops.Hadamard()), to_su2(ops.Tgate())]

  u = (ops.RotationX(2.0 * np.pi * random.random()) @
       ops.RotationY(2.0 * np.pi * random.random()) @
       ops.RotationZ(2.0 * np.pi * random.random()))

  min_dist = 1000
  for _ in range(num_experiments):
    seq_length = min_length + random.randint(0, max_length)
    u_approx = ops.Identity()

    for _ in range(seq_length):
      g = random.randint(0, 1)
      u_approx = u_approx @ base[g]

    dist = trace_dist(u, u_approx)
    min_dist = min(dist, min_dist)

  phi1 = u(state.zeros(1))
  phi2 = u_approx(state.zeros(1))
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

    u = (ops.RotationX(2.0 * np.pi * random.random()) @
         ops.RotationY(2.0 * np.pi * random.random()) @
         ops.RotationZ(2.0 * np.pi * random.random()))

    u_approx = sk_algo(u, gates, recursion)

    dist = trace_dist(u, u_approx)
    sum_dist += dist

    phi1 = u(state.zeros(1))
    phi2 = u_approx(state.zeros(1))
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
