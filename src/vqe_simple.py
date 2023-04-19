# python3
"""Example: Simple Variational Quantum Eigensolver."""


# VQE is a classic-quantum algorithm to find the Eigenvalues of
# a matrix H, which may be too large for classical methods (solving
# the characteristic equation).
#
# The algorithm has two parts:
#    1) Prepare a quantum state psi(theta)
#    2) Measure the expectation value of H in psi:
#        <H> = <psi(theta)|H|psi(theta)>
#
# This expectation value is always greater or equal to the lowest
# Eigenvalue lambda of H (variational theorem):
#        <H> >= lambda
#
# We need to chose real valued parameters 'theta' such that the expectation
# value is minimized.
#
# This example is based on the youtube video:
#    https://www.youtube.com/watch?v=E947xs9-Mso
# from Pranav Gokhale at ISCA 2018.
#
# Note: In the youtube video the larger matrix H contains a subterm
#       -1.04*sigma_z*I. So in order to minimize the expectation value
#       of the larger matrix, one has to maximize the expectation value
#       for sigma_z*I, which is what we do below. The algorithm works
#       just the same if we'd look for a minimum (-1.0) instead of 1.0
#
# Note: The vide also does _not_ actually multiply the ansatz with XxI,
# as that is the default measurement operator that requires no additional
# gates or mechanics to make anything work. In the book there is a big
# chapter on "Measuring in the Pauli Basis", which is important to
# understand to appreciate the code below.

import math
import random

from absl import app
from absl import flags
import numpy as np

from src.lib import circuit
from src.lib import ops

flags.DEFINE_integer('experiments', 1000, 'Number of experiments')
flags.DEFINE_integer('shots', 1000, 'Number of random samples')


# We want to measure sigma_z x I, which is this matrix:
#
#   1  0  0  0
#   0  1  0  0
#   0  0 -1  0
#   0  0  0 -1
#
# The Eigenvalues of this matrix are +1 for |00> and |01>
#                                are -1 for |10> and |11>
#
# Measuring an observable (sigma_z x I) means projecting the
# quantum state to one of the Eigenstates of the observable.
#
# Since the Eigenstates for sigma_z x I are so simple, they
# can just be measured.


# These are the 10 angles for the ansatz. Where this ansatz specifically
# came from is not clear. In general, the choice of ansatz can be arbitrary.
# Yet, some perform better than others and there seems to be a theory behind it.
# We take the ansatz as described in the youtube video.
#
angles = [0.0] * 10


def full_ansatz(qc):
  """The Ansatz circuit for this example."""

  qc.reg(2, 'r')

  # Step 1: Initial Rotations.
  qc.rx(0, angles[0])
  qc.rx(1, angles[1])
  qc.rz(0, angles[2])
  qc.rz(1, angles[3])

  # Step 2: Entangler.
  qc.h([0, 1])
  qc.cx(0, 1)
  qc.cx(1, 0)

  # Step 3: Final Rotations.
  qc.rz(0, angles[4])
  qc.rz(1, angles[5])
  qc.rx(0, angles[6])
  qc.rx(1, angles[7])
  qc.rz(0, angles[8])
  qc.rz(1, angles[9])


def run_two_qubit_zi_experiment():
  """Run VQE experiments with a given ansatz."""

  # Best achieved result. Goal is to get as close to +1 as possible.
  max_expect = 0.0

  # Perform experiments with randomly selected angles.
  for experiment in range(flags.FLAGS.experiments):
    # Pick random angles.
    for i in range(10):
      angles[i] = random.random() * 2.0 * math.pi

    # Construct and run the circuit.
    qc = circuit.qc('vqe')
    full_ansatz(qc)
    qc.z(0)

    # Measure the probablities as computed from the amplitudes.
    # We only do this once per experiment.
    p0, _ = qc.measure_bit(0, collapse=True)
    p1, _ = qc.measure_bit(1, collapse=True)

    # Simulate multiple measurements by sampling over the probabilities
    # to obtain a distribution of sampled states. The measurements above
    # are the probablities that a state would be found in the |0> state.
    # For each bit, we compare this probability against another random value r.
    # If the measured probability is < r, we pretend we've actually measured an
    # |0> state, else a |1> state. We do this via sample_state() on both qubits.
    #
    def sample_state(prob_state0: float):
      if prob_state0 < random.random():
        return 1
      return 0

    num_shots = flags.FLAGS.shots
    counts = [0] * 4
    for _ in range(num_shots):
      bit0 = sample_state(p0)
      bit1 = sample_state(p1)
      counts[bit1 * 2 + bit0] += 1

    # Compute the expectation value from samples measurements. Again,
    #   |00> and |01> map to Eigenvalue +1
    #   |10> and |11> map to Eigenvalue -1
    #
    # This is a bit of cheating. In this example we _know_ the
    # Eigenvalues and can therefore properly construct the expectation
    # value. I'd think in the general case it has to actually be
    # computed with <psi|H|psi>, which is still O(n^2).
    #
    expect = (counts[0] + counts[1] - counts[2] - counts[3]) / num_shots

    # Update and print currently best result.
    #
    if expect > max_expect:
      max_expect = expect
      print('Max expecation of H for experiment {:5d}: {:.4f} (target: 1.0)'.
            format(experiment, max_expect))
      print('  |00>: {}, |01>: {}, |10>: {}, |11>: {}'.format(
          counts[0], counts[1], counts[2], counts[3]))
      print('  ', end='')
      for i in range(10):
        print('{:.1f} '.format(angles[i] / 2 / math.pi * 360), end='')
      print()


def single_qubit_ansatz(theta: float, phi: float) -> circuit.qc:
  """Generate a single qubit ansatz."""

  qc = circuit.qc('single-qubit ansatz Y')
  qc.qubit(1.0)
  qc.rx(0, theta)
  qc.ry(0, phi)
  return qc


def run_single_qubit_mult():
  """Run experiments with single qubits."""

  # Construct Hamiltonian.
  hamil = (random.random() * ops.PauliX() +
           random.random() * ops.PauliY() +
           random.random() * ops.PauliZ())
  # Compute known minimum eigenvalue.
  eigvals = np.linalg.eigvalsh(hamil)

  # Brute force over the Bloch sphere.
  min_val = 1000.0
  for i in range(0, 180, 10):
    for j in range(0, 180, 10):
      theta = np.pi * i / 180.0
      phi = np.pi * j / 180.0

      # Build the ansatz with two rotation gates.
      ansatz = single_qubit_ansatz(theta, phi)

      # Compute <psi ! H ! psi>. Find smallest one, which will be
      # the best approximation to the minimum eigenvalue from above.
      # In this version, we just multiply out the result.
      psi = np.dot(ansatz.psi.adjoint(), hamil(ansatz.psi))
      if psi < min_val:
        min_val = psi

  # Result from brute force approach:
  print('Minimum: {:.4f}, Estimated: {:.4f}, Delta: {:.4f}'.format(
      eigvals[0], np.real(min_val), np.real(min_val - eigvals[0])))


def run_single_qubit_measure():
  """Run measurement experiments with single qubits."""

  # Construct Hamiltonian.
  a = random.random()
  b = random.random()
  c = random.random()
  hamil = a * ops.PauliX() + b * ops.PauliY() + c * ops.PauliZ()

  # Compute known minimum eigenvalue.
  eigvals = np.linalg.eigvalsh(hamil)

  min_val = 1000.0
  for i in range(0, 360, 5):
    for j in range(0, 180, 5):

      theta = np.pi * i / 360.0
      phi = np.pi * j / 180.0

      # X Basis
      qc = single_qubit_ansatz(theta, phi)
      qc.h(0)
      val_a = a * qc.pauli_expectation(0)

      # Y Basis
      qc = single_qubit_ansatz(theta, phi)
      qc.sdag(0)
      qc.h(0)
      val_b = b * qc.pauli_expectation(0)

      # Z Basis
      qc = single_qubit_ansatz(theta, phi)
      val_c = c * qc.pauli_expectation(0)

      expectation = val_a + val_b + val_c
      if expectation < min_val:
        min_val = expectation

  print('Minimum eigenvalue: {:.3f}, Delta: {:.3f}'
        .format(eigvals[0], min_val - eigvals[0]))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for _ in range(5):
    run_single_qubit_measure()
  for _ in range(5):
    run_single_qubit_mult()
  run_two_qubit_zi_experiment()


if __name__ == '__main__':
  app.run(main)
