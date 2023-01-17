# python3
"""Example: Quantum Teleportation."""

import math

from absl import app
import numpy as np

from src.lib import bell
from src.lib import ops
from src.lib import state


def alice_measures(alice: state.State,
                   expect0: np.complexfloating, expect1: np.complexfloating,
                   qubit0: np.complexfloating, qubit1: np.complexfloating):
  """Force measurement and get teleported qubit."""

  # Alices measure her state and get a collapsed |qubit0 qubit1>.
  # She let's Bob know which one of the 4 combinations she obtained.

  # We force measurement here, collapsing to a state with the
  # first two qubits collapsed. Bob's qubit is still unmeasured.
  _, alice0 = ops.Measure(alice, 0, tostate=qubit0)
  _, alice1 = ops.Measure(alice0, 1, tostate=qubit1)

  # Depending on what was measured and communicated, Bob has to
  # one of these things to his qubit2:
  if qubit0 == 0 and qubit1 == 0:
    pass
  if qubit0 == 0 and qubit1 == 1:
    alice1 = ops.PauliX()(alice1, idx=2)
  if qubit0 == 1 and qubit1 == 0:
    alice1 = ops.PauliZ()(alice1, idx=2)
  if qubit0 == 1 and qubit1 == 1:
    alice1 = ops.PauliX()(ops.PauliZ()(alice1, idx=2), idx=2)

  # Now Bob measures his qubit (2) (without collapse, so we can
  # 'measure' it twice. This is not necessary, but good to double check
  # the maths).
  p0, _ = ops.Measure(alice1, 2, tostate=0, collapse=False)
  p1, _ = ops.Measure(alice1, 2, tostate=1, collapse=False)

  # Alice should now have 'teleported' the qubit in state 'x'.
  # We sqrt() the probability, we want to show (original) amplitudes.
  bob_a = math.sqrt(p0.real)
  bob_b = math.sqrt(p1.real)
  print('Teleported (|{:d}{:d}>)   a={:.2f}, b={:.2f}'.format(
      int(qubit0), int(qubit1), bob_a, bob_b))

  if (not math.isclose(expect0, bob_a, abs_tol=1e-6) or
      not math.isclose(expect1, bob_b, abs_tol=1e-6)):
    raise AssertionError('Invalid result.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Step 1: Alice and Bob share an entangled pair, and separate.
  psi = bell.bell_state(0, 0)

  # Step 2: Alice wants to teleport a qubit |x> to Bob,
  #         which is in the state:
  #         |x> = a|0> + b|1> (with a^2 + b^2 == 1)
  a = 0.6
  b = math.sqrt(1.0 - a * a)
  x = state.qubit(a, b)
  print('Quantum Teleportation')
  print('Start with EPR Pair a={:.2f}, b={:.2f}'.format(a, b))

  # Produce combined state.
  alice = x * psi

  # Alice lets the 1st qubit interact with the 2nd qubit, which is her
  # part of the entangle state with Bob.
  alice = ops.Cnot(0, 1)(alice)

  # Now she applies a Hadamard to qubit 0. Bob still owns qubit 2.
  alice = ops.Hadamard()(alice, idx=0)

  # Alices measures and communicates the result (|00>, |01>, ...) to Bob.
  alice_measures(alice, a, b, 0, 0)
  alice_measures(alice, a, b, 0, 1)
  alice_measures(alice, a, b, 1, 0)
  alice_measures(alice, a, b, 1, 1)

if __name__ == '__main__':
  app.run(main)
