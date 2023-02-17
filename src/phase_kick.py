# python3
"""Example: Simple test program: Implementation of phase kick."""

from absl import app
import numpy as np

from src.lib import ops
from src.lib import state

# Simple phase kick.
#
# The top 2 qubits are put in super position. The third qubit
# is initialized to |1>. Two controlled gates are being applied:
#    a controlled S (90 deg) from 0 to 2
#    a controlled T (45 deg) from 1 to 2
#
# The interesting part is that the result is additive, the state
# representing |111> adds the two angles for the first 2 qubits:
#
# |001> (|1>):  ampl:  0.50+0.00j prob: 0.25 Phase:   0.00
# |011> (|3>):  ampl:  0.35+0.35j prob: 0.25 Phase:  45.00
# |101> (|5>):  ampl:  0.00+0.50j prob: 0.25 Phase:  90.00
# |111> (|7>):  ampl: -0.35+0.35j prob: 0.25 Phase: 135.00
#
# This phase kick is what underlies the Quantum Fourier Transform
# as well as quantum arithmetic.


def simple_kick():
  psi = state.bitstring(0, 0, 1)
  psi = ops.Hadamard(2)(psi)
  psi = ops.ControlledU(0, 2, ops.Sgate())(psi)
  psi = ops.ControlledU(1, 2, ops.Tgate())(psi, 1)
  psi.dump()


# Simple form of a phase kick, as it used in Bernstein.
#
# 1) Input is all |0> plus an additional |1>
#
# 2) Hadamard all |0-> inputs to put them in the corresponding
#    |+> basis, the |1> will become |->.
#
# 3) A Cnot to |-> (former |1>) will flip the |+> to a |->
#    This is the key "trick" used in Bernstein.
#
# 4) The final Hadamard will flip back |+> to |0>, but the flipped
#    |-> will become |1> (or -|1>, to be precise)


def basis_kick1():
  """Simple H-Cnot-H phase kick."""

  psi = state.zeros(3) * state.ones(1)
  psi = ops.Hadamard(4)(psi)
  psi = ops.Cnot(2, 3)(psi, 2)
  psi = ops.Hadamard(4)(psi)
  if psi.prob(0, 0, 1, 1) < 0.9:
    raise AssertionError('Something is wrong with the phase kick')


def basis_kick2():
  """Another way to look at this H-Cnot-H phase kick."""

  # This produces the vector [0, 1, 0, 0]
  psi = state.bitstring(0, 1)

  # Applying Hadamard: [0.5, -0.5, 0.5, -0.5]
  h2 = ops.Hadamard(2)
  psi = h2(psi)

  # Acting Cnot on this vector: [0.5, -0.5, -0.5, 0.5]
  psi = ops.Cnot()(psi)

  # Final Hadamard: [0, 0, 0, 1]
  psi = h2(psi)

  # which is |11>
  p11 = state.bitstring(1, 1)
  if not psi.is_close(p11):
    raise AssertionError('Something is wrong with the phase kick')


def basis_changes():
  """Explore basis changes via Hadamard."""

  # Generate [1, 0]
  psi = state.zeros(1)

  # Hadamard will result in 1/sqrt(2) [1, 1] (|+>)
  psi = ops.Hadamard()(psi)

  # Generate [0, 1]
  psi = state.ones(1)

  # Hadamard on |1> will result in 1/sqrt(2) [1, -1] (|->)
  psi = ops.Hadamard()(psi)

  # Simple PauliX will result in 1/sqrt(2) [-1, 1]
  psi = ops.PauliX()(psi)

  # But back to computational, will result in -|1>.
  # Global phases can be ignored.
  psi = ops.Hadamard()(psi)
  if not np.allclose(psi[1], -1.0):
    raise AssertionError('Invalid basis change.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  simple_kick()
  basis_changes()
  basis_kick1()
  basis_kick2()


if __name__ == '__main__':
  app.run(main)
