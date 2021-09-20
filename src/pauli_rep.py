# python3
"""Example: Pauli Representation of single and two-qubit system."""

import math
import random

from absl import app
import numpy as np

from src.lib import circuit
from src.lib import ops


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for i in range(10):
    # First we construct a circuit with just one, very random qubit.
    #
    qc = circuit.qc('random qubit')
    qc.qubit(random.random())
    qc.rx(0, math.pi * random.random())
    qc.ry(0, math.pi * random.random())
    qc.rz(0, math.pi * random.random())

    # Every qubit (rho) can be put in the Pauli Representation,
    # which is this Sum over i from 0 to 3 inclusive, representing
    # the four Pauli matrices (including the Identity):
    #
    #                  3
    #    rho = 1/2 * Sum(c_i * Pauli_i)
    #                 i=0
    #
    # To compute the various factors c_i, we multiply the Pauli
    # matrices with the density matrix and take the trace. This
    # trace is the computed factor:
    #
    rho = qc.psi.density()
    i = np.trace(ops.Identity() @ rho)
    x = np.trace(ops.PauliX() @ rho)
    y = np.trace(ops.PauliY() @ rho)        
    z = np.trace(ops.PauliZ() @ rho)    

    # Let's verify the result and construct a density matrix
    # from the Pauli matrices using the computed factors:
    #
    new_rho = 0.5 * (i * ops.Identity() + x * ops.PauliX() +
                     y * ops.PauliY() + z * ops.PauliZ())
    if (not np.allclose(rho, new_rho)):
      raise AssertionError('Invalid Pauli Representation')

    print(f'qubit({qc.psi[0]:11.2f}, {qc.psi[1]:11.2f}) = ', end='')
    print(f'{i:11.2f} I + {x:11.2f} X + {y:11.2f} Y + {z:11.2f} Z')


if __name__ == '__main__':
  app.run(main)
