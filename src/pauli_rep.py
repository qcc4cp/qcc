# python3
"""Example: Pauli Representation of single and two-qubit system."""

import math
import random

from absl import app
import numpy as np

from src.lib import circuit
from src.lib import ops
from src.lib import state


def single_qubit():
  """Compute Pauli representation of single qubit."""

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
    new_rho = 0.5 * (
        i * ops.Identity()
        + x * ops.PauliX()
        + y * ops.PauliY()
        + z * ops.PauliZ()
    )
    if not np.allclose(rho, new_rho, atol=1e-06):
      raise AssertionError('Invalid Pauli Representation')

    print(f'qubit({qc.psi[0]:11.2f}, {qc.psi[1]:11.2f}) = ', end='')
    print(f'{i:11.2f} I + {x:11.2f} X + {y:11.2f} Y + {z:11.2f} Z')

    # There is another way to decompose 2x2 matrices in terms of
    # Pauli matrices and rank-one projectors. See:
    #   https://quantumcomputing.stackexchange.com/q/29497/11582
    #
    zero_projector = state.zeros(1).density()
    one_projector = state.ones(1).density()
    plus_projector = state.plus(1).density()
    i_projector = state.plusi(1).density()

    a1 = (i + z) * zero_projector
    a2 = (i - z) * one_projector
    a3 = x * (2 * plus_projector - zero_projector - one_projector)
    a4 = y * (2 * i_projector - zero_projector - one_projector)
    a = 0.5 * (a1 + a2 + a3 + a4)
    if not np.allclose(rho, a, atol=1e-06):
      raise AssertionError('Invalid representation as projectors')


def two_qubit():
  """Compute Pauli representation for two-qubit system."""

  for _ in range(10):
    # First we construct a circuit with two, very random qubits.
    #
    qc = circuit.qc('random qubit')
    qc.qubit(random.random())
    qc.qubit(random.random())

    # Potentially entangle them.
    qc.h(0)
    qc.cx(0, 1)

    # Additionally rotate around randomly.
    for i in range(2):
      qc.rx(i, math.pi * random.random())
      qc.ry(i, math.pi * random.random())
      qc.rz(i, math.pi * random.random())

    # Compute density matrix.
    rho = qc.psi.density()

    # Every rho can be put in the 2-qubit Pauli representation,
    # which is this Sum over i, j from 0 to 3 inclusive, representing
    # the four Pauli matrices (including the Identity):
    #
    #                 3
    #    rho = 1/4 * Sum(c_ij * Pauli_i kron Pauli_j)
    #                i,j=0
    #
    # To compute the various factors c_ij, we multiply the Pauli
    # tensor products with the density matrix and take the trace. This
    # trace is the computed factor:
    #
    paulis = [ops.Identity(), ops.PauliX(), ops.PauliY(), ops.PauliZ()]
    c = np.zeros((4, 4), dtype=np.complex64)
    for i in range(4):
      for j in range(4):
        tprod = paulis[i] * paulis[j]
        c[i][j] = np.trace(rho @ tprod)

    # To test whether the two qubits are entangled, the diagonal factors
    # (without c[0][0]) are added up. If the sum is <= 1.0, the qubit
    # states are still seperable.
    #
    # Note: According to this answer following,
    #       this entanglement test is incorrect (hence commented):
    #       https://quantumcomputing.stackexchange.com/a/26667/11582
    #
    # diag = np.abs(c[1][1]) + np.abs(c[2][2]) + np.abs(c[3][3])
    # print(f'{iteration}: diag: {diag:5.2f} ', end='')
    # if diag > 1.0:
    #   print('--> Entangled')
    # else:
    #   print('Seperable')

    # Let's verify the result and construct a density matrix
    # from the Pauli matrices using the computed factors:
    #
    new_rho = np.zeros((4, 4), dtype=np.complex64)
    for i in range(4):
      for j in range(4):
        tprod = paulis[i] * paulis[j]
        new_rho = new_rho + c[i][j] * tprod

    if not np.allclose(rho, new_rho / 4, atol=1e-5):
      raise AssertionError('Invalid Pauli Representation')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  single_qubit()
  two_qubit()


if __name__ == '__main__':
  app.run(main)
