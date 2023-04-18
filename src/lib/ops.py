# python3
# pylint: disable=invalid-name

"""class Operator represents unitary operators for multi-qubit systems."""

from __future__ import annotations

import cmath
import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from src.lib import helper
from src.lib import state
from src.lib import tensor


class Operator(tensor.Tensor):
  """Operators are represented by square, unitary matrices."""

  def __repr__(self) -> str:
    s = 'Operator('
    s += super().__str__().replace('\n', '\n' + ' ' * len(s))
    s += ')'
    return s

  def __str__(self) -> str:
    s = f'Operator for {self.nbits}-qubit state space.'
    s += ' Tensor:\n'
    s += super().__str__()
    return s

  def dump(
      self, description: Optional[str] = None, zeros: bool = False
  ) -> None:
    res = ''
    if description:
      res += f'{description} ({self.nbits}-qubits operator)\n'
    for row in range(self.shape[0]):
      for col in range(self.shape[1]):
        val = self[row, col]
        res += f'{val.real:+.1f}{val.imag:+.1f}j  '
      res += '\n'
    if not zeros:
      res = res.replace('+0.0j', '    ')
      res = res.replace('-0.0j', '    ')
      res = res.replace('+0.0', ' -  ')
      res = res.replace('-0.0', ' -  ')
      res = res.replace('+', ' ')
    print(res)

  def adjoint(self) -> Operator:
    return Operator(np.conj(self.transpose()))

  # Operators operate on a state via function invocation, eg:
  #    Hadamard()(psi)
  #
  # On State:
  # ------------
  # If idx != 0 the operator is expanded to the size of the state the
  # following way:
  #    First create Identity ops up to idx
  #    Then tensor in the n-bits operator itself
  #    The finish up by tensoring Identities until the size of the
  #    operator matches the size of the state.
  #
  # Once the operator has been constructed a simple matmul does the
  # application and produces a new state.
  #
  # On Operator:
  # -------------
  # Op(op) equals Op @ op equal matmul(Op, op), to produce a new state.
  #
  # Note: Note that in the book the return type was marked as
  #       -> state.State, which is incorrect. (Un-)fortunately, the
  #       type annotations are being ignored by the runtime.
  #
  def apply(
      self, arg: Union[state.State, Operator], idx: int
  ) -> Union[state.State, Operator]:
    """Apply operator to a state or operator."""

    if isinstance(arg, Operator):
      arg_bits = arg.nbits
      if idx > 0:
        arg = Identity().kpow(idx) * arg
      if self.nbits > arg.nbits:
        arg = arg * Identity().kpow(self.nbits - idx - arg_bits)

      if self.nbits != arg.nbits:
        raise AssertionError('Operator(O) with mis-matched dimensions.')

      # Note: We reverse the order in this matmul. So:
      #   x(y) == y @ x
      #
      # This is to mirror that for a circuit like this:
      #   --- X --- Y --- psi
      #
      # Incrementally updating states we would write:
      #   psi = X(psi)
      #   psi = Y(psi)
      #
      # But in a combined operator matrix, Y comes first:
      #   (YX)(psi)
      #
      # The function call should mirror this semantic, since parameters
      # are typically evaluated first (and this mirrors the left to
      # right in the circuit notation):
      #   X(Y) = YX
      #
      return arg @ self

    if not isinstance(arg, state.State):
      raise AssertionError('Invalid parameter, expected State.')

    op = self
    if idx > 0:
      op = Identity().kpow(idx) * op
    if arg.nbits - idx - self.nbits > 0:
      op = op * Identity().kpow(arg.nbits - idx - self.nbits)

    return state.State(np.matmul(op, arg))

  def __call__(
      self, arg: Union[state.State, Operator], idx=0
  ) -> Union[state.State, Operator]:
    return self.apply(arg, idx)


# --------------------------------------------------------------
# Single Qubit Gates / Generators.
# --------------------------------------------------------------
def Identity(d: int = 1) -> Operator:
  return Operator([[1.0, 0.0], [0.0, 1.0]]).kpow(d)


def PauliX(d: int = 1) -> Operator:
  return Operator([[0.0, 1.0], [1.0, 0.0]]).kpow(d)


def PauliY(d: int = 1) -> Operator:
  return Operator([[0.0, -1.0j], [1.0j, 0.0]]).kpow(d)


def PauliZ(d: int = 1) -> Operator:
  return Operator([[1.0, 0.0], [0.0, -1.0]]).kpow(d)


def Pauli(d: int = 1) -> Tuple[Operator, Operator, Operator, Operator]:
  return Identity(d), PauliX(d), PauliY(d), PauliZ(d)


def Hadamard(d: int = 1) -> Operator:
  return Operator(1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]])).kpow(d)


# Phase gate, also called S or Z90. Rotate by 90 deg around z-axis.
def Phase(d: int = 1) -> Operator:
  return Operator([[1.0, 0.0], [0.0, 1.0j]]).kpow(d)


# Phase gate is also called S-gate.
def Sgate(d: int = 1) -> Operator:
  return Phase(d)


# T-gate, which is sqrt(S).
def Tgate(d: int = 1) -> Operator:
  return Operator([[1.0, 0.0], [0.0, cmath.exp(cmath.pi * 1j / 4)]]).kpow(d)


# V-gate, which is sqrt(X). Note that there are more roots:
#   https://quantumcomputing.stackexchange.com/q/15381/11582
def Vgate(d: int = 1) -> Operator:
  return Operator(0.5 * np.array([(1 + 1j, 1 - 1j), (1 - 1j, 1 + 1j)])).kpow(d)


# Yroot is sqrt(Y).
def Yroot(d: int = 1) -> Operator:
  """As found in: https://arxiv.org/pdf/quant-ph/0511250.pdf."""

  return Operator(0.5 * np.array([(1 + 1j, -1 - 1j), (1 + 1j, 1 + 1j)])).kpow(d)


# IBM's U1-gate.
def U1(lam: float, d: int = 1) -> Operator:
  return Operator([(1.0, 0.0), (0.0, cmath.exp(1j * lam))]).kpow(d)


# IBM's general U3-gate.
def U3(theta: float, phi: float, lam: float, d: int = 1) -> Operator:
  return Operator(
      [(np.cos(theta / 2),
        -cmath.exp(1j * lam)*np.sin(theta / 2)),
       (cmath.exp(1j * phi)*np.sin(theta / 2),
        cmath.exp(1j * (phi + lam))*np.cos(theta / 2))]).kpow(d)


def Rk(k: int, d: int = 1) -> Operator:
  return U1(2 * math.pi / (2**k)).kpow(d)


# Cache Pauli matrices for performance reasons.
_PAULI_X = PauliX()
_PAULI_Y = PauliY()
_PAULI_Z = PauliZ()


# Make a single-qubit rotation operator.
# This is a simple implementation of the mechanism outlined here:
# http://www.vcpc.univie.ac.at/~ian/hotlist/qc/talks/bloch-sphere-rotations.pdf
#        (page 22)
def Rotation(vparm: List[float], theta: float) -> Operator:
  """Produce the single-qubit rotation operator."""

  v = np.asarray(vparm)
  if v.shape != (3,) or not math.isclose(v @ v, 1) or not np.all(np.isreal(v)):
    raise ValueError('Rotation vector v must be a 3D real unit vector.')

  return Operator(np.cos(theta / 2) * Identity() - 1j * np.sin(theta / 2) * (
      v[0] * _PAULI_X + v[1] * _PAULI_Y + v[2] * _PAULI_Z))


def RotationX(theta: float) -> Operator:
  return Rotation([1.0, 0.0, 0.0], theta)


def RotationY(theta: float) -> Operator:
  return Rotation([0.0, 1.0, 0.0], theta)


def RotationZ(theta: float) -> Operator:
  return Rotation([0.0, 0.0, 1.0], theta)


def ZeroProjector(nbits: int) -> Operator:
  """Return a projector for n qubits of |0>."""

  zero_projector = np.zeros((2**nbits, 2**nbits))
  zero_projector[0, 0] = 1
  return Operator(zero_projector)


def OneProjector(nbits: int) -> Operator:
  """Return a projector for n qubits of |0>."""

  dim = 2**nbits
  zero_projector = np.zeros((dim, dim))
  zero_projector[dim - 1, dim - 1] = 1
  return Operator(zero_projector)


# Note on indices for controlled operators:
#
# The important aspects are direction and difference, not absolute
# values. In that regards, these are equivalen:
#  ControlledU(0, 3, U) == ControlledU(1, 4, U)
#  ControlledU(2, 0, U) == ControlledU(4, 2, U)
# We could have used -3 and +3, but felt this representation was
# more intuitive.
#
# Operator matrices are stored with all intermittent qubits
# (as Identities). When applying an operator, the starting qubit
# index can be specified.
def ControlledU(idx0: int, idx1: int, u: Operator) -> Operator:
  """Control qubit at idx1 via controlling qubit at idx0."""

  if idx0 == idx1:
    raise ValueError('Control and controlled qubit must not be equal.')

  p0 = ZeroProjector(1)
  p1 = OneProjector(1)

  # space between qubits
  ifill = Identity(abs(idx1 - idx0) - 1)
  # 'width' of U in terms of Identity matrices
  ufill = Identity().kpow(u.nbits)

  if idx1 > idx0:
    op = p0 * ifill * ufill + p1 * ifill * u
  else:
    op = ufill * ifill * p0 + u * ifill * p1
  return op


def Cnot(idx0: int = 0, idx1: int = 1) -> Operator:
  """Controlled Not between idx0 and idx1, controlled by |1>."""

  return ControlledU(idx0, idx1, PauliX())


def Cnot0(idx0: int = 0, idx1: int = 1) -> Operator:
  """Controlled Not between idx0 and idx1, controlled by |0>."""

  if idx1 > idx0:
    x2 = PauliX() * Identity(idx1 - idx0)
  else:
    x2 = Identity(idx0 - idx1) * PauliX()
  return x2 @ ControlledU(idx0, idx1, PauliX()) @ x2


# A nice description of how to make swap gates can be found here:
#   https://algassert.com/post/1717 (from fellow Googler Craig Gidney)
#
# pylint: disable=arguments-out-of-order
def Swap(idx0: int = 0, idx1: int = 1) -> Operator:
  """Swap qubits at idx0 and idx1 via combination of Cnot gates."""

  return Cnot(idx1, idx0) @ Cnot(idx0, idx1) @ Cnot(idx1, idx0)


# Make universal Toffoli gate out of 2 controlled Cnot's.
#    idx1 and idx2 define the 'inner' cnot
#    idx0 defines the 'outer' cnot.
#
# For a Toffoli gate to control with qubit 5
# a Cnot from 4 to 1:
#    Toffoli(5, 4, 1)
#
def Toffoli(idx0: int, idx1: int, idx2: int) -> Operator:
  """Make a toffoli gate."""

  cnot = Cnot(idx1, idx2)
  toffoli = ControlledU(idx0, idx1, cnot)
  return toffoli


def OracleUf(nbits: int, f: Callable[[List[int]], int]) -> Operator:
  """Make an n-qubit Oracle for function f (e.g. Deutsch, Grover)."""

  # This Oracle is constructed similar to the implementation in
  # ./deutsch.py, just with an n-bit |x> and a 1-bit |y>
  #
  dim = 2**nbits
  u = np.zeros(dim**2).reshape(dim, dim)
  for row in range(dim):
    bits = helper.val2bits(row, nbits)
    fx = f(bits[0:-1])  # f(x) without the y.
    xor = bits[-1] ^ fx

    new_bits = bits[0:-1]
    new_bits.append(xor)

    # Construct new column (int) from the new bit sequence.
    new_col = helper.bits2val(new_bits)
    u[row][new_col] = 1.0

  op = Operator(u)
  if not op.is_unitary():
    raise AssertionError('Constructed non-unitary operator.')
  return op


# Build the QFT operator. A good explanation can be found here:
# https://en.wikipedia.org/wiki/Quantum_Fourier_transform
#
# |x1>  -> 1/sqrt(2)(|0> + exp(2*pi*i[0.x1x2x3...xn] |1>))
# |x2>  -> 1/sqrt(2)(|0> + exp(2*pi*i[0.x2x3...xn] |1>))
# |x3>  -> 1/sqrt(2)(|0> + exp(2*pi*i[0.x3...xn] |1>))
#
# While the phases change and are fourier transformed in binary
# fractional form, this doesn't really help, because measurement will
# collapse to a random state. So this operator is usually only
# a first step.
#
def Qft(nbits: int, swap: bool = True) -> Operator:
  """Make an n-bit QFT operator."""

  op = Identity(nbits)
  h = Hadamard()

  for idx in range(nbits):
    # Each qubit first gets a Hadamard
    op = op(h, idx)

    # Each qubit now gets a sequence of Rk(2), Rk(3), ..., Rk(nbits)
    # controlled by qubit (1, 2, ..., nbits-1).
    for rk in range(2, nbits - idx + 1):
      controlled_from = idx + rk - 1
      op = op(ControlledU(controlled_from, idx, Rk(rk)), idx)

  # Now the qubits may need to change their order.
  if swap:
    for idx in range(nbits // 2):
      op = op(Swap(idx, nbits - idx - 1), idx)

  if not op.is_unitary():
    raise AssertionError('Constructed non-unitary operator.')
  return op


def PhaseEstimation(op: Operator, psi: state.State,
                    nbits_phase: int, target: int, offset: int = 0):
  """Apply phase estimation."""

  cu = op
  for inv in reversed(range(nbits_phase)):
    psi = ControlledU(inv + offset, target, cu)(psi, inv + offset)
    cu = cu(cu)
  return psi


# Trace out a qubit from a density matrix and return the
# remaining density matrix.
#
def TraceOutSingle(rho: Operator, index: int) -> Operator:
  """Trace out single qubit from density matrix."""

  nbits = int(math.log2(rho.shape[0]))
  if index > nbits:
    raise AssertionError('Error in TraceOutSingle invalid index (>nbits).')

  eye = Identity()
  zero = Operator([1.0, 0.0])
  one = Operator([0.0, 1.0])

  p0 = p1 = tensor.Tensor(1.0)
  for idx in range(nbits):
    if idx == index:
      p0 = p0 * zero
      p1 = p1 * one
    else:
      p0 = p0 * eye
      p1 = p1 * eye

  rho0 = p0 @ rho
  rho0 = rho0 @ p0.transpose()
  rho1 = p1 @ rho
  rho1 = rho1 @ p1.transpose()
  rho_reduced = rho0 + rho1
  return rho_reduced


def TraceOut(rho: Operator, index_set: List[int]) -> Operator:
  """Trace out multiple qubits from density matrix."""

  for idx, val in enumerate(index_set):
    nbits = int(math.log2(rho.shape[0]))
    if val > nbits:
      raise AssertionError('Error TraceOut, invalid index (>nbits).')
    rho = TraceOutSingle(rho, val)

    # Tracing out a bit means that rho is now 1 bit smaller, the
    # indices right to the traced out qubit need to shift left by 1.
    # Example, to trace out bits 2, 4:
    # Before:
    #    qubit 0  1  2  3  4  5
    #          a  b  c  d  e  f
    # Trace out 2:
    #    qubit 0  1 <-  3  4  5
    #    qubit 0  1  2  3  4
    #          a  b  d  e  f
    # Trace out 4 (is now 3)
    #    qubit 0  1  2  <-  4
    #    qubit 0  1  2  3
    #          a  b  d  f
    for i in range(idx + 1, len(index_set)):
      index_set[i] = index_set[i] - 1
  return rho


def Measure(
    psi: state.State, idx: int, tostate: int = 0, collapse: bool = True
) -> Tuple[float, state.State]:
  """Measure a qubit via a projector on the density matrix."""

  # Measure() measure qubit 'idx' in state 'psi'. It both measures the
  # probability of the result being state `tostate` and, if `collapse`
  # is set to true, also collapses the state to `tostate`. It is helpful
  # for debugging to have this forcing function, but care must
  # be taken not to collapse the state to one with 0 probability.

  # Compute probability of qubit(idx) to be in state 0 / 1.
  rho = psi.density()
  op = ZeroProjector(1) if tostate == 0 else OneProjector(1)

  # Construct full matrix to apply to density matrix:
  if idx > 0:
    op = Identity().kpow(idx) * op
  if idx < psi.nbits - 1:
    op = op * Identity().kpow(psi.nbits - idx - 1)

  # Probability is the trace.
  prob0 = np.trace(np.matmul(op, rho))

  # Collapse state and normalize
  if collapse:
    mvmul = np.dot(op, psi)
    divisor = np.real(np.linalg.norm(mvmul))
    if divisor > 1e-10:
      normed = mvmul / divisor
    else:
      raise AssertionError('Measure() collapses to 0.0 probability state.')
    return np.real(prob0), state.State(normed)

  # Return original state to enable chaining.
  return np.real(prob0), psi
