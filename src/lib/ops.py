# python3
# pylint: disable=invalid-name

"""class Operator represents unitary operators for multi-qubit systems."""

import cmath
import math
from typing import Optional

import numpy as np

from src.lib import helper
from src.lib import state
from src.lib import tensor


class Operator(tensor.Tensor):
  """Operators are represented by square, unitary matrices."""

  def __repr__(self):
    s = 'Operator('
    s += super().__str__().replace('\n', '\n' + ' ' * len(s))
    s += ')'
    return s

  def __str__(self):
    s = f'Operator for {self.nbits}-qubit state space.'
    s += ' Tensor:\n'
    s += super().__str__()
    return s

  def dump(self, description: Optional[str]=None, zeros: bool=False) -> None:
    res = ''
    if description:
      res += f'{description} ({self.nbits}-qubits operator)\n'
    for row in range(self.shape[0]):
      for col in range(self.shape[1]):
        val = self[row, col]
        res += f'{val.real:+.1f}{val.imag:+.1f}i  '
      res += '\n'
    if not zeros:
      res = res.replace('+0.0i', '    ')
      res = res.replace('-0.0i', '    ')
      res = res.replace('+0.0', ' -  ')
      res = res.replace('-0.0', ' -  ')
      res = res.replace('+', ' ')
    print(res)

  def adjoint(self):
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
  #    The finish up by tensoring Identities until the size of the operator
  #    matches the size of the state.
  #
  # Once the operator has been constructed a simple matmul does the application
  # and produces a new state.
  #
  # On Operator:
  # -------------
  # Op(op) equals Op @ op equal matmul(Op, op), to produce a new state.
  #
  def _apply(self, arg, idx):
    """Apply operator to a state or operator."""

    if isinstance(arg, Operator):
      arg_bits = arg.nbits
      if idx > 0:
        arg = Identity()**idx * arg
      if self.nbits > arg.nbits:
        arg = arg * Identity()**(self.nbits - idx - arg_bits)

      if self.nbits != arg.nbits:
        raise AssertionError('Operator(Operator) with mis-matched dimensions.')

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
      # are typically evaluated first (and this mirrors the left to right
      # in the pictorial):
      #   X(Y) = YX
      #
      return arg @ self

    if not isinstance(arg, state.State):
      raise AssertionError('Invalid parameter, expected State.')

    op = self
    if idx > 0:
      op = Identity()**idx  * op
    if arg.nbits - idx - self.nbits > 0:
      op = op * Identity()**(arg.nbits - idx - self.nbits)

    return state.State(np.matmul(op, arg))

  def __call__(self, arg, idx=0):
    return self._apply(arg, idx)


#--------------------------------------------------------------
# Single Qubit Gates / Generators.
#--------------------------------------------------------------
def Identity(d=1):
  return Operator(np.array([[1.0, 0.0], [0.0, 1.0]]))**d


def PauliX(d=1):
  return Operator(np.array([[0.0, 1.0], [1.0, 0.0]]))**d


def PauliY(d=1):
  return Operator(np.array([[0.0, -1.0j], [1.0j, 0.0]]))**d


def PauliZ(d=1):
  return Operator(np.array([[1.0, 0.0], [0.0, -1.0]]))**d


def Pauli(d=1):
  return Identity(d), PauliX(d), PauliY(d), PauliZ(d)


def Hadamard(d=1):
  return Operator(
      1 / np.sqrt(2) *
      np.array([[1.0, 1.0], [1.0, -1.0]]))**d


# Phase gate, also called S or Z90. Rotate by 90 deg around z-axis.
def Phase(d=1):
  return Operator(np.array([[1.0, 0.0], [0.0, 1.0j]]))**d


# Phase gate is also called S-gate.
def Sgate(d=1):
  return Phase(d)


# T-gate, which is sqrt(S).
def Tgate(d=1):
  return Operator(np.array([[1.0, 0.0],
                            [0.0, cmath.exp(cmath.pi * 1j / 4)]]))**d


# V-gate, which is sqrt(X)
def Vgate(d=1):
  return Operator(0.5 * np.array([(1+1j, 1-1j), (1-1j, 1+1j)]))**d


# Yroot is sqrt(Y).
def Yroot(d=1):
  """As found in: https://arxiv.org/pdf/quant-ph/0511250.pdf."""

  return Operator(0.5 * np.array([(1+1j, -1-1j), (1+1j, 1+1j)]))**d


# Rk is the rotation gate used in QFT.
def Rk(k):
  return Operator(np.array([(1.0, 0.0),
                            (0.0, cmath.exp(2.0 * cmath.pi * 1j / 2**k))]))


def U1(lam):
  return Operator(np.array([(1.0, 0.0),
                            (0.0, cmath.exp(1j * lam))]))


# Make a single-qubit rotation operator.
# This is a simple implementation of the mechanism outlined here:
# http://www.vcpc.univie.ac.at/~ian/hotlist/qc/talks/bloch-sphere-rotations.pdf
#        (page 22)
def Rotation(v: np.ndarray, theta: float) -> np.ndarray:
  """Produce the single-qubit rotation operator."""

  v = np.asarray(v)
  if v.shape != (3,) or abs(v.dot(v) - 1.0) > 1e-8 or not np.all(np.isreal(v)):
    raise ValueError('Rotation vector v must be a 3D real unit vector.')

  return np.cos(theta / 2) * Identity() - 1j * np.sin(theta / 2) * (
      v[0] * PauliX() + v[1] * PauliY() + v[2] * PauliZ())


def RotationX(theta):
  return Rotation([1., 0., 0.], theta)


def RotationY(theta):
  return Rotation([0., 1., 0.], theta)


def RotationZ(theta):
  return Rotation([0., 0., 1.], theta)


def Projector(psi):
  """Construct projection operator from state by computing outer product."""
  return Operator(psi.density())


# Note on indices for controlled operators:
#
# The important aspects are direction and difference, not absolute values. In
# that regards, Op(0, 3, U) is the same as Op(1, 4, U) and Op(2,0) is the same
# as Op(4, 2). We could have used -3 and +3, but felt this representation was
# more intuitive.
#
# Operator matrices are stored with all intermittend qubits (as Identities).
# When applying an operator, the starting qubit index can be specified.
def ControlledU(idx0, idx1, u):
  """Control qubit at idx1 via controlling qubit at idx0."""

  if idx0 == idx1:
    raise ValueError('Control and controlled qubit must not be equal.')

  p0 = Projector(state.zeros(1))
  p1 = Projector(state.ones(1))
  ifill = Identity(int(math.fabs(idx1 - idx0)) - 1)  # space between qubits
  ufill = Identity()**u.nbits  # 'width' of U in terms of Identity matrices

  if idx1 > idx0:
    if idx1 - idx0 > 1:
      op = p0 * ifill * ufill + p1 * ifill * u
    else:
      op = p0 * ufill + p1 * u
  else:
    if idx0 - idx1 > 1:
      op = ufill * ifill * p0 + u * ifill * p1
    else:
      op = ufill * p0 + u * p1
  return op


def Cnot(idx0=0, idx1=1):
  """Controlled Not between idx0 and idx1, controlled by |1>."""

  return ControlledU(idx0, idx1, PauliX())


def Cnot0(idx0=0, idx1=1):
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
def Swap(idx0=0, idx1=1):
  """Swap qubits at idx0 and idx1 via combination of Cnot gates."""

  return Cnot(idx1, idx0) @ Cnot(idx0, idx1) @ Cnot(idx1, idx0)


# Make universal Toffoli gate out of 2 controlled Cnot's.
#    idx1 and idx2 define the 'inner' cnot
#    idx0 defines the 'outer' cnot.
#
# For a Toffoli gate to control qubit 5 via cnot from 4 and 1:
#    Toffoli(1, 4, 5)
#
def Toffoli(idx0, idx1, idx2):
  """Make a toffoli gate."""

  cnot = Cnot(idx1, idx2)
  toffoli = ControlledU(idx0, idx1, cnot)
  return toffoli


def OracleUf(nbits, f):
  """Make an n-qubit Oracle for function f (eg. Deutsch, Grover)."""

  # This Oracle is constructed similar to the implementation in
  # ./deutsch.py, just with an n-bit |x> and a 1-bit |y>
  #
  dim = 2**nbits
  u = np.zeros(dim**2).reshape(dim, dim)
  for row in range(dim):
    bits = helper.val2bits(row, nbits)
    fx = f(bits[0:-1])   # f(x) without the y.
    xor = bits[-1] ^ fx

    new_bits = bits[0:-1]
    new_bits.append(xor)

    # Construct new column (int) from the new bit sequence.
    new_col = helper.bits2val(new_bits)
    u[row][new_col] = 1.0

  op = Operator(u)
  if not op.is_unitary():
    raise AssertionError('constructed non-unitary operators.')
  return op


# It is possible to construct 1- and 2-qubit oracles from a
# permutation matrix. First step is to compute the permutation.
# This follows the same method as OraceUf, but it does not
# populate a matrix, it only collects the permutations.

def Permutation(nbits, f):
  """Compute a permutation from function f."""

  dim = 2**nbits
  perm=[]
  for row in range(dim):
    bits = helper.val2bits(row, nbits)
    fx = f(bits[0:-1])
    xor = bits[-1] ^ fx
    new_bits = bits[0:-1]
    new_bits.append(xor)

    # Construct new column (int) from the new bit sequence.
    new_col = helper.bits2val(new_bits)
    perm.append(new_col)
  return perm


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
def Qft(nbits):
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

  # Now the qubits need to change their order.
  for idx in range(nbits // 2):
    op = op(Swap(idx, nbits - idx - 1), idx)

  if not op.is_unitary():
    raise AssertionError('constructed non-unitary operator')
  return op


# Trace out a qubit from a density matrix and return the
# remaining density matrix.
#
def TraceOutSingle(rho, index):
  """Trace out single qubit from density matrix."""

  nbits = int(math.log2(rho.shape[0]))
  if index > nbits:
    raise AssertionError('Invalid use of Ptrace, invalid index (>nbits).')

  eye = Identity()
  zero = Operator(np.array([1.0, 0.0]))
  one = Operator(np.array([0.0, 1.0]))

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


def TraceOut(rho, index_set):
  """Trace out multiple qubits from density matrix."""

  for index in range(len(index_set)):
    nbits = int(math.log2(rho.shape[0]))
    if index_set[index] > nbits:
      raise AssertionError('Invalid use of Ptrace, invalid index (>nbits).')
    rho = TraceOutSingle(rho, index_set[index])

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
    for i in range(index+1, len(index_set)):
      index_set[i] = index_set[i] - 1
  return rho


def Measure(psi, idx, tostate=0, collapse=True):
  """Measure a qubit out of a state via a projector on the density matrix."""

  # Measure() measure qubit 'idx' in state 'psi'. It both measures the
  # probability of the result being state `tostate` and, if `collapse`
  # is set to true, also collapses the state to `tostate`. It is helpful
  # for debugging to have this forcing function, but care must
  # be taken not to collapse the state to one with 0 probability.

  # Compute probability of qubit(idx) to be in state 0 / 1
  rho = psi.density()
  if tostate == 0:
    op = Projector(state.zero)
  else:
    op = Projector(state.one)

  # Construct full matrix to apply to density matrix:
  if idx > 0:
    op = Identity()**idx * op
  if idx < psi.nbits - 1:
    op = op * Identity()**(psi.nbits - idx -1)

  # Probability is the trace.
  prob0 = np.trace(np.matmul(op, rho))

  # Collapse state and normalize
  if collapse:
    mvmul = np.dot(op, psi)
    divisor = np.real(np.linalg.norm(mvmul))
    if divisor > 1e-10:
      normed = mvmul / np.real(np.linalg.norm(mvmul))
    else:
      raise AssertionError('Measure() collapses to 0.0 probability state')
    return np.real(prob0), state.State(normed)

  # Return original state to enable chaining.
  return np.real(prob0), psi
