# python3
"""class State wraps a tensor as underlying representation."""

import cmath
import math
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
from src.lib import helper
from src.lib import tensor


class State(tensor.Tensor):
  """class State represents single and multi-qubit states."""

  def density(self) -> tensor.Tensor:
    return tensor.Tensor(np.outer(self, self.conj()))

  def adjoint(self):
    return self.conj().transpose()

  def normalize(self):
    """Renormalize the state. Sum of squared amplitudes==1.0."""

    dprod = np.conj(self) @ self
    assert not dprod.is_close(0.0), 'Normalizing to 0-probability state'
    self /= np.sqrt(np.real(dprod))  # modify object in place.
    return self

  def ampl(self, *bits: Tuple[int]) -> np.complexfloating:
    """Return amplitude for state indexed by 'bits'."""

    return self[helper.bits2val(bits)]

  def prob(self, *bits: Tuple[int]) -> float:
    """Return probability for state indexed by 'bits'."""

    amplitude = self.ampl(*bits)
    return np.real(amplitude.conj() * amplitude)

  def phase(self, *bits: Tuple[int]) -> float:
    """Return phase of a state from the complex amplitude."""

    amplitude = self.ampl(*bits)
    return math.degrees(cmath.phase(amplitude))

  def diff(self, psi) -> bool:
    """Print element-wise differences to another state."""

    same = True
    for i in range(len(self)):
      if not cmath.isclose(self[i], psi[i], abs_tol=1e-4):
        same = False
        print(f'State{helper.val2bits(i, self.nbits)} (|{i}>) differs:', end='')
        print(f'{self[i]:+.3f}  {psi[i]:+.3f}')
    return same

  def maxprob(self) -> Tuple[List[float], float]:
    """Find state with highest probability."""

    # This is the as described in the book, which is good
    # for learning:
    # maxbits, maxprob = [], 0.0
    # for bits in helper.bitprod(self.nbits):
    #   cur_prob = self.prob(*bits)
    #   if cur_prob > maxprob:
    #     maxbits, maxprob = bits, cur_prob

    # However, we can do a lot faster. We just iterate
    # over the state vector, find the index of the maximum
    # amplitude with a numpy function, and return the probability
    # and the index as a bitstring.
    idx = np.abs(self).argmax()
    maxprob = np.real(self[idx].conj() * self[idx])
    maxbits = helper.val2bits(idx, self.nbits)
    return maxbits, maxprob

  def apply1(self, gate: np.ndarray, index: int) -> None:
    """Apply single-qubit gate to this state."""

    # To maintain qubit ordering in this infrastructure,
    # index needs to be reversed.
    index = self.nbits - index - 1
    if index < 0:
      print('***Error***: Negative qubit index in apply1().')
      print('             Perhaps using wrongly shaped state?\n')
      sys.exit(1)
    pow_2_index = 1 << index
    g00 = gate[0, 0]
    g01 = gate[0, 1]
    g10 = gate[1, 0]
    g11 = gate[1, 1]
    for g in range(0, 1 << self.nbits, 1 << (index + 1)):
      for i in range(g, g + pow_2_index):
        t1 = g00 * self[i] + g01 * self[i + pow_2_index]
        t2 = g10 * self[i] + g11 * self[i + pow_2_index]
        self[i] = t1
        self[i + pow_2_index] = t2

  def applyc(self, gate: np.ndarray, control: int, target: int) -> None:
    """Apply a controlled 2-qubit gate via explicit indexing."""

    # To maintain qubit ordering in this infrastructure,
    # index needs to be reversed.
    index = self.nbits - target - 1
    if index < 0:
      print('***Error***: Negative qubit index in applyc().')
      print('             Perhaps using wrongly shaped state?\n')
      sys.exit(1)
    control = self.nbits - control - 1
    pow_2_index = 1 << index
    g00 = gate[0, 0]
    g01 = gate[0, 1]
    g10 = gate[1, 0]
    g11 = gate[1, 1]
    for g in range(0, 1 << self.nbits, 1 << (index + 1)):
      idx_base = g * (1 << self.nbits)
      for i in range(g, g + pow_2_index):
        #idx = idx_base + i
        if (idx_base + i) & (1 << control):
          t1 = g00 * self[i] + g01 * self[i + pow_2_index]
          t2 = g10 * self[i] + g11 * self[i + pow_2_index]
          self[i] = t1
          self[i + pow_2_index] = t2

  def dump(self, desc: str = None, prob_only: bool = True) -> None:
    """Dump probabilities for a state, including local qubit state."""

    def state_to_string(bits: Tuple[int]) -> str:
      s = ''.join(str(i) for i in bits)
      dec_digits = int(math.log10(2 ** len(bits))) + 1
      return f'|{s}> (|{int(s, 2):{dec_digits}d}>)'

    if desc:
      print('|', end='')
      for i in range(self.nbits):
        print(i % 10, end='')
      print(f"> '{desc}'")

    state_list: List[str] = []
    for bits in helper.bitprod(self.nbits):
      if prob_only and (self.prob(*bits) < 10e-6):
        continue
      state_list.append(
          '{:s}:  ampl: {:+.2f} prob: {:.2f} Phase: {:5.1f}'.format(
              state_to_string(bits),
              self.ampl(*bits),
              self.prob(*bits),
              self.phase(*bits),
          )
      )
    state_list.sort()
    print(*state_list, sep='\n')


def qubit(alpha: complex = None, beta: complex = None) -> State:
  """Produce a given state for a single qubit."""

  if alpha is None and beta is None:
    raise ValueError('alpha, beta, or both, need to be specified')

  # Note that multiplying a complex conjugate with its non-conjugate
  # is a real number, but we still have to type-cast it to avoid
  # Python warnings (hence the use of np.real()).
  if beta is None:
    beta = np.sqrt(1.0 - np.real(np.conj(alpha) * alpha))
  if alpha is None:
    alpha = np.sqrt(1.0 - np.real(np.conj(beta) * beta))

  if not math.isclose(
      np.real(np.conj(alpha) * alpha) + np.real(np.conj(beta) * beta), 1.0
  ):
    raise ValueError('Qubit probabilities do not sum to 1.')

  return State([alpha, beta])


# The functions zeros() and ones() produce the all-zero or all-one
# computational basis vector for `d` qubits, ie,
#     |000...0> or
#     |111...1>
#
# The result of this tensor product is
#   always [1, 0, 0, ..., 0]^T or [0, 0, 0, ..., 1]^T

def zeros_or_ones(d: int = 1, idx: int = 0) -> State:
  """Produce the all-0/1 basis vector for `d` qubits."""

  assert d > 0, 'Need to specify at least 1 qubit'
  t = np.zeros(2**d, dtype=tensor.tensor_type())
  t[idx] = 1
  return State(t)


def zeros(d: int = 1) -> State:
  """Produce state with 'd' |0>, eg., |0000>."""

  return zeros_or_ones(d, 0)


def ones(d: int = 1) -> State:
  """Produce state with 'd' |1>, eg., |1111>."""

  return zeros_or_ones(d, 2**d - 1)


def plus(d: int = 1) -> State:
  """Product state |+>."""

  return State([1 / np.sqrt(2), 1 / np.sqrt(2)]).kpow(d)


def minus(d: int = 1) -> State:
  """Product state |->."""

  return State([1 / np.sqrt(2), -1 / np.sqrt(2)]).kpow(d)


def plusi(d: int = 1) -> State:
  """Product state |i>."""

  return State([1 / np.sqrt(2), 1j / np.sqrt(2)]).kpow(d)


def minusi(d: int = 1) -> State:
  """Product state |-i>."""

  return State([1 / np.sqrt(2), -1j / np.sqrt(2)]).kpow(d)


def bitstring(*bits) -> State:
  """Produce a state from a given bit sequence, eg., |0101>."""

  arr = np.asarray(bits)
  assert len(arr) > 0, 'Need to specify at least 1 qubit'
  assert ((arr == 1) | (arr == 0)).all(), 'Bits must be 0 or 1'

  t = np.zeros(1 << len(bits), dtype=tensor.tensor_type())
  t[helper.bits2val(bits)] = 1
  return State(t)


def rand_bits(n: int) -> State:
  """Produce random combination of |0> and |1>."""

  bits = [random.randint(0, 1) for _ in range(n)]
  return bitstring(*bits)


class Reg:
  """Simple register class."""

  def __init__(self, size: int, it=0, global_reg: int = 0) -> None:
    self.size = size
    self.global_idx = list(range(global_reg, global_reg + size))
    self.val = [0] * size
    global_reg += size

    if not it:
      return
    if isinstance(it, int):
      it = format(it, '0{}b'.format(size))
    if isinstance(it, (str, tuple, list)):
      for idx, val in enumerate(it):
        if val == '1' or val == 1:
          self.val[idx] = 1

  def __str__(self) -> str:
    s = '|'
    for _, val in enumerate(self.val):
      s += f'{val}'
    return s + '>'

  def __getitem__(self, idx: int) -> int:
    return self.global_idx[idx]

  def __setitem__(self, idx: int, val: int) -> None:
    self.val[idx] = val

  def psi(self) -> State:
    return bitstring(*self.val)

  @property
  def reg(self):
    return self.global_idx

  @property
  def nbits(self) -> int:
    return self.size
