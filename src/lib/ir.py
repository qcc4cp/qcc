# python3
# pylint: disable=invalid-name

"""Compiler IR."""

import enum


class Op(enum.Enum):
  UNK = 0
  SINGLE = 1
  CTL = 2
  SECTION = 3
  END_SECTION = 4


class Node:
  """Single node in the IR."""

  def __init__(self, opcode, name, idx0, idx1, gate, val):
    self._opcode = opcode
    self._name = name
    self._idx0 = idx0
    self._idx1 = idx1
    self._gate = gate
    self._val = val

  def is_single(self):
    return self._opcode == Op.SINGLE

  def is_ctl(self):
    return self._opcode == Op.CTL

  def is_gate(self):
    return self.is_single() or self.is_ctl()

  def is_section(self):
    return self._opcode == Op.SECTION

  def is_end_section(self):
    return self._opcode == Op.END_SECTION

  @property
  def opcode(self):
    return self._opcode

  @property
  def name(self):
    if not self._name:
      return '*unk*'
    return self._name

  @property
  def desc(self):
    return self._name

  @property
  def idx0(self):
    if not self.is_single():
      raise AssertionError('Invalid use of idx0(), must be single gate.')
    return self._idx0

  @property
  def ctl(self):
    if not self.is_ctl():
      raise AssertionError('Invalid use of ctl(), must be controlled gate.')
    return self._idx0

  @property
  def idx1(self):
    if not self.is_ctl():
      raise AssertionError('Invalid use of idx1(), must be controlled gate.')
    return self._idx1

  @property
  def val(self):
    return self._val

  @property
  def gate(self):
    return self._gate


class Ir:
  """Compiler IR."""

  def __init__(self):
    self._ngates = 0
    self.gates = []
    self.regs = []
    self.nregs = 0
    self.regset = []

  def reg(self, size, name, register):
    self.regset.append((name, size, register))
    for i in range(size):
      self.regs.append((self.nregs + i, name, i))
    self.nregs += size

  def single(self, name, idx0, gate, val=None):
    self.gates.append(Node(Op.SINGLE, name, idx0, None, gate, val))
    self._ngates += 1

  def controlled(self, name, idx0, idx1, gate, val=None):
    self.gates.append(Node(Op.CTL, name, idx0, idx1, gate, val))
    self._ngates += 1

  def section(self, desc):
    self.gates.append(Node(Op.SECTION, desc, 0, 0, None, None))

  def end_section(self):
    self.gates.append(Node(Op.END_SECTION, 0, 0, 0, None, None))

  @property
  def ngates(self):
    return self._ngates
