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

  def __init__(self, opcode, name, idx0=0, idx1=None, val=None):
    self._opcode = opcode
    self._name = name
    self._idx0 = idx0
    self._idx1 = idx1
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
    return self._idx0

  @property
  def ctl(self):
    return self._idx0

  @property
  def idx1(self):
    return self._idx1

  @property
  def val(self):
    return self._val


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

  def single(self, name, idx0, val=None):
    self.gates.append(Node(Op.SINGLE, name, idx0, None, val))
    self._ngates += 1

  def controlled(self, name, idx0, idx1, val=None):
    self.gates.append(Node(Op.CTL, name, idx0, idx1, val))
    self._ngates += 1

  def section(self, desc):
    self.gates.append(Node(Op.SECTION, desc))

  def end_section(self):
    self.gates.append(Node(Op.END_SECTION, 0))

  @property
  def ngates(self):
    return self._ngates
