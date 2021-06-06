# python3
# pylint: disable=invalid-name

"""class qc (quantum circuit) represents state and operators."""

from __future__ import annotations

import random
from typing import Callable

from absl import flags
import numpy as np
from src.lib import dumpers
from src.lib import ir
from src.lib import ops
from src.lib import optimizer
from src.lib import state
from src.lib import tensor

# Configure: This line might have to change, depending on
#            the current build environment.
#
# Google internal:
# import xgates
#
# GitHub Linux:
import libxgates as xgates

flags.DEFINE_string('libq', '', 'Generate libq output file, or empty')
flags.DEFINE_string('qasm', '', 'Generate qasm output file, or empty')
flags.DEFINE_string('cirq', '', 'Generate cirq output file, or empty')
flags.DEFINE_string('latex', '', 'Generate Latex output file, or empty')


class qc:
  """Wrapper class to maintain state + operators."""

  def __init__(self, name=None, eager: bool = True):
    self.name = name
    self.psi = 1.0
    self.ir = ir.Ir()
    self.build_ir = True
    self.eager = eager
    self.global_reg = 0

  class scope:
    """Scope object to allow grouping of gates in the output."""

    def __init__(self, ir_param, desc: str):
      self.ir = ir_param
      self.desc = desc

    def __enter__(self):
      self.ir.section(self.desc)

    def __exit__(self, t, value, traceback):
      self.ir.end_section()

  # --- States ----------------------------------------------------
  def reg(self, size: int, it=0, *, name: str = None) -> state.Reg:
    ret = state.Reg(size, it, self.global_reg)
    self.global_reg = self.global_reg + size
    self.psi = self.psi * ret.psi()
    self.ir.reg(size, name, ret)
    return ret

  def qubit(self,
            alpha: np.complexfloating = None,
            beta: np.complexfloating = None) -> None:
    self.psi = self.psi * state.qubit(alpha, beta)
    self.global_reg = self.global_reg + 1

  def zeros(self, n: int) -> None:
    self.psi = self.psi * state.zeros(n)
    self.global_reg = self.global_reg + n

  def ones(self, n: int) -> None:
    self.psi = self.psi * state.ones(n)
    self.global_reg = self.global_reg + n

  def bitstring(self, *bits) -> None:
    self.psi = self.psi * state.bitstring(*bits)
    self.global_reg = self.global_reg + len(bits)

  def arange(self, n: int) -> None:
    self.zeros(n)
    for i in range(0, 2**n):
      self.psi[i] = float(i)
    self.global_reg = self.global_reg + n

  def rand(self, n: int) -> None:
    self.psi = self.psi * state.rand(n)
    self.global_reg = self.global_reg + n

  def stats(self) -> str:
    return ('Circuit Statistics\n' +
            '  Qubits: {}\n'.format(self.nbits) +
            '  Gates : {}\n'.format(self.ir.ngates))

  def dump_with_dumper(self, flag: bool,
                       dumper_func: Callable) -> None:
    if flag:
      result = dumper_func(self.ir)
      with open(flag, 'w') as f:
        print(result, file=f)

  def dump_to_file(self) -> None:
    self.dump_with_dumper(flags.FLAGS.libq, dumpers.libq)
    self.dump_with_dumper(flags.FLAGS.qasm, dumpers.qasm)
    self.dump_with_dumper(flags.FLAGS.cirq, dumpers.cirq)
    self.dump_with_dumper(flags.FLAGS.latex, dumpers.latex)

  def optimize(self):
    self. ir = optimizer.optimize(self.ir)

  @property
  def nbits(self) -> int:
    return self.psi.nbits

  def ctl_by_0(self, ctl):
    ctl_qubit = ctl
    ctl_by_0 = False
    if not isinstance(ctl, int):
      ctl_qubit = ctl[0]
      ctl_by_0 = True
    return ctl_qubit, ctl_by_0

  # --- Gates  ----------------------------------------------------
  def apply1(self, gate: ops.Operator, idx: int,
             name: str  = None, *, val: float = None):
    """Apply single gates."""

    if isinstance(idx, state.Reg):
      for reg in range(idx.nbits):
        if self.build_ir:
          self.ir.single(name, idx[reg], gate, val)
        if self.eager:
          xgates.apply1(self.psi, gate.reshape(4), self.psi.nbits, idx[reg],
                        tensor.tensor_width)
      return
    if self.build_ir:
      self.ir.single(name, idx, gate, val)
    if self.eager:
      xgates.apply1(self.psi, gate.reshape(4), self.psi.nbits, idx,
                    tensor.tensor_width)

  def applyc(self, gate: ops.Operator, ctl: int, idx: int,
             name: str = None, *, val: float = None):
    """Apply controlled gates."""

    if isinstance(idx, state.Reg):
      raise AssertionError('controlled register not supported')

    ctl_qubit, by_0 = self.ctl_by_0(ctl)
    if by_0:
      self.x(ctl_qubit)
    if self.build_ir:
      self.ir.controlled(name, ctl_qubit, idx, gate, val)
    if self.eager:
      xgates.applyc(self.psi, gate.reshape(4), self.psi.nbits, ctl_qubit, idx,
                    tensor.tensor_width)
    if by_0:
      self.x(ctl_qubit)

  def cv(self, idx0: int, idx1: int):
    self.applyc(ops.Vgate(), idx0, idx1, 'cv')

  def cv_adj(self, idx0: int, idx1: int):
    self.applyc(ops.Vgate().adjoint(), idx0, idx1, 'cv_adj')

  def cx0(self, idx0: int, idx1: int):
    self.applyc(ops.PauliX(), idx0, idx1, 'cx')

  def cx(self, idx0: int, idx1: int):
    self.applyc(ops.PauliX(), idx0, idx1, 'cx')

  def cy(self, idx0: int, idx1: int):
    self.applyc(ops.PauliY(), idx0, idx1, 'cy')

  def cz(self, idx0: int, idx1: int):
    self.applyc(ops.PauliZ(), idx0, idx1, 'cz')

  def cu1(self, idx0: int, idx1: int, value):
    self.applyc(ops.U1(value), idx0, idx1, 'cu1', val=value)

  def crk(self, idx0: int, idx1: int, value):
    self.applyc(ops.Rk(value), idx0, idx1, 'crk', val=value)

  def ccx(self, idx0: int, idx1: int, idx2: int):
    """Sleator-Weinfurter Construction."""

    i0, c0_by_0 = self.ctl_by_0(idx0)
    i1, c1_by_0 = self.ctl_by_0(idx1)
    i2 = idx2

    with self.scope(self.ir, f'ccx({idx0}, {idx1}, {idx2})'):
      if c0_by_0:
        self.x(i0)
      if c1_by_0:
        self.x(i1)

      self.cv(i0, i2)
      self.cx(i0, i1)
      self.cv_adj(i1, i2)
      self.cx(i0, i1)
      self.cv(i1, i2)

      if c0_by_0:
        self.x(i0)
      if c1_by_0:
        self.x(i1)

  def toffoli(self, idx0: int, idx1: int, idx2: int):
    self.ccx(idx0, idx1, idx2)

  def h(self, idx: int):
    self.apply1(ops.Hadamard(), idx, 'h')

  def s(self, idx: int):
    self.apply1(ops.Sgate(), idx, 's')

  def sdag(self, idx: int):
    self.apply1(ops.Sgate().adjoint(), idx, 'sdag')

  def t(self, idx: int):
    self.apply1(ops.Tgate(), idx, 't')

  def u1(self, idx: int, val):
    self.apply1(ops.U1(val), idx, 'u1', val=val)

  def v(self, idx: int):
    self.apply1(ops.Vgate(), idx, 'v')

  def x(self, idx: int):
    self.apply1(ops.PauliX(), idx, 'x')

  def y(self, idx: int):
    self.apply1(ops.PauliY(), idx, 'y')

  def z(self, idx: int):
    self.apply1(ops.PauliZ(), idx, 'z')

  def yroot(self, idx: int):
    self.apply1(ops.Yroot(), idx, 'yroot')

  def rx(self, idx: int, theta: float):
    self.apply1(ops.RotationX(theta), idx, 'rx', val=theta)

  def ry(self, idx: int, theta: float):
    self.apply1(ops.RotationY(theta), idx, 'ry', val=theta)

  def rz(self, idx: int, theta: float):
    self.apply1(ops.RotationZ(theta), idx, 'rz', val=theta)

#  Appplying a random unitary is possible, but it is not a
#  1- or 2-qubit gate, hence slow.
#  Do not use (unless really unavoidable)
#
#    def unitary(self, op, idx):
#      self.psi = ops.Operator(op)(self.psi, idx)

# --- Measure ----------------------------------------------------
  def measure_bit(self, idx: int, tostate: int = 0,
                  collapse: bool = True) -> (float, state.State):
    return ops.Measure(self.psi, idx, tostate, collapse)

  def pauli_expectation(self, idx: int):
    """We can compute the Pauli expectation value from probabilities."""

    # Pauli eigenvalues are -1 and +1, hence we can compute the
    # expectation value like this:
    p0, _ = self.measure_bit(idx, 0, False)
    return p0 - (1 - p0)

  def sample_state(self, prob_state0: float):
    if prob_state0 < random.random():
      return 1
    return 0

# --- Advanced ---------------------------------------------------
  def swap(self, idx0: int, idx1: int):
    """Simple Swap operation."""

    # pylint: disable=arguments-out-of-order
    with self.scope(self.ir, f'swap({idx0}, {idx1})'):
      self.cx(idx1, idx0)
      self.cx(idx0, idx1)
      self.cx(idx1, idx0)

  def cswap(self, ctl, idx0, idx1):
    """Controlled Swap."""

    with self.scope(self.ir, f'cswap({ctl}, {idx0}, {idx1})'):
      self.ccx(ctl, idx1, idx0)
      self.ccx(ctl, idx0, idx1)
      self.ccx(ctl, idx1, idx0)

  def multi_control(self, ctl, idx1, aux, gate, desc: str):
    """Multi-controlled gate, using aux as ancilla."""

    # This is a simpler version that requires n-1 ancillaries, instead
    # of n-2. The benefit is that the gate can be used as a
    # single-controlled gate, which means we don't need to take the
    # root (no need to include scipy). This construction also makes
    # the controlled-by-0 gates a little bit easier, those controllers
    # are being passed as single-element lists, eg.:
    #   ctl = [1, 2, [3], [4], 5]
    #
    # This can be optimized (later) to turn into a space-optimized
    # n-2 version.
    #
    # We also generalize to the case where ctl is empty or only has 1
    # control qubit. This is very flexible and practically any gate
    # could be expressed this way. This would make bulk control of
    # whole gate sequences straight-forward, but changes the trivial
    # IR we're working with here. Something to keep in mind.

    with self.scope(self.ir, f'multi({ctl}, {idx1}) # {desc})'):
      if not ctl:
        self.apply1(gate, idx1, desc)
        return
      if len(ctl) == 1:
        self.applyc(gate, ctl[0], idx1, desc)
        return

      # Compute the predicate.
      self.ccx(ctl[0], ctl[1], aux[0])
      aux_idx = 0
      for i in range(2, len(ctl)):
        self.ccx(ctl[i], aux[aux_idx], aux[aux_idx+1])
        aux_idx = aux_idx + 1

      # Use predicate to single-control qubit at idx1.
      self.applyc(gate, aux[aux_idx], idx1, desc)

      # Uncompute predicate.
      aux_idx = aux_idx - 1
      for i in range(len(ctl)-1, 1, -1):
        self.ccx(ctl[i], aux[aux_idx], aux[aux_idx+1])
        aux_idx = aux_idx - 1
      self.ccx(ctl[0], ctl[1], aux[0])

  def flip(self, reg: state.Reg):
    """Flip a quantum register via swaps."""

    for idx in range(reg[0], reg[0] + reg.nbits // 2):
      self.swap(idx, reg[0] + reg.nbits - idx - 1)

  def qft_rk(self, reg, swap: bool = True):
    """Apply Qft with Rk gates directly."""

    nbits = reg.nbits
    for idx in range(reg[0], reg[0] + nbits):
      # Each qubit first gets a Hadamard.
      self.h(idx)

      # Each qubit now gets a sequence of Rk(2), Rk(3), ..., Rk(nbits)
      # controlled by qubit (1, 2, ..., nbits-1).
      for rk in range(2, nbits - idx + 1):
        controlled_from = idx + rk - 1
        self.crk(controlled_from, idx, rk)

    if swap:
      self.flip(reg)

# --- qc of qc ------------------------------------------
  def qc(self, qc_parm: qc, offset=0):
    """Add another full circuit to this circuit."""

    # Iterate of the new circuit and add the gates one by one,
    # using this circuit's eager mode.
    #
    for gate in qc_parm.ir.gates:
      if gate.is_single():
        self.apply1(gate.gate, gate.idx0+offset, gate.name, val=gate.val)
      if gate.is_ctl():
        self.applyc(gate.gate, gate.ctl+offset, gate.idx1+offset,
                    gate.name, val=gate.val)

  def run(self):
    """Apply gates in this qc, don't rebuild IR."""

    build_ir = self.build_ir
    eager = self.eager
    self.build_ir = False
    self.eager = True
    self.qc(self)
    self.build_ir = build_ir
    self.eager = eager

  def inverse(self):
    """Return, but don't apply, the inverse circuit."""

    # The order of the gates is reversed and the each gates
    # itself becomes its adjoint. After this, a new circuit
    # is returned. Eager mode is False. The expectation
    # is that an inverse circuit inv is constructed and then applied
    # via circuit.qc(inv), at which point it is applied according to the
    # eager mode of the qc circuit. Usage model:
    #
    #    main = circuit.qc('main circuit')
    #    ... add gates, eager or not.
    #
    #    c = circuit.qc('sub circuit', eager=False)
    #    ... add gates to c, not eager.
    #
    #    Now let's add c to main, at which point the gates are applied.
    #      main.qc(c)
    #
    #    Let's construct the inverse (non-Eager) and add to main (eager)
    #    at an offset.
    #      c_inv = c0.inverse()
    #      main.qc(c_inv, offset=3)
    #
    newqc = qc(self.name, eager=False)
    for gate in self.ir.gates[::-1]:
      val = -gate.val if gate.val else None
      if gate.is_single():
        newqc.apply1(gate.gate.adjoint(), gate.idx0, gate.name+'*', val=val)
      if gate.is_ctl():
        newqc.applyc(gate.gate.adjoint(), gate.ctl, gate.idx1,
                     gate.name+'*', val=val)
    return newqc

# --- Debug --------------------------------------------------
  def dump(self):
    """Simple dumper for basic debugging of a circuit."""

    if self.name:
      print(f'Circuit: {self.name}, Nodes: {len(self.ir.gates)}')
    print(self.ir, end='')
    self.psi.dump('Current state')
