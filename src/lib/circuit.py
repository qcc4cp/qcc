# python3
# pylint: disable=invalid-name

"""class qc (quantum circuit) represents state and operators."""

import random

from absl import flags

from src.lib import dumpers
from src.lib import ir
from src.lib import ops
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

  def __init__(self, name=None, eager=True):
    self.name = name
    self.psi = 1.0
    self.ir = ir.Ir()
    self.eager = eager
    self.global_reg = 0

  class scope:
    """Scope object to allow grouping of gates in the output."""

    def __init__(self, ir_param, desc):
      self.ir = ir_param
      self.desc = desc

    def __enter__(self):
      self.ir.section(self.desc)

    def __exit__(self, t, value, traceback):
      self.ir.end_section()

  # --- States ----------------------------------------------------
  def reg(self, size, it=0, *, name=None):
    ret = state.Reg(size, it, self.global_reg)
    self.global_reg = self.global_reg + size
    self.psi = self.psi * ret.psi()
    self.ir.reg(size, name, ret)
    return ret

  def qubit(self, alpha=None, beta=None):
    self.psi = self.psi * state.qubit(alpha, beta)

  def zeros(self, n):
    self.psi = self.psi * state.zeros(n)

  def ones(self, n):
    self.psi = self.psi * state.ones(n)

  def bitstring(self, *bits):
    self.psi = self.psi * state.bitstring(*bits)

  def arrange(self, n):
    self.zeros(n)
    for i in range(0, 2**n):
      self.psi[i] = float(i)

  def rand(self, n):
    self.psi = self.psi * state.rand(n)

  def stats(self):
    return ('Circuit Statistics\n' +
            '  Qubits: {}\n'.format(self.nbits) +
            '  Gates : {}\n'.format(self.ir.ngates))

  def dump_with_dumper(self, flag, dumper_func):
    if flag:
      result = dumper_func(self.ir)
      with open(flag, 'w') as f:
        print(result, file=f)

  def dump_to_file(self):
    self.dump_with_dumper(flags.FLAGS.libq, dumpers.libq)
    self.dump_with_dumper(flags.FLAGS.qasm, dumpers.qasm)
    self.dump_with_dumper(flags.FLAGS.cirq, dumpers.cirq)
    self.dump_with_dumper(flags.FLAGS.latex, dumpers.latex)

  @property
  def nbits(self):
    return self.psi.nbits

  # --- Gates  ----------------------------------------------------
  def apply1(self, gate, idx, name=None, *, val=None):
    if isinstance(idx, state.Reg):
      for reg in range(idx.nbits):
        self.ir.single(name, idx[reg], gate, val)
        if self.eager:
          xgates.apply1(self.psi, gate.reshape(4), self.psi.nbits, idx[reg],
                        tensor.tensor_width)
      return
    self.ir.single(name, idx, gate, val)
    if self.eager:
      xgates.apply1(self.psi, gate.reshape(4), self.psi.nbits, idx,
                    tensor.tensor_width)

  def apply_controlled(self, gate, ctl, idx, name=None, *, val=None):
    if isinstance(idx, state.Reg):
      raise AssertionError('controlled register not supported')
    self.ir.controlled(name, ctl, idx, gate, val)
    if self.eager:
      xgates.applyc(self.psi, gate.reshape(4), self.psi.nbits, ctl, idx,
                    tensor.tensor_width)

  def cv(self, idx0, idx1):
    self.apply_controlled(ops.Vgate(), idx0, idx1, 'cv')

  def cv_adj(self, idx0, idx1):
    self.apply_controlled(ops.Vgate().adjoint(), idx0, idx1, 'cv_adj')

  def cx(self, idx0, idx1):
    self.apply_controlled(ops.PauliX(), idx0, idx1, 'cx')

  def cy(self, idx0, idx1):
    self.apply_controlled(ops.PauliY(), idx0, idx1, 'cy')

  def cz(self, idx0, idx1):
    self.apply_controlled(ops.PauliZ(), idx0, idx1, 'cz')

  def cu1(self, idx0, idx1, value):
    self.apply_controlled(ops.U1(value), idx0, idx1, 'cu1', val=value)

  def crk(self, idx0, idx1, value):
    self.apply_controlled(ops.Rk(value), idx0, idx1, 'crk', val=value)

  def ccx(self, idx0, idx1, idx2):
    """Sleator-Weinfurter Construction."""

    with self.scope(self.ir, 'ccx'):
      self.cv(idx0, idx2)
      self.cx(idx0, idx1)
      self.cv_adj(idx1, idx2)
      self.cx(idx0, idx1)
      self.cv(idx1, idx2)

  def toffoli(self, idx0, idx1, idx2):
    self.ccx(idx0, idx1, idx2)

  def h(self, idx):
    self.apply1(ops.Hadamard(), idx, 'h')

  def t(self, idx):
    self.apply1(ops.Tgate(), idx, 't')

  def u1(self, idx, val):
    self.apply1(ops.U1(val), idx, 'u1', val=val)

  def v(self, idx):
    self.apply1(ops.Vgate(), idx, 'v')

  def x(self, idx):
    self.apply1(ops.PauliX(), idx, 'x')

  def y(self, idx):
    self.apply1(ops.PauliY(), idx, 'y')

  def z(self, idx):
    self.apply1(ops.PauliZ(), idx, 'z')

  def yroot(self, idx):
    self.apply1(ops.Yroot(), idx, 'yroot')

  def rx(self, idx, theta):
    self.apply1(ops.RotationX(theta), idx, 'rx')

  def ry(self, idx, theta):
    self.apply1(ops.RotationY(theta), idx, 'ry')

  def rz(self, idx, theta):
    self.apply1(ops.RotationZ(theta), idx, 'rz')

#  This one is possible, but it is not a 1- or 2-qubit gate, hence
#  super slow. Do not use (unless really unavoidable)
#
#  def unitary(self, op, idx):
#    self.psi = ops.Operator(op)(self.psi, idx, 'u')

# --- Measure ----------------------------------------------------
  def measure_bit(self, idx, tostate=0, collapse=True):
    return ops.Measure(self.psi, idx, tostate, collapse)

  def sample_state(self, prob_state0):
    if prob_state0 < random.random():
      return 1
    return 0

# --- Advanced ---------------------------------------------------
  def swap(self, idx0, idx1):
    # pylint: disable=arguments-out-of-order
    with self.scope(self.ir, 'swap'):
      self.cx(idx1, idx0)
      self.cx(idx0, idx1)
      self.cx(idx1, idx0)

  def cswap(self, ctl, idx0, idx1):
    with self.scope(self.ir, 'cswap'):
      self.ccx(ctl, idx1, idx0)
      self.ccx(ctl, idx0, idx1)
      self.ccx(ctl, idx1, idx0)

  def qft_rk(self, reg, swap=True):
    """Apply Qft with Rk gates directly."""

    nbits = reg.nbits
    for idx in range(reg[0], reg[0] + nbits):
      # Each qubit first gets a Hadamard
      self.had(idx)

      # Each qubit now gets a sequence of Rk(2), Rk(3), ..., Rk(nbits)
      # controlled by qubit (1, 2, ..., nbits-1).
      for rk in range(2, nbits - idx + 1):
        controlled_from = idx + rk - 1
        self.crk(controlled_from, idx, rk)

    # Now the qubits need to change their order.
    if swap:
      for idx in range(reg[0], reg[0] + nbits // 2):
        self.swap(idx, reg[0] + nbits - idx - 1)

# --- qc's of qc's ------------------------------------------
  def qc(self, qc, offset=0):
    """Add another full circuit to this circuit."""

    # Iterate of the new circuit and add the gates one by one,
    # inheriting this circuit's eager mode.
    #
    for gate in qc.ir.gates:
      if gate.is_single():
        self.apply1(gate.gate, gate.idx0+offset, gate.name, val=gate.val)
      if gate.is_ctl():
        self.apply_controlled(gate.gate, gate.ctl+offset, gate.idx1+offset,
                              gate.name, val=gate.val)

  def inverse(self):
    """Return, but don't apply, an inverted circuit."""

    # The order of the gates is reversed and the each gates
    # itself becomes its adjoint. After this, the new circuit
    # is returned. The eager mode is False. The expectation
    # is that an inverse circuit inv is constructed and then applied
    # via qc.qc(inv), at which point it is applied according to the
    # eager mode of the qc circuit.
    #
    newqc = qc(self.name, eager=False)
    inv = self.ir.gates.copy()
    inv.reverse()
    for gate in inv:
      if gate.is_single():
        newqc.apply1(gate.gate.adjoint(), gate.idx0, gate.name+'^-1', val=gate.val)
      if gate.is_ctl():
        newqc.apply_controlled(gate.gate.adjoint(), gate.ctl, gate.idx1,
                              gate.name+'^-1', val=gate.val)
    return newqc


# --- Debug --------------------------------------------------
  def dump(self, desc=None, gates=None):
    """Simple dumper for basic debugging of a circuit."""

    if desc:
      print(desc)
    for gate in self.ir.gates:
      if gate.is_single():
        print('  {}({})'.format(gate.name, gate.idx0))
        if gates:
          gate.gate.dump()
      if gate.is_ctl():
        print('  {}({}, {})'.format(gate.name, gate.ctl, gate.idx1))
        if gates:
          gate.gate.dump()
