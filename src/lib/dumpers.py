# python3
# pylint: disable=invalid-name

"""Various output formats for the compiler IR."""

import cmath
import itertools
import math
import re
from typing import List

import numpy as np

from src.lib import helper


def reg2str(ir, idx):
  """Convert absolute register index to register-based string."""

  for r in ir.regs:
    if r[0] == idx:
      return f'{r[1]}[{r[2]}]'
  return '???'


def _wires(ir) -> int:
  """Number of wires: those of the registers, and those only used by gates."""

  n = ir.nregs
  for op in ir.gates:
    if op.is_single():
      n = max(n, op.idx0 + 1)
    elif op.is_ctl():
      n = max(n, op.ctl + 1, op.idx1 + 1)
  return n


def _initial_ones(ir):
  """Yield the qubits that the registers initialize to |1>."""

  idx = 0
  for _, _, reg in ir.regset:
    for val in reg.val:
      if val == 1:
        yield idx
      idx += 1


# --- qasm dumper --------------------------------------------------
#
# The output uses nothing but the gates of "qelib1.inc", the standard
# library of OpenQASM 2.0. Gates with other names, such as the square root
# gates, controlled phase gates, or the adjoints and sub-gates that
# circuits create, are expressed by these gates.
_QASM_RESERVED = frozenset([
    'qreg', 'creg', 'gate', 'opaque', 'include', 'measure', 'barrier',
    'reset', 'if', 'pi', 'u', 'cx', 'sin', 'cos', 'tan', 'exp', 'ln', 'sqrt',
])
_QASM_SINGLE = {
    'h': 'h', 'x': 'x', 'y': 'y', 'z': 'z', 's': 's', 't': 't',
    'sdag': 'sdg', 'tdag': 'tdg', 'hdag': 'h', 'xdag': 'x', 'ydag': 'y',
    'zdag': 'z',
}
_QASM_CONTROLLED = {
    'cx': 'cx', 'cy': 'cy', 'cz': 'cz', 'ch': 'ch',
    'cxdag': 'cx', 'cydag': 'cy', 'czdag': 'cz', 'chdag': 'ch',
}
_QASM_PARAMETER = ('u1', 'rx', 'ry', 'rz')
_QASM_CONTROLLED_PARAMETER = ('cu1', 'crz')
_EPS = 1e-12


def _qasm_num(x: float) -> str:
  """Format an angle as a fraction of pi if it is one, else as a number."""

  # The gates are computed in single precision, hence the tolerance.
  x = float(x)
  for num, denom in itertools.product(range(1, 4), range(1, 129)):
    if math.isclose(abs(x), num * math.pi / denom, rel_tol=1e-6):
      return (('-' if x < 0 else '') + (f'{num}*' if num > 1 else '') + 'pi'
              + (f'/{denom}' if denom > 1 else ''))
  return f'{x:.12g}'


def _qasm_wires(ir):
  """Return the register declarations, and the qasm name of every wire."""

  decls = []
  used = set()

  def unique(name: str) -> str:
    # Names in qasm start with a lower case letter and are not keywords.
    name = re.sub(r'\W', '_', name)
    if not re.match('[a-z]', name) or name in _QASM_RESERVED:
      name = 'r_' + name
    while name in used:
      name += '_'
    used.add(name)
    return name

  wires = []
  for idx, (name, size, _) in enumerate(ir.regset):
    name = unique(str(name) if name else f'q{idx}')
    decls.append((name, size))
    wires += [f'{name}[{i}]' for i in range(size)]

  # Qubits that no register has allocated, e.g., with bitstring().
  extra = _wires(ir) - len(wires)
  if extra > 0:
    name = unique('anon')
    decls.append((name, extra))
    wires += [f'{name}[{i}]' for i in range(extra)]
  return decls, wires


def _zyz(u):
  """Decompose u = exp(i alpha) Rz(beta) Ry(gamma) Rz(delta)."""

  u = np.asarray(u, dtype=complex).reshape(2, 2)
  alpha = cmath.phase(np.linalg.det(u)) / 2
  v = u * cmath.exp(-1j * alpha)       # In SU(2).
  gamma = 2 * math.atan2(abs(v[1, 0]), abs(v[0, 0]))
  total = 2 * cmath.phase(v[1, 1]) if abs(v[1, 1]) > _EPS else 0.0  # beta+delta
  diff = 2 * cmath.phase(v[1, 0]) if abs(v[1, 0]) > _EPS else 0.0  # beta-delta
  return alpha, (total + diff) / 2, gamma, (total - diff) / 2


def _qasm_rotation(name: str, angle: float, wire: str) -> List[str]:
  return [f'{name}({_qasm_num(angle)}) {wire};'] if abs(angle) > _EPS else []


def _qasm_single_generic(u, wire: str) -> List[str]:
  """Any single qubit gate, up to a global phase, which has no effect."""

  u = np.asarray(u, dtype=complex).reshape(2, 2)
  if abs(u[0, 1]) < _EPS and abs(u[1, 0]) < _EPS:
    return _qasm_rotation(
        'u1', cmath.phase(u[1, 1]) - cmath.phase(u[0, 0]), wire)
  _, beta, gamma, delta = _zyz(u)
  return (_qasm_rotation('rz', delta, wire) + _qasm_rotation('ry', gamma, wire)
          + _qasm_rotation('rz', beta, wire))


def _qasm_controlled_generic(u, ctl: str, tgt: str) -> List[str]:
  """Any controlled single qubit gate, including the phase."""

  u = np.asarray(u, dtype=complex).reshape(2, 2)
  if abs(u[0, 1]) < _EPS and abs(u[1, 0]) < _EPS:
    # Diagonal: a phase on the control, and a controlled phase gate.
    phase = cmath.phase(u[0, 0])
    lam = cmath.phase(u[1, 1]) - phase
    lines = _qasm_rotation('u1', phase, ctl)
    if abs(lam) > _EPS:
      lines.append(f'cu1({_qasm_num(lam)}) {ctl},{tgt};')
    return lines

  # With u = exp(i alpha) A X B X C and A B C = 1, where X is the Pauli X,
  # we get the controlled u from 2 cnot gates and the phase on the control.
  alpha, beta, gamma, delta = _zyz(u)
  cx = f'cx {ctl},{tgt};'
  return (_qasm_rotation('rz', (delta - beta) / 2, tgt)               # C
          + [cx]
          + _qasm_rotation('rz', -(delta + beta) / 2, tgt)            # B
          + _qasm_rotation('ry', -gamma / 2, tgt)
          + [cx]
          + _qasm_rotation('ry', gamma / 2, tgt)                      # A
          + _qasm_rotation('rz', beta, tgt)
          + _qasm_rotation('u1', alpha, ctl))


def _qasm_gate(op, wires) -> List[str]:
  """The qasm statements for a single gate of the IR."""

  name = op.name
  if op.is_single():
    wire = wires[op.idx0]
    if name in _QASM_SINGLE:
      return [f'{_QASM_SINGLE[name]} {wire};']
    if name in _QASM_PARAMETER and op.val is not None:
      return [f'{name}({helper.pi_fractions(op.val)}) {wire};']
    return _qasm_single_generic(op.gate, wire)

  ctl, tgt = wires[op.ctl], wires[op.idx1]
  if name in _QASM_CONTROLLED:
    return [f'{_QASM_CONTROLLED[name]} {ctl},{tgt};']
  if name in _QASM_CONTROLLED_PARAMETER and op.val is not None:
    return [f'{name}({helper.pi_fractions(op.val)}) {ctl},{tgt};']
  return _qasm_controlled_generic(op.gate, ctl, tgt)


def qasm(ir) -> str:
  """Dump IR in qasm format (OpenQASM 2.0)."""

  decls, wires = _qasm_wires(ir)
  res = 'OPENQASM 2.0;\n'
  res += 'include "qelib1.inc";\n'
  for name, size in decls:
    res += f'qreg {name}[{size}];\n'
  res += '\n'

  for idx in _initial_ones(ir):
    res += f'x {wires[idx]};\n'

  for op in ir.gates:
    if op.is_gate():
      for line in _qasm_gate(op, wires):
        res += line + '\n'
  return res


def libq(ir) -> str:
  """Dump IR to a compilable C++ program with libq."""

  res = (
      '// This file was generated by qc.dump_to_file()\n\n'
      + '#include <math.h>\n'
      + '#include <stdio.h>\n\n'
      + '#include "libq.h"\n\n'
      + 'int main(int argc, char* argv[]) {\n\n'
  )

  total_regs = 0
  for regs in ir.regset:
    total_regs += regs[1]
  res += f'  libq::qureg* q = libq::new_qureg(0, {total_regs});\n\n'

  total_regs = 0
  for regs in ir.regset:
    for r in regs[2].val:
      if r == 1:
        res += f'  libq::x({total_regs}, q);\n'
      total_regs += 1
  res += '\n'

  for op in ir.gates:

    if op.is_gate():
      res += f'  libq::{op.name}('

      if op.is_single():
        res += f'{op.idx0}'
        if op.val is not None:
          res += ', {}'.format(helper.pi_fractions(op.val, 'M_PI'))
        res += ', q);\n'

      if op.is_ctl():
        res += f'{op.ctl}, {op.idx1}'
        if op.val is not None:
          res += ', {}'.format(helper.pi_fractions(op.val, 'M_PI'))
        res += ', q);\n'

  res += '\n  libq::flush(q);\n'
  res += '  libq::print_qureg(q);\n'
  res += '  libq::delete_qureg(q);\n'
  res += '  return EXIT_SUCCESS;\n'
  res += '}\n'
  return res


_CIRQ_SINGLE = {
    'h': 'cirq.H', 'x': 'cirq.X', 'y': 'cirq.Y', 'z': 'cirq.Z', 's': 'cirq.S',
    't': 'cirq.T',
}
_CIRQ_ROTATION = {'rx': 'cirq.rx', 'ry': 'cirq.ry', 'rz': 'cirq.rz'}


def _cirq_matrix(gate) -> str:
  """Python code for the matrix of a gate."""

  # The gates are computed in single precision (and some, like the square
  # roots in ccu(), numerically), and they are not exactly unitary. cirq
  # insists on this, so we use the nearest unitary matrix.
  w, _, vh = np.linalg.svd(np.asarray(gate, dtype=complex).reshape(2, 2))
  m = w @ vh

  def num(z):
    return repr(complex(round(z.real, 12), round(z.imag, 12)))

  return ('np.array([[{}, {}], [{}, {}]])'
          .format(num(m[0, 0]), num(m[0, 1]), num(m[1, 0]), num(m[1, 1])))


def _cirq_gate(name: str, val, gate):
  """Return the code to set up and the expression for a cirq gate.

  Gates which cirq knows are named, all others are made from the matrix of
  the gate, which is always correct and works for any name.

  Args:
    name: The name of the gate (of its target for controlled gates).
    val: Gate parameter, or None.
    gate: The 2x2 matrix of the gate.

  Returns:
    A tuple of the setup code, which can be empty, and the gate expression.
  """

  if name in _CIRQ_SINGLE:
    return '', _CIRQ_SINGLE[name]
  if name in _CIRQ_ROTATION and val is not None:
    return '', f'{_CIRQ_ROTATION[name]}({helper.pi_fractions(val)})'
  return f'm = {_cirq_matrix(gate)}\n', 'cirq.MatrixGate(m)'


def cirq(ir) -> str:
  """Dump IR to a Cirq Python file."""

  res = (
      '# This file was generated by qc.dump_to_file()\n\n'
      + 'import cirq\n'
      + 'import cmath\n'
      + 'from cmath import pi\n'
      + 'import numpy as np\n\n'
  )

  res += 'qc = cirq.Circuit()\n\n'
  res += f'r = cirq.LineQubit.range({_wires(ir)})\n'
  res += '\n'

  for idx in _initial_ones(ir):
    res += f'qc.append(cirq.X(r[{idx}]))\n'

  for op in ir.gates:
    if not op.is_gate():
      continue

    if op.is_single():
      setup, gate = _cirq_gate(op.name, op.val, op.gate)
      res += setup
      res += f'qc.append({gate}(r[{op.idx0}]))\n'
      continue

    if op.name == 'cx':
      res += f'qc.append(cirq.CX(r[{op.ctl}], r[{op.idx1}]))\n'
    elif op.name == 'cz':
      res += f'qc.append(cirq.CZ(r[{op.ctl}], r[{op.idx1}]))\n'
    else:
      # The name of a controlled gate is c + the name of its target gate.
      setup, gate = _cirq_gate(op.name[1:], op.val, op.gate)
      res += setup
      res += f'qc.append({gate}.controlled()(r[{op.ctl}], r[{op.idx1}]))\n'

  res += 'sim = cirq.Simulator()\n'
  res += "print('Simulate...')\n"
  res += 'result = sim.simulate(qc)\n'
  res += 'res_str = str(result)\n'
  res += "print(res_str.encode('utf-8'))\n"

  return res


# --- LaTeX dumper -------------------------------------------------
#
# Gate names as used in circuit.py, mapped to their math-mode label.
_LATEX_NAMES = {
    'h': 'H', 'x': 'X', 'y': 'Y', 'z': 'Z', 's': 'S', 't': 'T', 'v': 'V',
    'yroot': r'\sqrt{Y}', 'u1': 'U_1', 'rx': 'R_x', 'ry': 'R_y', 'rz': 'R_z',
}

_LATEX_SPECIALS = {
    '\\': r'\textbackslash{}', '#': r'\#', '_': r'\_', '&': r'\&',
    '%': r'\%', '{': r'\{', '}': r'\}', '^': r'\^{}', '~': r'\~{}',
    '$': r'\$', '|': r'\textbar{}', '<': r'\textless{}', '>': r'\textgreater{}',
}


def _latex_text(s: str) -> str:
  """Escape arbitrary text (e.g., section names) for LaTeX text mode."""

  return ''.join(_LATEX_SPECIALS.get(c, c) for c in str(s))


def _latex_param(val) -> str:
  """Format a gate parameter as '(x)', as a fraction of pi if possible."""

  if val is None:
    return ''
  s = helper.pi_fractions(val, r'\pi').replace('*', '')
  if r'\pi' not in s and s != '0':
    # Not an exact fraction of pi; retry with a looser tolerance before
    # falling back to a plain number.
    s = f'{val:.4g}'
    for num, denom in itertools.product(range(1, 4), range(1, 129)):
      if math.isclose(abs(val), num * math.pi / denom, rel_tol=1e-6):
        s = ('-' if val < 0 else '') + (str(num) if num > 1 else '')
        s += r'\pi' + (f'/{denom}' if denom > 1 else '')
        break
  return f'({s})'


def _latex_label(name: str, val) -> str:
  """Math-mode label for a single gate (or the target of a controlled one)."""

  dagger = ''
  for suffix in ('dag', '_adj', '^t'):
    if name.endswith(suffix) and name != suffix:
      name, dagger = name[:-len(suffix)], r'^\dagger'
      break
  label = _LATEX_NAMES.get(name)
  if label is None:
    # Unknown names (e.g. 'X^{1/2}' from ccu, or user descriptions) are
    # used verbatim, as they already tend to be math.
    label = name
  return label + dagger + _latex_param(val)


def latex(ir, options: str = 'column sep=0.2cm', max_depth: int = 1) -> str:
  """Dump IR to quantikz LaTeX.

  Gates are scheduled as-soon-as-possible: a gate goes into the first
  column that is free on all wires it touches (including the wires
  between control and target), so independent gates line up vertically.
  Scopes (sections) up to nesting level max_depth are drawn as labelled,
  dashed gate groups; deeper scopes are ignored (None: show all).
  """

  # Wires may come from registers or (e.g. with bitstring()) just gates.
  n = ir.nregs
  for op in ir.gates:
    if op.is_single():
      n = max(n, op.idx0 + 1)
    elif op.is_ctl():
      n = max(n, op.ctl + 1, op.idx1 + 1)
  cols: List[List[str]] = []  # cols[c][q] -> cell content
  frontier = [0] * n  # first free column per wire
  groups = []  # finished scopes: [label, lo, hi, c0, c1, depth]
  stack = []  # open scopes, same layout (lo > hi: no gates yet)
  depth = 0

  def cell(c, q, content):
    while len(cols) <= c:
      cols.append([r'\qw'] * n)
    cols[c][q] = content

  def place(lo, hi, c):
    for q in range(lo, hi + 1):
      frontier[q] = c + 1
    for scope in stack:  # grow all open scopes to cover this gate
      scope[1:5] = [min(scope[1], lo), max(scope[2], hi),
                    min(scope[3], c), max(scope[4], c)]

  for op in ir.gates:
    if op.is_section():
      depth += 1
      if max_depth is None or depth <= max_depth:
        # Barrier: nothing from before the scope may end up inside its box.
        c = max(frontier, default=0)
        for q in range(n):
          frontier[q] = c
        stack.append([_latex_text(op.name), n, -1, c, c - 1, depth])
    elif op.is_end_section():
      if max_depth is None or depth <= max_depth:
        scope = stack.pop()
        if scope[1] <= scope[2]:  # non-empty scope
          groups.append(scope)
          # Later gates on the boxed wires must stay to the right of the box.
          for q in range(scope[1], scope[2] + 1):
            frontier[q] = max(frontier[q], scope[4] + 1)
      depth -= 1
    elif op.is_single():
      q = op.idx0
      c = frontier[q]
      cell(c, q, r'\gate{' + _latex_label(op.name, op.val) + '}')
      place(q, q, c)
    elif op.is_ctl():
      ctl, tgt = op.ctl, op.idx1
      lo, hi = min(ctl, tgt), max(ctl, tgt)
      c = max(frontier[lo:hi + 1])
      name = op.name
      if name == 'cx':
        target = r'\targ{}'
      elif name == 'cz':
        target = r'\control{}'
      elif name == 'cu1':
        target = r'\phase{' + _latex_param(op.val)[1:-1] + '}'
      else:
        if name.startswith('c') and len(name) > 1:
          name = name[1:]
        target = r'\gate{' + _latex_label(name, op.val) + '}'
      cell(c, ctl, r'\ctrl{' + str(tgt - ctl) + '}')
      cell(c, tgt, target)
      place(lo, hi, c)

  # Nested boxes grow outwards and their labels move up by nesting level.
  max_level = max((g[5] for g in groups), default=0)
  for label, lo, hi, c0, c1, level in groups:
    pad = 2 + 2 * (max_level - level)
    cols[c0][lo] += (
        r'\gategroup[wires=' + str(hi - lo + 1) + ',steps=' + str(c1 - c0 + 1)
        + ',style={dashed,rounded corners,'
        + f'inner xsep={pad}pt,inner ysep={pad + 2}pt}},'
        + 'label style={label position=above,anchor=south,'
        + f'yshift={-0.1 + 0.3 * (max_level - level):.1f}cm}}]'
        + '{' + label + '}')

  # Wire labels: register name, index, and initial value.
  def lstick(q):
    for reg in ir.regs:
      if reg[0] == q:
        name, idx = reg[1], reg[2]
        break
    else:
      name, idx = 'q', q
    init = ''
    for rname, size, reg in ir.regset:
      if rname == name and idx < size and hasattr(reg, 'val'):
        init = f' = |{reg.val[idx]}\\rangle'
    if not name:
      name = 'q'
    elif name.isalnum():
      name = r'\mathrm{' + name + '}'  # else: assume it's already math
    return r'\lstick{$' + f'{name}_{{{idx}}}{init}' + '$}'

  opts = f'[{options}]' if options else ''
  rows = []
  for q in range(n):
    rows.append(' & '.join([lstick(q)] + [col[q] for col in cols] + [r'\qw']))
  return (r'\begin{qcc}' + opts + '\n' + ' \\\\\n'.join(rows) + '\n'
          + r'\end{qcc}')


def totext(ir) -> str:
  """Minimal Dumper to ASCII text."""

  def mkname(op):
    name = op.name.upper()
    if name == 'CZ':
      name = 'o'
    if op.val is not None:
      parm = '({})'.format(helper.pi_fractions(op.val, r'pi'))
      name += parm
    return name

  # Sections are not drawn. Wires may come from registers or (e.g. with
  # bitstring()) just from gates.
  gates = [op for op in ir.gates if op.is_gate()]
  n = _wires(ir)

  # First let's create a matrix according to circuit size,
  # before populating it with gates or lines.
  larr = []
  for q in range(n):
    larr.append([''] * (len(gates) + 1))

  depth = 0
  for op in gates:
    name = mkname(op)
    for r in range(n):
      larr[r][depth] = '-' * (len(name) + 2)

    if op.is_single():
      larr[op.idx0][depth] = '-' + name + '-'

    if op.is_ctl():
      larr[op.ctl][depth] = '-' + 'o' + ('-' * (len(name) - 1)) + '-'
      larr[op.idx1][depth] = '-' + name + '-'
      direction = int((op.idx1 - op.ctl) / abs(op.idx1 - op.ctl))
      for i in range(op.ctl + direction, op.idx1, direction):
        larr[i][depth] = '-|-'

    depth += 1

  ret = ''
  for q in range(n):
    ret += ''.join(larr[q]) + '\n'

  return ret
