# python3
"""Optimize the IR with a variety of (simple) techniques."""

from src.lib import ir

# Currently, this is work-in-progress.
# The code just builds a 2D grid representation from the IR.


def build_2d_grid(parm_ir):
  """Build simple grid with a row for each qubit and a column for each node."""

  # The rows are the qubits: those of the registers, and any qubit that a
  # gate uses without having been allocated via a register.
  nqubits = max(parm_ir.nregs, 1)
  for g in parm_ir.gates:
    if g.is_single():
      nqubits = max(nqubits, g.idx0 + 1)
    if g.is_ctl():
      nqubits = max(nqubits, g.ctl + 1, g.idx1 + 1)

  grid = []
  for g in parm_ir.gates:
    step = [None] * nqubits
    if g.is_single():
      step[g.idx0] = g
    if g.is_ctl():
      step[g.ctl] = g.ctl
      step[g.idx1] = g
    if g.is_section() or g.is_end_section():
      # Sections apply to all qubits, we keep them in the first row.
      step[0] = g
    grid.append(step)
  return grid


def ir_from_grid(grid):
  """From a grid, reconstruct the IR."""

  new_ir = ir.Ir()
  for step in grid:
    for i in range(len(step)):
      if not step[i]:
        continue
      if isinstance(step[i], ir.Node):
        new_ir.add_node(step[i])
  return new_ir


def optimize(parm_ir):
  """Optimize the IR with a variety of techniques."""

  grid = build_2d_grid(parm_ir)
  new_ir = ir_from_grid(grid)

  # The grid only has the gates. Keep the registers, they are the names
  # of the qubits in the output.
  new_ir.regs = list(parm_ir.regs)
  new_ir.nregs = parm_ir.nregs
  new_ir.regset = list(parm_ir.regset)
  return new_ir
