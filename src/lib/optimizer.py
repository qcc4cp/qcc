# python3
"""Optimize the IR with a variety of (simple) techniques."""

from src.lib import ir

# Currently, this is work-in-progress.
# The code just builds a 2D grid representation from the IR.


def build_2d_grid(parm_ir):
  """Build simple grid with a column for each gate."""

  grid = []
  for g in parm_ir.gates:
    step = [None] * parm_ir.ngates
    if g.is_single():
      step[g.idx0] = g
    if g.is_ctl():
      step[g.ctl] = g.ctl
      step[g.idx1] = g
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
  return new_ir
