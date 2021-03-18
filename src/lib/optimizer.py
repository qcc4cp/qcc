# python3
"""Optimize the IR with a variety of (simple) techniques."""

import math

from src.lib import helper
from src.lib import ir

def build_2d_grid(ir):
    """ Build simple grid with a column for each gate."""
    grid = []
    for g in ir.gates:
        step = [None] * ir.ngates
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
            if isinstance(step[i],ir. Node):
                new_ir.add_node(step[i])
    return new_ir

def optimize(ir):
    """Optimize the IR with a variety of techniques."""

    grid = build_2d_grid(ir)
    ir = ir_from_grid(grid)
    return ir
