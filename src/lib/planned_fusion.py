# python3
"""Small-block gate fusion primitives for the planned CPU backend."""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class BlockGate:
  """A one-qubit operation expressed in block-local qubit indices."""

  gate: np.ndarray
  target: int
  control: Optional[int] = None


def pair_indices(nbits: int, target: int):
  """Return low/high state-vector indices in qcc's MSB qubit order."""

  if not 0 <= target < nbits:
    raise ValueError(f'target {target} is outside [0, {nbits})')
  target_bit = nbits - target - 1
  stride = 1 << target_bit
  groups = 1 << (nbits - target_bit - 1)
  low = np.empty(groups * stride, dtype=np.int64)
  cursor = 0
  for group in range(groups):
    base = group * (2 * stride)
    low[cursor:cursor + stride] = np.arange(base, base + stride)
    cursor += stride
  return low, low + stride


def apply_gate(state, gate, target, control=None):
  """Apply one gate out of place using the same indexing as libxgates."""

  state = np.asarray(state)
  nbits = state.size.bit_length() - 1
  low, high = pair_indices(nbits, target)
  if control is not None:
    if not 0 <= control < nbits or control == target:
      raise ValueError(f'invalid control {control} for target {target}')
    control_bit = nbits - control - 1
    active = (low & (1 << control_bit)) != 0
    low, high = low[active], high[active]
  result = state.copy()
  old_low = state[low].copy()
  old_high = state[high].copy()
  gate = np.asarray(gate).reshape(2, 2)
  result[low] = gate[0, 0] * old_low + gate[0, 1] * old_high
  result[high] = gate[1, 0] * old_low + gate[1, 1] * old_high
  return result


def build_block_unitary(block_qubits: int,
                          operations: list[BlockGate]) -> np.ndarray:
  """Build a fused unitary in O(gates * 4**block_qubits)."""

  size = 1 << block_qubits
  unitary = np.eye(size, dtype=np.complex64)
  for operation in operations:
    if not 0 <= operation.target < block_qubits:
      raise ValueError('operation target lies outside the fused block')
    if (operation.control is not None and
        not 0 <= operation.control < block_qubits):
      raise ValueError('operation control lies outside the fused block')
    low, high = pair_indices(block_qubits, operation.target)
    if operation.control is not None:
      control_bit = block_qubits - operation.control - 1
      active = (low & (1 << control_bit)) != 0
      low, high = low[active], high[active]
    old_low = unitary[low, :].copy()
    old_high = unitary[high, :].copy()
    gate = np.asarray(operation.gate)
    unitary[low, :] = gate[0, 0] * old_low + gate[0, 1] * old_high
    unitary[high, :] = gate[1, 0] * old_low + gate[1, 1] * old_high
  return unitary
