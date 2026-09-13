# python3
"""Dependency-safe block planning and host execution for qcc gates."""

from __future__ import annotations

import dataclasses
import time
from typing import Iterable, Optional

import numpy as np

from src.lib.planned_fusion import BlockGate
from src.lib.planned_fusion import apply_gate
from src.lib.planned_fusion import build_block_unitary


@dataclasses.dataclass(frozen=True)
class PrimitiveGate:
  """A captured qcc single- or controlled-single-qubit operation."""

  gate: np.ndarray
  target: int
  control: Optional[int] = None
  source_count: int = 1

  @property
  def qubits(self) -> frozenset[int]:
    if self.control is None:
      return frozenset((self.target,))
    return frozenset((self.control, self.target))


@dataclasses.dataclass(frozen=True)
class PlanSegment:
  operations: tuple[PrimitiveGate, ...]
  qubits: frozenset[int]
  strategy: str
  source_gate_count: int


@dataclasses.dataclass
class ExecutionStats:
  original_gates: int = 0
  compacted_gates: int = 0
  segments: int = 0
  matrix_segments: int = 0
  fallback_segments: int = 0
  matrix_source_gates: int = 0
  fallback_source_gates: int = 0
  layout_transitions: int = 0
  estimated_permutation_bytes: int = 0
  unitary_build_ms: float = 0.0
  matrix_apply_ms: float = 0.0
  fallback_ms: float = 0.0
  final_canonicalize_ms: float = 0.0
  total_execute_ms: float = 0.0

  def to_dict(self):
    return dataclasses.asdict(self)


class BlockFinder:
  """Find maximal contiguous blocks without reordering dependent gates."""

  def __init__(self, max_qubits=7, matrix_min_gates=2,
               matrix_min_state_qubits=8):
    if max_qubits < 1:
      raise ValueError('max_qubits must be positive')
    self.max_qubits = max_qubits
    self.matrix_min_gates = matrix_min_gates
    self.matrix_min_state_qubits = matrix_min_state_qubits

  @staticmethod
  def compact_adjacent(operations: Iterable[PrimitiveGate]):
    result = []
    for operation in operations:
      if (result and result[-1].target == operation.target and
          result[-1].control == operation.control):
        previous = result.pop()
        result.append(
            PrimitiveGate(
                gate=np.asarray(operation.gate @ previous.gate,
                                dtype=np.complex64),
                target=operation.target,
                control=operation.control,
                source_count=previous.source_count + operation.source_count,
            ))
      else:
        result.append(operation)
    return result

  def _strategy(self, nbits, operations, qubits):
    if (nbits >= self.matrix_min_state_qubits and
        len(operations) >= self.matrix_min_gates and
        1 < len(qubits) <= self.max_qubits):
      return 'matrix'
    return 'butterfly'

  def plan(self, operations: Iterable[PrimitiveGate], nbits: int):
    compacted = self.compact_adjacent(operations)
    segments = []
    current = []
    active = set()

    def close():
      nonlocal current, active
      if not current:
        return
      source_count = sum(op.source_count for op in current)
      segments.append(
          PlanSegment(
              tuple(current), frozenset(active),
              self._strategy(nbits, current, active), source_count))
      current, active = [], set()

    for operation in compacted:
      if not 0 <= operation.target < nbits:
        raise ValueError(f'invalid target {operation.target}')
      if operation.control is not None:
        if not 0 <= operation.control < nbits:
          raise ValueError(f'invalid control {operation.control}')
        if operation.control == operation.target:
          raise ValueError('control equals target')
      if current and len(active | set(operation.qubits)) > self.max_qubits:
        close()
      current.append(operation)
      active.update(operation.qubits)
    close()
    return compacted, segments


class LayoutState:
  """State tensor plus the logical qubit stored on each tensor axis."""

  def __init__(self, values):
    values = np.asarray(values, dtype=np.complex64)
    nbits = values.size.bit_length() - 1
    if values.ndim != 1 or (1 << nbits) != values.size:
      raise ValueError('state must be a power-of-two complex vector')
    self.nbits = nbits
    self.tensor = values.reshape((2,) * nbits)
    self.axis_qubits = list(range(nbits))

  def canonical(self):
    permutation = [self.axis_qubits.index(q) for q in range(self.nbits)]
    if permutation != list(range(self.nbits)):
      value = np.transpose(self.tensor, permutation)
    else:
      value = self.tensor
    return np.ascontiguousarray(value.reshape(-1), dtype=np.complex64)

  def reset_canonical(self, values):
    self.tensor = np.asarray(values, dtype=np.complex64).reshape(
        (2,) * self.nbits)
    self.axis_qubits = list(range(self.nbits))

  def apply_matrix(self, segment: PlanSegment, stats: ExecutionStats):
    selected = [q for q in self.axis_qubits if q in segment.qubits]
    remaining = [q for q in self.axis_qubits if q not in segment.qubits]
    desired = selected + remaining
    permutation = [self.axis_qubits.index(q) for q in desired]
    if permutation != list(range(self.nbits)):
      stats.layout_transitions += 1
      stats.estimated_permutation_bytes += (
          self.tensor.size * self.tensor.dtype.itemsize)
    view = np.transpose(self.tensor, permutation)
    matrix = view.reshape(1 << len(selected), -1)
    local = {qubit: index for index, qubit in enumerate(selected)}
    operations = [
        BlockGate(op.gate, local[op.target],
                  None if op.control is None else local[op.control])
        for op in segment.operations
    ]
    start = time.perf_counter()
    unitary = build_block_unitary(len(selected), operations)
    stats.unitary_build_ms += (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    with np.errstate(all='ignore'):
      result = unitary @ matrix
    stats.matrix_apply_ms += (time.perf_counter() - start) * 1000.0
    self.tensor = np.asarray(result, dtype=np.complex64).reshape(
        (2,) * self.nbits)
    self.axis_qubits = desired


class PlannedExecutor:
  """Execute a captured gate epoch using fused matrices where profitable."""

  def __init__(self, finder=None):
    self.finder = finder or BlockFinder()

  def execute(self, values, operations: Iterable[PrimitiveGate]):
    operations = list(operations)
    nbits = np.asarray(values).size.bit_length() - 1
    compacted, segments = self.finder.plan(operations, nbits)
    stats = ExecutionStats(
        original_gates=sum(op.source_count for op in operations),
        compacted_gates=len(compacted),
        segments=len(segments),
    )
    current = LayoutState(values)
    total_start = time.perf_counter()
    for segment in segments:
      if segment.strategy == 'matrix':
        stats.matrix_segments += 1
        stats.matrix_source_gates += segment.source_gate_count
        current.apply_matrix(segment, stats)
      else:
        stats.fallback_segments += 1
        stats.fallback_source_gates += segment.source_gate_count
        start = time.perf_counter()
        canonical = current.canonical()
        for operation in segment.operations:
          canonical = apply_gate(canonical, operation.gate,
                                 operation.target, operation.control)
        current.reset_canonical(canonical)
        stats.fallback_ms += (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    result = current.canonical()
    stats.final_canonicalize_ms = (time.perf_counter() - start) * 1000.0
    stats.total_execute_ms = (time.perf_counter() - total_start) * 1000.0
    return result, stats
