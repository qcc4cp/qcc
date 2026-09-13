# python3
"""libxgates-compatible entry points backed by qcc's CPU block planner."""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import time

import numpy as np

from src.lib.block_planner import BlockFinder
from src.lib.block_planner import PlannedExecutor
from src.lib.block_planner import PrimitiveGate

_CONTEXTS = {}
_COMPLETED = []


def _install_circuit_hooks():
  from src.lib import circuit

  cls = circuit.qc
  if hasattr(cls, '_planner_original_unitary'):
    return

  cls._planner_original_unitary = cls.unitary
  cls._planner_original_measure_bit = cls.measure_bit
  cls._planner_original_tprod = cls._tprod

  def unitary(self, op, idx):
    flush(self.psi)
    return cls._planner_original_unitary(self, op, idx)

  def measure_bit(self, idx, tostate=0, collapse=True):
    flush(self.psi)
    return cls._planner_original_measure_bit(
        self, idx, tostate=tostate, collapse=collapse)

  def tprod(self, new_state, nqubits):
    flush(self.psi)
    return cls._planner_original_tprod(self, new_state, nqubits)

  cls.unitary = unitary
  cls.measure_bit = measure_bit
  cls._tprod = tprod


def _context(psi):
  _install_circuit_hooks()
  key = id(psi)
  value = _CONTEXTS.get(key)
  if value is None:
    value = {
        'psi': psi,
        'operations': [],
        'capture_start': time.perf_counter(),
    }
    _CONTEXTS[key] = value
  return value


def apply1(psi, gate, nbits, target, bitwidth=64):
  """Capture a qcc apply1 operation."""

  del bitwidth
  if np.asarray(psi).size.bit_length() - 1 != nbits:
    raise ValueError('state size and nbits disagree')
  _context(psi)['operations'].append(
      PrimitiveGate(np.asarray(gate).reshape(2, 2).copy(), int(target)))


def applyc(psi, gate, nbits, control, target, bitwidth=64):
  """Capture a qcc applyc operation."""

  del bitwidth
  if np.asarray(psi).size.bit_length() - 1 != nbits:
    raise ValueError('state size and nbits disagree')
  # Match qcc's Python fallback for negative controls: its reversed control
  # mask is outside the state width, so the operation is a no-op.
  if int(control) < 0:
    return
  _context(psi)['operations'].append(
      PrimitiveGate(
          np.asarray(gate).reshape(2, 2).copy(),
          int(target),
          int(control),
      ))


def flush(psi):
  """Materialize the pending epoch into the original qcc State object."""

  context = _CONTEXTS.pop(id(psi), None)
  if context is None:
    return None
  max_qubits = int(os.environ.get('QCC_PLANNER_MAX_QUBITS', '7'))
  min_gates = int(os.environ.get('QCC_PLANNER_MIN_GATES', '2'))
  finder = BlockFinder(max_qubits=max_qubits, matrix_min_gates=min_gates)
  start = time.perf_counter()
  result, stats = PlannedExecutor(finder).execute(
      psi, context['operations'])
  np.copyto(np.asarray(psi), result, casting='unsafe')
  report = stats.to_dict()
  report.update({
      'state_qubits': np.asarray(psi).size.bit_length() - 1,
      'capture_ms': (start - context['capture_start']) * 1000.0,
      'flush_wall_ms': (time.perf_counter() - start) * 1000.0,
      'max_block_qubits': max_qubits,
      'matrix_min_gates': min_gates,
  })
  _COMPLETED.append(report)
  return report


def completed_stats():
  return tuple(_COMPLETED)


def _install_state_hooks():
  from src.lib.state import State
  from src.lib.tensor import Tensor

  if not hasattr(State, '_planner_original_prob'):
    for name in ('prob', 'ampl', 'maxprob', 'phase', 'density', 'adjoint',
                 'normalize', 'diff', 'dump'):
      original = getattr(State, name)
      setattr(State, f'_planner_original_{name}', original)

      def make_wrapper(method):
        def wrapper(self, *args, **kwargs):
          flush(self)
          return method(self, *args, **kwargs)
        return wrapper

      setattr(State, name, make_wrapper(original))

    State._planner_original_getitem = State.__getitem__

    def getitem(self, key):
      flush(self)
      return State._planner_original_getitem(self, key)

    State.__getitem__ = getitem

  if not hasattr(Tensor, '_planner_original_is_close'):
    Tensor._planner_original_is_close = Tensor.is_close

    def is_close(self, arg, tolerance=1e-6):
      if isinstance(self, State):
        flush(self)
      if isinstance(arg, State):
        flush(arg)
      return Tensor._planner_original_is_close(self, arg, tolerance)

    Tensor.is_close = is_close

  if not hasattr(np, '_planner_original_allclose'):
    np._planner_original_allclose = np.allclose

    def allclose(a, b, *args, **kwargs):
      if isinstance(a, State):
        flush(a)
      if isinstance(b, State):
        flush(b)
      return np._planner_original_allclose(a, b, *args, **kwargs)

    np.allclose = allclose


def _finish():
  for context in list(_CONTEXTS.values()):
    flush(context['psi'])
  output = os.environ.get('QCC_PLANNER_STATS_FILE')
  if output:
    path = pathlib.Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_COMPLETED, indent=2) + '\n')


_install_state_hooks()
atexit.register(_finish)
