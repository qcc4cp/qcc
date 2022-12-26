# python3
"""Oracle Synthesis via BQSKit."""

# Explore Oracle-to-gate synthesis with BQSKit
#
# For the four potential Deutsch Oracles (listed below in the
# list 'deutsch', we use BQSKit to synthesize circuits that would
# produce those Oracle matrices.
#
# This needs BQSKit to be installed.
#
# Run (without bazel) as:
#    python3 oracle_synth.py

import shutil
import sys

from src.lib import ops


# =========================================================
# First, let's make sure bqskit has been installed.
#
try:
  # pylint: disable=g-import-not-at-top
  # pylint: disable=redefined-builtin
  from bqskit import compile
except Exception:  # pylint: disable=broad-except
  print('*** WARNING ***')
  print('Could not import bqskit.')
  print('Please install before trying to run this script.\n')
  sys.exit(0)

# bqskit relies on 'dask-scheduler', let's make sure it can be found
#
try:
  sched = shutil.which('dask-scheduler')
  if sched is None:
    print('*** WARNING ***')
    print('Could not locate binary "dask-scheduler", required by bqskit')
    sys.exit(0)
except Exception:  # pylint: disable=broad-except
  print('Something wrong with importing "shutil"')
  sys.exit(0)
# =========================================================


# The four possible Deutsch Oracles:
# BQSKit fails on the identity gate - nothing to be done.
deutsch = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
           [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
           [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
           [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]]

# The circuits as they were being produced by BQSKit.
#
# We list the circuits here and ensure they are correct.
# Below in this script we re-generate the circuits.
#
circuits = [ops.Identity() * ops.Identity(),
            ops.Cnot(0, 1),
            ops.Cnot(0, 1) @
            (ops.Identity() *
             ops.U(3.1415926, 0.0, 3.1415926)),
            ops.Identity() *
            ops.U(3.1415926, 0.0, 3.1415926)]


# We upfront compare the (generated) operators to the
# intended operators.
#
for idx, gate in enumerate(deutsch):
  for i in range(4):
    for j in range(4):
      diff = gate[i][j] - circuits[idx][i][j]
      if abs(diff) > 0.00001:
        raise AssertionError('Gates DIFFER', i, j, '->', diff)
  print(f'Gate[{idx}]: Match')


# We synthesize the circuit gates here via BQSKit's compile().
#
print('Re-generate the gates')
for idx, gate in enumerate(deutsch):
  print(f'Gate[{idx}]:', gate)
  try:
    circ = compile(gate, optimization_level=3)
  except Exception:  # pylint: disable=broad-except
    print('  Compilation failed (expected at opt-level=3 for Gate[0]).')
    continue

  filename = '/tmp/deutsch' + str(idx) + '.qasm'
  print('Gates  :', circ.gate_counts, ' write to:', filename)
  try:
    circ.save(filename)
    file = open(filename, 'r+')
    print(file.read())
  except Exception:  # pylint: disable=broad-except
    print('*** WARNING ***')
    print('Cannot write to file:', filename)
