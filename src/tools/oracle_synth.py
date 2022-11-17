# Explore Oracle-to-gate synthesis with BQSKit
#
# For the four potential Deutsch Oracles (listed below in the
# list 'deutsch', we use BQSKit to synthesize circuits that would
# produce those Oracle matrices.
#
# This needs BQSKit to be installed.
from bqskit import compile

from src.lib import ops


# The four possible Deutsch Oracles:
#
deutsch= [
           # BQSKit fails on the identity gate - nothing to be done.
           [ [ 1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
           [ [ 1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
           [ [ 0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
           [ [ 0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
         ]

# The circuits as they are being produced by BQSKit.
# We list the circuits here and ensure they are correct.
# Below in this script we re-generate the circuits.
#
circuits = [
              ops.Identity() * ops.Identity(),
              
              ops.Cnot(0, 1),
              
              ops.Cnot(0, 1) @
              (ops.Identity() *
               ops.U(9.4247779, 3.1415926, 6.2831852)),
              
              ops.Identity() *
              ops.U(3.1415926, 0.0, 3.1415926)
           ]


# We upfront compare the (generated) gates to the
# intended gates.
#
for idx, gate in enumerate(deutsch):
  for i in range(4):
     for j in range(4):
        diff = gate[i][j] - circuits[idx][i][j]
        if abs(diff) > 0.00001:
           raise AssertionError('Gates DIFFER', i, j, '->', diff)
  print(f'Gate[{idx}]: Match')


# We generate the gates here.
#
print('\nRe-generate the gates')
for idx, gate in enumerate(deutsch):
  print(f'Gate[{idx}]:', gate)
  try:
    circ = compile(gate, optimization_level=3)
  except:
    print('  Compilation failed (expected for Gate[0]).')
    continue
  filename = '/tmp/deutsch' + str(idx) + '.qasm'
  print('  Gates:', circ.gate_counts, ' write to:', filename)
  circ.save(filename)

