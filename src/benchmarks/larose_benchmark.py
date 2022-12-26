# python3
"""A benchmark based on the paper from Ryan LaRose."""

# The paper can be found here:
#   Overview and Comparison of Gate Level Quantum Software Platforms
#   https://arxiv.org/pdf/1807.02500.pdf
#
# We write the benchmark in Python and then generated the various flavors
# from it, eg., libq, projectq, qasm, etc.

import random

from absl import app
from absl import flags

from src.lib import circuit

flags.DEFINE_integer('nbits', 28, 'Number of Qubits')
flags.DEFINE_integer('depth', 28, 'Depth of Circuit')

# Informal benchmarking on my workstation shows that
# this xgate accelerated benchmark runs in about:
#
# qubit      time
#  26           7 secs
#  27          14 secs
#  28          29 secs
#  29          58 secs
#  30         122 secs
#
# A 28/28 libq based circuit runs in about 3 seconds. But that's
# because this (kind of silly) benchmark does not introduce a
# lot of states with non-zero amplitudes. In other words,
# this is a good benchmark for full-state sims, but not for
# simulations based on spare representations, like libq.


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print(f'LaRose benchmark with {flags.FLAGS.nbits} qubits, ' +
        f'depth: {flags.FLAGS.depth}...')

  qc = circuit.qc(eager=False)
  qc.reg(flags.FLAGS.nbits, random.randint(0, 2^flags.FLAGS.nbits), name='q')

  for d in range(flags.FLAGS.depth):
    print(f'  depth: {d}')
    for bit in range(flags.FLAGS.nbits):
      qc.h(bit)
      qc.v(bit)
      if bit > 0:
        qc.cx(bit, 0)
  qc.dump_to_file()


if __name__ == '__main__':
  app.run(main)
