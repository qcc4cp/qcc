# python3
"""Example: Quantum Supremacy Circuit."""

import enum
import random
import time

from absl import app
from absl import flags
from src.lib import circuit

flags.DEFINE_integer('nbits', 20, 'Number of Qubits')
flags.DEFINE_integer('depth', 20, 'Depth of Circuit')
flags.DEFINE_integer('target_nbits', 53, 'Number or target qubits')
flags.DEFINE_integer('target_depth', 20, 'Depth of target circuit')
flags.DEFINE_integer('machines', 100, 'Number of machines used')
flags.DEFINE_integer('cores', 255, 'Number of cores per machine')


# This code seeks to implement / simulate the circuit that was
# outlined in Google's Quantum Supremacy paper in Nature:
#   https://arxiv.org/pdf/1608.00263.pdf
#
# While the circuit is random, there are strict rules that
# guarantee that the circuit can execute on te Google hardware.

# Note: There is a variety of ways how the rules for circuit
# construction can be interpreted. We believe that even if the
# generated circuit below is off target by a bit, it still serves
# well to estimate the complexity of interpreting / simulating
# the circuit.
#
# In this code we try to replicate the circuit generation and then
# measure it's simulation performance, estimating how long it would
# take to simulate 50+ qubits on some hypothetical big HW.


# The paper suggests 8 patterns of size 6*6 of CZ gates. to fully
# encode the patterns one would need at least 36 qubits, but that's
# hard to simulate. We make a compromise and try to apply as many
# gates as possible. The patterns are encoded as simple lists, where
# a non-zero element at index i serves as the control and has
# the offset to the target qubit. To go right, the offset is 1, to
# go down the offset is 6.
#
# Pattern number 1 looks like this:
#    o o o-o o o
#    o-o o o o-o
#    o o o-o o o
#    o-o o o o-o
#    o o o-o o o
#    o-o o o o-o
#
# fmt: off
pattern1 = [0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0] * 3

pattern2 = [1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0] * 3

pattern3 = [0, 0, 0, 0, 0, 0,
            0, 6, 0, 6, 0, 6,
            0, 0, 0, 0, 0, 0,
            0, 6, 0, 6, 0, 6,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0]

pattern4 = [0, 0, 0, 0, 0, 0,
            6, 0, 6, 0, 6, 0,
            0, 0, 0, 0, 0, 0,
            6, 0, 6, 0, 6, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0]

pattern5 = [0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 0] * 3

pattern6 = [0, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0] * 3

pattern7 = [6, 0, 6, 0, 6, 0,
            0, 0, 0, 0, 0, 0,
            0, 6, 0, 6, 0, 6,
            0, 0, 0, 0, 0, 0,
            6, 0, 6, 0, 6, 0,
            0, 0, 0, 0, 0, 0]

pattern8 = [0, 6, 0, 6, 0, 6,
            0, 0, 0, 0, 0, 0,
            0, 6, 0, 6, 0, 6,
            0, 0, 0, 0, 0, 0,
            0, 6, 0, 6, 0, 6,
            0, 0, 0, 0, 0, 0]

patterns = [pattern1, pattern2, pattern3, pattern4,
            pattern5, pattern6, pattern7, pattern8]
# fmt: on


class Gate(enum.Enum):
  UNK = 0
  H = 1
  T = 2
  U = 3
  CZ = 4


def gstr(g):
  """Convert enum to string."""

  if g == Gate.UNK:
    return '   '
  if g == Gate.H:
    return 'h  '
  if g == Gate.T:
    return 't  '
  if g == Gate.U:
    return 'u  '
  if g == Gate.CZ:
    return 'cz '


def build_circuit(nbits, depth):
  """Construct the full circuit."""

  def apply_pattern(pattern):
    bits_touched = []
    for i in range(min(nbits, len(pattern))):
      if pattern[i] != 0 and i + pattern[i] < nbits:
        bits_touched.append((i, i + pattern[i]))
    return bits_touched

  print('\nBuild smaller, representative circuit ({} qubits, depth {})\n'.
        format(nbits, depth))

  state0 = [Gate.H] * nbits
  states = []
  states.append(state0)

  for _ in range(depth - 1):
    state1 = [Gate.UNK] * nbits
    touched = apply_pattern(patterns[random.randint(0, 7)])
    for idx1, idx2 in touched:
      state1[idx1] = Gate.CZ
      state1[idx2] = Gate.CZ
    for i in range(len(state0)):
      if state0[i] == Gate.CZ and state1[i] != Gate.CZ:
        state1[i] = Gate.U
      if state0[i] == Gate.U and state1[i] != Gate.CZ:
        state1[i] = Gate.T
      if state0[i] == Gate.H and state1[i] != Gate.CZ:
        state1[i] = Gate.T
    state0 = state1
    states.append(state0)

  state0 = [Gate.H] * nbits
  states.append(state0)
  return states


def print_state(states, nbits, depth):
  """Print states in horizontal layout."""

  print('  ', end='')
  for idx in range(depth + 1):
    print(' {:2d}'.format(idx), end='')
  print()
  for idx in range(nbits):
    print('{:2d}: '.format(idx), end='')
    for s in states:
      print(gstr(s[idx]), end='')
    print()


def optimize_circuit(states, nbits, depth):
  """Simple optimizations of the circuit."""

  def is_combinable(state):
    """Check whether a gate is a combinable single qubit gate."""

    if state == Gate.T or state == Gate.H or state == Gate.U:
      return True
    return False

  def combine_right(states, index, bit):
    # Pass over Gate.UNK
    for travel in range(index, depth + 1):
      curr_state = states[travel]
      if curr_state[bit] != Gate.UNK:
        break
    # Check current gate
    if is_combinable(curr_state[bit]):
      return True
    return False

  num_gates_removed = 0
  for bit in range(nbits):
    for index in range(0, depth):
      state0 = states[index]
      if not is_combinable(state0[bit]):
        continue
      if combine_right(states, index + 1, bit):
        state0[bit] = Gate.UNK
        num_gates_removed += 1
  print(f'\nOptimizer: Combined {num_gates_removed} gates\n')


def sim_circuit(states, nbits, depth, target_nbits, target_depth):
  """Simulate the geerated circuit."""

  print('\nSimulate...\n')

  start_time = time.time()
  ngates = 0
  qc = circuit.qc('Supremacy Circuit')
  qc.reg(nbits)

  for d in range(depth):
    s = states[d]
    for i in range(nbits):
      if s[i] == Gate.UNK:
        continue
      ngates += 1
      if s[i] == Gate.T:
        qc.t(i)
      if s[i] == Gate.H:
        qc.h(i)
      if s[i] == Gate.U:
        if random.randint(0, 1) == 0:
          qc.v(i)
        else:
          qc.yroot(i)
      if s[i] == Gate.CZ:
        ngates += 1  # This is just an estimate of the overhead
        if i < nbits - 1 and s[i + 1] == Gate.CZ:
          qc.cz(i, i + 1)
          s[i + 1] = Gate.UNK
        if i < nbits - 6 and s[i + 6] == Gate.CZ:
          qc.cz(i, i + 6)
          s[i + 6] = Gate.UNK

    end_time = time.time()
    duration = end_time - start_time

    if ngates > 0:
      print(('Depth={:2d}, Time: {:.2f} Iter: {:.3f} [Sec]' +
             '{:.3f}/g {:.3f}/g/b').
            format(d, duration, duration / (d + 1),
                   duration / ngates,
                   1000000000 * duration / ngates /
                   (2 ** (nbits - 1) * 16)))

  qc.dump_to_file()

  # To estimate simulation time of a 53-qubit circuit we make
  # several assumptions, as listed below. Applying 1-qubit gates and
  # controlled 2-qubit gates is basically linear over the size of
  # the state vector. This is definitely memory bound, so we use
  # the metric "time per gate per byte" as the basis. Experiments
  # show that this metric is quite stable over different qubit
  # sizes on a single machine (around 0.053e-6 on my workstatioin)..
  #
  # We assume 0 communication costs, so the following is an
  # optimistic estimate.

  print('\nEstimate simulation time on larger circuit:\n')
  gate_ratio = ngates / nbits / depth
  print('Simulated circuit:')
  print('  Qubits                  : {:d}'.format(nbits))
  print('  Circuit Depth           : {:d}'.format(depth))
  print('  Gates                   : {:.2f}'.format(ngates))
  print('  State Memory            : {:.4f} MB'.format(
      2 ** (nbits - 1) * 16 / (1024 ** 2)))
#  print('Estimated C++ / Python    : {}x'.format(flags.FLAGS.ccfactor))
  print('Estimated Circuit Qubits  : {}'.format(target_nbits))
  print('Estimated Circuit Depth   : {}'.format(target_depth))
  print('Estimated State Memory    : {:.5f} TB'.format(
      2 ** (target_nbits - 1) * 16 / (1024 ** 4)))
  print('Machines used             : {}'.format(flags.FLAGS.machines))
  print('Estimated cores per server: {}'.format(flags.FLAGS.cores))
  print('Estimated gate density    : {:.2f}'.format(gate_ratio))

  estimated_sim_time_secs = (
      # time per gate per byte
      (duration / ngates / (2**(nbits - 1) * 16))
      # gates
      * target_nbits
      # gate ratio scaling factor to circuit size
      * gate_ratio
      # depth
      * target_depth
      # memory
      * 2**(target_nbits - 1) * 16
      # number of machines
      / flags.FLAGS.machines
      # Active core per machine
      / flags.FLAGS.cores)

  print('Estimated for {} qbits: {:.2f} y or {:.2f} d or ({:.0f} sec)'.
        format(target_nbits,
               estimated_sim_time_secs / 3600 / 24 / 365,
               estimated_sim_time_secs / 3600 / 24,
               estimated_sim_time_secs))

  print('Estimated sim for FULL experiment, {} qbits: {:.2f} years'.
        format(target_nbits,
               1000000 * estimated_sim_time_secs / 3600 / 24 / 365))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  nbits = flags.FLAGS.nbits
  depth = flags.FLAGS.depth

  states = build_circuit(nbits, depth)
  print_state(states, nbits, depth)
  sim_circuit(states, nbits, depth,
              flags.FLAGS.target_nbits, flags.FLAGS.target_depth)


if __name__ == '__main__':
  app.run(main)
