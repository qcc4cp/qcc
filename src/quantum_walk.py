# python3
"""Example: Simple quantum walk with Hadamard Coin."""

from absl import app

from src.lib import circuit
from src.lib import helper
from src.lib import ops


def incr(qc, idx: int, nbits: int, aux, controller):
  """Increment-by-1 circuit."""

  # See "Efficient Quantum Circuit Implementation of
  # Quantum Walks" by Douglas, Wang.
  #
  #  -X--
  #  -o--X--
  #  -o--o--X--
  #  -o--o--o--X--
  #  ...
  for i in range(nbits):
    ctl = controller.copy()
    for j in range(nbits - 1, i, -1):
      ctl.append(j + idx)
    qc.multi_control(ctl, i+idx, aux, ops.PauliX(), 'multi-1-X')


def decr(qc, idx: int, nbits: int, aux, controller):
  """Decrement-by-1 circuit."""

  # See "Efficient Quantum Circuit Implementation of
  # Quantum Walks" by Douglas, Wang.
  #
  # Similar to incr, except controlled-by-0's are being used.
  #
  #  -X--
  #  -0--X--
  #  -0--0--X--
  #  -0--0--0--X--
  #  ...
  for i in range(nbits):
    ctl = controller.copy()
    for j in range(nbits - 1, i, -1):
      ctl.append([j + idx])
    qc.multi_control(ctl, i+idx, aux, ops.PauliX(), 'multi-0-X')


def experiment_incr():
  """Run a few incr experiments."""

  qc = circuit.qc('incr')
  qc.reg(4, 0)
  aux = qc.reg(4)

  for val in range(15):
    incr(qc, 0, 4, aux, [])

    maxbits, _ = qc.psi.maxprob()
    res = helper.bits2val(maxbits[0:4])
    if val + 1 != res:
      raise AssertionError('Invalid Result')


def experiment_decr():
  """Run a few decr experiments."""
  qc = circuit.qc('decr')
  qc.reg(4, 15)
  aux = qc.reg(4)

  for val in range(15, 0, -1):
    decr(qc, 0, 4, aux, [])

    maxbits, _ = qc.psi.maxprob()
    res = helper.bits2val(maxbits[0:4])
    if val - 1 != res:
      raise AssertionError('Invalid Result')


def incr_mod_9(qc, aux):
  """Increment-by-1 modulo 9 circuit."""

  # We achieve this with help of an ancilla:
  #
  #  -X------------ o  X  0
  #  -o--X--------- 0  |  0
  #  -o--o--X------ 0  |  0
  #  -o--o--o--X--- o  X  0
  #                 |  |  |
  #  needs an extra ancillary:
  #                 |  |  |
  #  ...            X--o--X  -> |0>
  #
  for i in range(4):
    ctl = []
    for j in range(4 - 1, i, -1):
      ctl.append(j)
    qc.multi_control(ctl, i, aux, ops.PauliX(), 'multi-X')

  qc.multi_control([0, [1], [2], 3], aux[4], aux, ops.PauliX(), 'multi-X')
  qc.cx(aux[4], 0)
  qc.cx(aux[4], 3)
  qc.multi_control([[0], [1], [2], [3]], aux[4], aux, ops.PauliX(), 'multi-X')


def experiment_mod_9():
  """Run a few incr-mod-9 experiments."""

  qc = circuit.qc('incr')
  qc.reg(4, 0)
  aux = qc.reg(5)  # extra aux

  for val in range(18):
    incr_mod_9(qc, aux)
    maxbits, _ = qc.psi.maxprob()
    res = helper.bits2val(maxbits[0:4])
    if ((val + 1) % 9) != res:
      raise AssertionError('Invalid Result')


def simple_walk():
  """Simple quantum walk, allowing initial experiments."""

  nbits = 8
  qc = circuit.qc('simple_walk')
  qc.reg(nbits, 0b10000000)
  aux = qc.reg(nbits, 0)
  coin = qc.reg(1, 0)

  for _ in range(32):
    # Using a Hadamard coin, others are possible, of course.
    qc.h(coin[0])
    incr(qc, 0, nbits, aux, [coin[0]])
    decr(qc, 0, nbits, aux, [[coin[0]]])

  # Find and print the non-zero amplitudes for all states
  for bits in helper.bitprod(nbits):
    idx_bits = bits
    for _ in range(nbits):
      idx_bits = idx_bits + (0,)
    idx_bits0 = idx_bits + (0,)

    # Printing bits0 only, this can be changed, of course.
    if qc.psi.ampl(*idx_bits0) > 1e-5:
      print(
          '{:5.1f} {:5.4f}'.format(
              float(helper.bits2val(bits)), qc.psi.ampl(*idx_bits0).real
          )
      )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Increment...')
  experiment_incr()

  print('Decrement...')
  experiment_decr()

  print('Increment mod 9...')
  experiment_mod_9()

  print('Simple Walk...')
  simple_walk()


if __name__ == '__main__':
  app.run(main)
