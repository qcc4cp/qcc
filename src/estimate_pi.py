# python3
"""Example: Estimate Pi via Phase Estimation."""


# This is a rather simple application of phase estimation. However,
# in order to understand this example, phase estimation must be
# understood first.
#
# Despite it being simple, it can be used effectively to estimate the
# accuracy of one-qubit operations on real machines, as shown in:
#  https://arxiv.org/abs/1912.12037
#
# In this implementation we only simulate the technique. A good explanation
# can also be found in Qiskit:
#   https://qiskit.org/textbook/ch-demos/piday-code.html


from absl import app
import numpy as np
from src.lib import circuit
from src.lib import helper


def run_experiment(nbits_phase):
  """Estimate Pi with nbits_phase qubits."""

  qc = circuit.qc('pi estimator')
  qclock = qc.reg(nbits_phase)
  qbit = qc.reg(1)

  # We perform phase estimation on the operator U1(1.0), which corresponds
  # to the matrix:
  #
  #   | 1.0    0.0        |
  #   | 0.0    exp(i phi) |
  #
  # and we set phi = 1.0.
  #
  # Phase estimation gives us the phase phi of an operator as the fraction:
  #    exp(2 Pi i phi)
  #
  # We did set phi to 1.0 above. So we know 2 Pi phi = 1.0 or
  #   Pi = 1 / (2 phi)
  #
  qc.h(qclock)
  qc.x(qbit)
  for inv in reversed(range(nbits_phase)):
    qc.cu1(qclock[inv], qbit[0], 2 ** (nbits_phase - inv -1))
  qc.inverse_qft(qclock)

  bits, _ = qc.psi.maxprob()
  theta = helper.bits2frac(bits[:nbits_phase][::-1])
  print(f'Pi Estimate: {1 / (2 * theta):.5f} (qubits: {nbits_phase:2d}) '
        f'Delta: {np.abs(1 / (2 * theta) - np.pi):.6f}')


# pylint: disable=unused-argument
def main(argv):
  print('Estimate Pi via phase estimation...')

  for nbits in range(5, 21):
    run_experiment(nbits)

if __name__ == '__main__':
  app.run(main)
