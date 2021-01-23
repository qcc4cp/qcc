# python3
import math

from absl.testing import absltest

from src.lib import circuit
from src.lib import ops
from src.lib import state


class CircuitTest(absltest.TestCase):

  def compare_to(self, psi, *bits):
    qc = circuit.qc()
    qc.bitstring(*bits)
    self.assertTrue(psi.is_close(qc.psi))

  def test_toffoli(self):
    qc = circuit.qc()
    qc.bitstring(1, 1, 0, 1, 1)
    qc.toffoli(0, 1, 3)
    self.compare_to(qc.psi, 1, 1, 0, 0, 1)

    qc = circuit.qc()
    qc.bitstring(1, 1, 0, 1, 1)
    qc.toffoli(4, 3, 0)
    self.compare_to(qc.psi, 0, 1, 0, 1, 1)

    qc = circuit.qc()
    qc.bitstring(1, 1, 0, 1, 1)
    qc.toffoli(1, 2, 3)
    self.compare_to(qc.psi, 1, 1, 0, 1, 1)

  def test_swap(self):
    qc = circuit.qc()
    qc.bitstring(1, 1, 0, 1, 1)
    qc.swap(1, 2)
    self.compare_to(qc.psi, 1, 0, 1, 1, 1)

  def test_cswap(self):
    qc = circuit.qc()
    qc.bitstring(1, 1, 0, 1, 1)
    qc.cswap(1, 2, 3)
    self.compare_to(qc.psi, 1, 1, 1, 0, 1)

    qc = circuit.qc()
    qc.bitstring(1, 1, 0, 1, 0)
    qc.cswap(2, 3, 4)
    self.compare_to(qc.psi, 1, 1, 0, 1, 0)

    qc = circuit.qc()
    qc.bitstring(1, 1, 0, 1, 0)
    qc.cswap(3, 2, 1)
    self.compare_to(qc.psi, 1, 0, 1, 1, 0)

  def test_acceleration(self):
    psi = state.bitstring(1, 0, 1, 0)
    qc = circuit.qc()
    qc.bitstring(1, 0, 1, 0)

    for i in range(4):
      qc.x(i)
      psi.apply(ops.PauliX(), i)
      qc.y(i)
      psi.apply(ops.PauliY(), i)
      qc.z(i)
      psi.apply(ops.PauliZ(), i)
      qc.had(i)
      psi.apply(ops.Hadamard(), i)
      if i:
        qc.cu1(0, i, 1.1)
        psi.apply_controlled(ops.U1(1.1), 0, i)

    if not psi.is_close(qc.psi):
      raise AssertionError('Numerical Problems')

    psi = state.bitstring(1, 0, 1, 0, 1)
    qc = circuit.qc()
    qc.bitstring(1, 0, 1, 0, 1)

    for n in range(5):
      qc.had(n)
      psi.apply(ops.Hadamard(), n)
      for i in range(0, 5):
        qc.cu1(n-(i+1), n, math.pi/float(2**(i+1)))
        psi.apply_controlled(ops.U1(math.pi/float(2**(i+1))), n-(i+1), n)
      for i in range(0, 5):
        qc.cu1(n-(i+1), n, -math.pi/float(2**(i+1)))
        psi.apply_controlled(ops.U1(-math.pi/float(2**(i+1))), n-(i+1), n)
      qc.had(n)
      psi.apply(ops.Hadamard(), n)

    if not psi.is_close(qc.psi):
      raise AssertionError('Numerical Problems')


if __name__ == '__main__':
  absltest.main()
