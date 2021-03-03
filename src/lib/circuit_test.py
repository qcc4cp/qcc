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
      qc.h(i)
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
      qc.h(n)
      psi.apply(ops.Hadamard(), n)
      for i in range(0, 5):
        qc.cu1(n-(i+1), n, math.pi/float(2**(i+1)))
        psi.apply_controlled(ops.U1(math.pi/float(2**(i+1))), n-(i+1), n)
      for i in range(0, 5):
        qc.cu1(n-(i+1), n, -math.pi/float(2**(i+1)))
        psi.apply_controlled(ops.U1(-math.pi/float(2**(i+1))), n-(i+1), n)
      qc.h(n)
      psi.apply(ops.Hadamard(), n)

    if not psi.is_close(qc.psi):
      raise AssertionError('Numerical Problems')

  def test_circuit_of_circuit(self):
    c1 = circuit.qc('c1')
    c1.reg(6, 0)
    c1.x(0)
    c1.cx(1, 2)
    c2 = circuit.qc('c2', eager=False)
    c2.x(1)
    c2.cx(3, 1)
    c1.qc(c2, 0)
    c1.qc(c2, 1)
    c1.qc(c2, 2)
    self.assertEqual(8, c1.ir.ngates)

  def test_circuit_of_inv_circuit(self):
    c1 = circuit.qc('c1')
    c1.reg(6, 0)
    c1.x(0)
    c1.rx(1, math.pi/3)
    c1.h(1)
    c1.cz(3, 2)

    c2 = c1.inverse()
    c1.qc(c2, 0)
    self.assertEqual(8, c1.ir.ngates)

  def test_circuit_exec(self):
    c = circuit.qc('c', eager=False)
    c.reg(3, 0)
    c.h(0)
    c.h(1)
    c.h(2)

    c.run()
    self.assertEqual(3, c.ir.ngates)
    for i in range(8):
      self.assertGreater(c.psi[i], 0.3)

    c.run()
    self.assertEqual(3, c.ir.ngates)
    self.assertGreater(c.psi[0], 0.99)
    for i in range(1, 8):
      self.assertLess(c.psi[i], 0.01)

  def test_multi1(self):
    c = circuit.qc('multi', eager=False)
    comp = c.reg(6)
    aux = c.reg(6)
    ctl=[0, 1, 2, 3, 4]
    c.multi_control(ctl, 5, aux, ops.PauliX(), f'multi-x({ctl}, 5)')
    self.assertEqual(41, c.ir.ngates)

  def test_multi0(self):
    c = circuit.qc('multi', eager=True)
    comp = c.reg(4, (1, 0, 0, 1))
    aux = c.reg(4)
    ctl=[0, [1], [2]]
    c.multi_control(ctl, 3, aux, ops.PauliX(), f'multi-x({ctl}, 5)')
    self.assertGreater(c.psi.prob(1, 0, 0, 0, 0, 0, 0, 0), 0.99)

  def test_multi_n(self):
    c = circuit.qc('multi', eager=True)
    comp = c.reg(4, (1, 0, 0, 1))
    aux = c.reg(4)
    ctl = []
    c.multi_control(ctl, 3, aux, ops.PauliX(), f'single')
    self.assertGreater(c.psi.prob(1, 0, 0, 0, 0, 0, 0, 0), 0.99)

    ctl = [0]
    c.multi_control(ctl, 3, aux, ops.PauliX(), f'single')
    self.assertGreater(c.psi.prob(1, 0, 0, 1, 0, 0, 0, 0), 0.99)

    ctl = [1]
    c.multi_control(ctl, 3, aux, ops.PauliX(), f'single')
    self.assertGreater(c.psi.prob(1, 0, 0, 1, 0, 0, 0, 0), 0.99)

    ctl = [[1]]
    c.multi_control(ctl, 3, aux, ops.PauliX(), f'single')
    self.assertGreater(c.psi.prob(1, 0, 0, 0, 0, 0, 0, 0), 0.99)

    ctl = [0, [1], [2]]
    c.multi_control(ctl, 3, aux, ops.PauliX(), f'single')
    self.assertGreater(c.psi.prob(1, 0, 0, 1, 0, 0, 0, 0), 0.99)

if __name__ == '__main__':
  absltest.main()
