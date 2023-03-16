# python3
import math

from absl.testing import absltest
import numpy as np

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

  def test_arange(self):
    qc = circuit.qc()
    qc.arange(10)
    self.assertEqual(qc.psi[5], 5)

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

  def test_rotation(self):
    qc = circuit.qc()
    qc.bitstring(0)
    qc.rx(0, 2 * np.arcsin(0.5))
    self.assertEqual(qc.psi.prob(0), 0.75)
    self.assertEqual(qc.psi.prob(1), 0.25)

  def test_acceleration(self):
    psi = state.bitstring(1, 0, 1, 0)
    qc = circuit.qc()
    qc.bitstring(1, 0, 1, 0)

    for i in range(4):
      qc.x(i)
      psi.apply1(ops.PauliX(), i)
      qc.y(i)
      psi.apply1(ops.PauliY(), i)
      qc.z(i)
      psi.apply1(ops.PauliZ(), i)
      qc.h(i)
      psi.apply1(ops.Hadamard(), i)
      if i:
        qc.cu1(0, i, 1.1)
        psi.applyc(ops.U1(1.1), 0, i)

    if not psi.is_close(qc.psi):
      raise AssertionError('Numerical Problems')

    psi = state.bitstring(1, 0, 1, 0, 1)
    qc = circuit.qc()
    qc.bitstring(1, 0, 1, 0, 1)

    for n in range(5):
      qc.h(n)
      psi.apply1(ops.Hadamard(), n)
      for i in range(0, 5):
        qc.cu1(n - (i + 1), n, math.pi / float(2 ** (i + 1)))
        psi.applyc(ops.U1(math.pi / float(2 ** (i + 1))), n - (i + 1), n)
      for i in range(0, 5):
        qc.cu1(n - (i + 1), n, -math.pi / float(2 ** (i + 1)))
        psi.applyc(ops.U1(-math.pi / float(2 ** (i + 1))), n - (i + 1), n)
      qc.h(n)
      psi.apply1(ops.Hadamard(), n)

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
    c1.rx(1, math.pi / 3)
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
    c.reg(6)
    aux = c.reg(6)
    ctl = [0, 1, 2, 3, 4]
    c.multi_control(ctl, 5, aux, ops.PauliX(), f'multi-x({ctl}, 5)')
    self.assertEqual(41, c.ir.ngates)

  def test_multi0(self):
    c = circuit.qc('multi', eager=True)
    c.reg(4, (1, 0, 0, 1))
    aux = c.reg(4)
    ctl = [0, [1], [2]]
    c.multi_control(ctl, 3, aux, ops.PauliX(), f'multi-x({ctl}, 5)')
    self.assertGreater(c.psi.prob(1, 0, 0, 0, 0, 0, 0, 0), 0.99)

  def test_multi_n(self):
    c = circuit.qc('multi', eager=True)
    c.reg(4, (1, 0, 0, 1))
    aux = c.reg(4)
    ctl = []
    c.multi_control(ctl, 3, aux, ops.PauliX(), 'single')
    self.assertGreater(c.psi.prob(1, 0, 0, 0, 0, 0, 0, 0), 0.99)

    ctl = [0]
    c.multi_control(ctl, 3, aux, ops.PauliX(), 'single')
    self.assertGreater(c.psi.prob(1, 0, 0, 1, 0, 0, 0, 0), 0.99)

    ctl = [1]
    c.multi_control(ctl, 3, aux, ops.PauliX(), 'single')
    self.assertGreater(c.psi.prob(1, 0, 0, 1, 0, 0, 0, 0), 0.99)

    ctl = [[1]]
    c.multi_control(ctl, 3, aux, ops.PauliX(), 'single')
    self.assertGreater(c.psi.prob(1, 0, 0, 0, 0, 0, 0, 0), 0.99)

    ctl = [0, [1], [2]]
    c.multi_control(ctl, 3, aux, ops.PauliX(), 'single')
    self.assertGreater(c.psi.prob(1, 0, 0, 1, 0, 0, 0, 0), 0.99)

  def test_x_error_first_approach(self):
    error_qubit = 0

    qc = circuit.qc('x-flip / correction')
    qc.qubit(alpha=0.6)
    qc.reg(2, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)

    # insert error (index 0, 1, or 2)
    qc.x(error_qubit)

    syndrom = qc.reg(2, 0)
    qc.cx(0, syndrom[0])
    qc.cx(1, syndrom[0])
    qc.cx(1, syndrom[1])
    qc.cx(2, syndrom[1])

    # Measure syndrom qubit 3, 4:
    #   00  - nothing needs to be done
    #   01  - x(2)
    #   10  - x(0)
    #   11  - x(1)
    qc.x(error_qubit)
    p2, _ = qc.measure_bit(error_qubit, 0)
    self.assertTrue(np.allclose(p2, 0.36, atol=0.001))

  def test_x_error(self):
    qc = circuit.qc('x-flip / correction')
    qc.qubit(0.6)

    qc.reg(2, 0)
    qc.cx(0, 2)
    qc.cx(0, 1)
    self.assertTrue(np.allclose(qc.psi.prob(0, 0, 0), 0.36, atol=0.001))
    self.assertTrue(np.allclose(qc.psi.prob(1, 1, 1), 0.64, atol=0.001))

    qc.x(0)

    # Fix
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.ccx(1, 2, 0)
    p0, _ = qc.measure_bit(0, 0, collapse=False)
    self.assertTrue(np.allclose(p0, 0.36))
    p1, _ = qc.measure_bit(0, 1, collapse=False)
    self.assertTrue(np.allclose(p1, 0.64))

  def test_shor_9_qubit_correction(self):
    for i in range(9):
      qc = circuit.qc('shor-9')
      # print(f'Initialize qubit as 0.60|0> + 0.80|1>, error on qubit {i}')
      qc.qubit(0.6)
      qc.reg(8, 0)

      # Left Side.
      qc.cx(0, 3)
      qc.cx(0, 6)
      qc.h(0)
      qc.h(3)
      qc.h(6)
      qc.cx(0, 1)
      qc.cx(0, 2)
      qc.cx(3, 4)
      qc.cx(3, 5)
      qc.cx(6, 7)
      qc.cx(6, 8)

      # Error insertion, use x(i), y(i), or z(i)
      qc.x(i)

      # Fix.
      qc.cx(0, 1)
      qc.cx(0, 2)
      qc.ccx(1, 2, 0)

      qc.h(0)
      qc.cx(3, 4)
      qc.cx(3, 5)
      qc.ccx(4, 5, 3)
      qc.h(3)
      qc.cx(6, 7)
      qc.cx(6, 8)
      qc.ccx(7, 8, 6)
      qc.h(6)

      qc.cx(0, 3)
      qc.cx(0, 6)
      qc.ccx(6, 3, 0)

      prob0, _ = qc.measure_bit(0, 0, collapse=False)
      prob1, _ = qc.measure_bit(0, 1, collapse=False)
      self.assertTrue(math.isclose(math.sqrt(prob0), 0.6, abs_tol=0.001))
      self.assertTrue(math.isclose(math.sqrt(prob1), 0.8, abs_tol=0.001))

  def test_opt(self):
    def decr(qc, idx, nbits, aux, controller):
      for i in range(0, nbits):
        ctl = controller.copy()
        for j in range(nbits - 1, i, -1):
          ctl.append([j + idx])
      qc.multi_control(ctl, i + idx, aux, ops.PauliX(), 'multi-0-X')

    qc = circuit.qc('decr')
    qc.reg(4, 15)
    aux = qc.reg(4)

    for _ in range(15, 0, -1):
      decr(qc, 0, 4, aux, [])

    # print(qc.stats())
    qc.optimize()
    # print(qc.stats())

  def test_to_ctl_single(self):
    qc = circuit.qc('test')
    qc.reg(2)

    sc = qc.sub()
    sc.x(1)
    sc.ry(1, 1.0)
    sc.z(1)
    sc.control_by(0)

    qc.qc(sc)
    self.assertTrue(qc.ir.gates[0].is_ctl())
    self.assertTrue(qc.ir.gates[1].is_ctl())
    self.assertTrue(qc.ir.gates[2].is_ctl())

  def test_to_ctl_multi(self):
    qc = circuit.qc('test')
    qc.reg(3)

    sc = qc.sub()
    sc.cx(0, 1)
    sc.control_by(2)

    qc.qc(sc)
    self.assertLen(qc.ir.gates, 5)

  def test_to_ctl_qft(self):
    def qft(qc: circuit.qc, reg: state.Reg, n: int) -> None:
      qc.h(reg[n])
      for i in range(n):
        qc.cu1(reg[n - (i + 1)], reg[n], math.pi / float(2 ** (i + 1)))

    def make_qc(nbits: int, init_val: int):
      qc = circuit.qc('test')
      qc.reg(1, init_val)
      rg = qc.reg(nbits, 3)
      sc = qc.sub()
      for i in range(nbits):
        qft(sc, rg, i)
      return qc, sc

    qc0, sc = make_qc(2, 0)
    qc0.qc(sc)

    qc1, sc = make_qc(2, 1)
    sc.control_by(0)
    qc1.qc(sc)

    self.assertLess(abs(qc0.psi.ampl(0, 0, 0) - qc1.psi.ampl(1, 0, 0)), 1e-5)
    self.assertLess(abs(qc0.psi.ampl(0, 0, 1) - qc1.psi.ampl(1, 0, 1)), 1e-5)
    self.assertLess(abs(qc0.psi.ampl(0, 1, 0) - qc1.psi.ampl(1, 1, 0)), 1e-5)
    self.assertLess(abs(qc0.psi.ampl(0, 1, 1) - qc1.psi.ampl(1, 1, 1)), 1e-5)

  def test_state_constructor(self):
    psi = state.bitstring(0, 0)
    psi = ops.Hadamard()(psi)
    psi = ops.Cnot(0, 1)(psi)

    qc = circuit.qc('test')
    qc.state(psi)
    self.assertTrue(math.isclose(np.real(psi[0]), 1 / np.sqrt(2), abs_tol=1e-6))
    self.assertTrue(math.isclose(np.real(psi[1]), 0, abs_tol=1e-6))
    self.assertTrue(math.isclose(np.real(psi[2]), 0, abs_tol=1e-6))
    self.assertTrue(math.isclose(np.real(psi[3]), 1 / np.sqrt(2), abs_tol=1e-6))
    self.assertEqual(psi.nbits, 2)


if __name__ == '__main__':
  absltest.main()
