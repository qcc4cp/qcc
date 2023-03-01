# python3
import math
from absl.testing import absltest
import numpy as np

from src.lib import helper
from src.lib import ops
from src.lib import state


class HelpersTest(absltest.TestCase):

  def test_bits_converstions(self):
    bits = helper.val2bits(6, 3)
    self.assertEqual(bits, [1, 1, 0])

    val = helper.bits2val(bits)
    self.assertEqual(val, 6)

  def test_density_to_cartesian(self):
    """Test density to cartesian conversion."""

    q0 = state.zeros(1)
    rho = q0.density()
    x, y, z = helper.density_to_cartesian(rho)
    self.assertEqual(x, 0.0)
    self.assertEqual(y, 0.0)
    self.assertEqual(z, 1.0)

    q1 = state.ones(1)
    rho = q1.density()
    x, y, z = helper.density_to_cartesian(rho)
    self.assertEqual(x, 0.0)
    self.assertEqual(y, 0.0)
    self.assertEqual(z, -1.0)

    qh = ops.Hadamard()(q0)
    rho = qh.density()
    x, y, z = helper.density_to_cartesian(rho)
    self.assertTrue(math.isclose(np.real(x), 1.0, abs_tol=1e-6))
    self.assertTrue(math.isclose(np.real(y), 0.0))
    self.assertTrue(math.isclose(np.real(z), 0.0, abs_tol=1e-6))

    qr = ops.RotationZ(math.pi / 2)(qh)
    rho = qr.density()
    x, y, z = helper.density_to_cartesian(rho)
    self.assertTrue(math.isclose(np.real(x), 0.0, abs_tol=1e-6))
    self.assertTrue(math.isclose(np.real(y), 1.0, abs_tol=1e-6))
    self.assertTrue(math.isclose(np.real(z), 0.0, abs_tol=1e-6))

  def test_frac(self):
    self.assertEqual(helper.bits2frac((0,)), 0)
    self.assertEqual(helper.bits2frac((1,)), 0.5)
    self.assertEqual(helper.bits2frac((0, 1)), 0.25)
    self.assertEqual(helper.bits2frac((1, 0)), 0.5)
    self.assertEqual(helper.bits2frac((1, 1)), 0.75)

  def test_frac2bits(self):
    self.assertEqual(helper.frac2bits(0, 3), [0, 0, 0])
    self.assertEqual(helper.frac2bits(0.5, 3), [1, 0, 0])
    self.assertEqual(helper.frac2bits(0.25, 3), [0, 1, 0])
    self.assertEqual(helper.frac2bits(0.5 + 0.25, 3), [1, 1, 0])
    self.assertEqual(helper.frac2bits(0.5 + 0.5 / 2 - 0.5 / 4, 3), [1, 0, 1])
    self.assertEqual(helper.frac2bits(0.5 / 4, 3), [0, 0, 1])
    tval = 0.5 + 0.5 / 2 + 0.5 / 4 + 0.5 / 8 - 0.0001
    self.assertEqual(helper.frac2bits(tval, 8), [1, 1, 1, 0, 1, 1, 1, 1])
    val = helper.bits2frac([1, 1, 1, 0, 1, 1, 1, 1])
    self.assertLess(tval - val, 0.004)

  def test_pi_fractions(self) -> None:
    self.assertEqual(helper.pi_fractions(-3 * math.pi / 2), '-3*pi/2')
    self.assertEqual(helper.pi_fractions(3 * math.pi / 2), '3*pi/2')
    self.assertEqual(helper.pi_fractions(-3 * math.pi), '-3*pi')
    self.assertEqual(helper.pi_fractions(math.pi * 3), '3*pi')
    self.assertEqual(helper.pi_fractions(math.pi / 3), 'pi/3')
    self.assertEqual(helper.pi_fractions(math.pi / -2), '-pi/2')

  def test_qubit_to_bloch(self) -> None:
    psi = state.bitstring(0, 0)
    x, y, z = helper.qubit_to_bloch(psi)
    self.assertEqual(x, 0.0)
    self.assertEqual(y, 0.0)
    self.assertEqual(z, 1.0)


if __name__ == '__main__':
  absltest.main()
