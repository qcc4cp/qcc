# python3
import math
from absl.testing import absltest
import numpy as np

from src.lib import bell
from src.lib import ops
from src.lib import state


class BellTest(absltest.TestCase):

  def test_bell(self):
    """Check successful entanglement."""

    b00 = bell.bell_state(0, 0)
    self.assertTrue(
        b00.is_close((state.zeros(2) + state.ones(2)) / math.sqrt(2))
    )

    # Note the order is reversed from pictorials.
    op_exp = ops.Cnot(0, 1) @ (ops.Hadamard() * ops.Identity())
    b00_exp = op_exp(state.zeros(2))
    self.assertTrue(b00.is_close(b00_exp))

  def test_not_pure(self):
    """Bell states are pure states."""

    for a in [0, 1]:
      for b in [0, 1]:
        b = bell.bell_state(a, b)
        self.assertTrue(b.density().is_pure())
        self.assertTrue(
            math.isclose(np.real(np.trace(b.density())), 1.0, abs_tol=1e-6)
        )

  def test_measure(self):
    b00 = bell.bell_state(0, 1)
    self.assertTrue(math.isclose(b00.prob(0, 1), 0.5, abs_tol=1e-6))
    self.assertTrue(math.isclose(b00.prob(1, 0), 0.5, abs_tol=1e-6))

    _, b00 = ops.Measure(b00, 0, tostate=0)
    self.assertTrue(math.isclose(b00.prob(0, 1), 1.0, abs_tol=1e-6))
    self.assertTrue(math.isclose(b00.prob(1, 0), 0.0, abs_tol=1e-6))

    # This state can't be measured, all zeros.
    _, b00 = ops.Measure(b00, 1, tostate=1)
    self.assertTrue(math.isclose(b00.prob(1, 0), 0.0, abs_tol=1e-6))

    b00 = bell.bell_state(0, 1)
    self.assertTrue(math.isclose(b00.prob(0, 1), 0.5, abs_tol=1e-6))
    self.assertTrue(math.isclose(b00.prob(1, 0), 0.5, abs_tol=1e-6))

    _, b00 = ops.Measure(b00, 0, tostate=1)
    self.assertTrue(math.isclose(b00.prob(0, 1), 0.0, abs_tol=1e-6))
    self.assertTrue(math.isclose(b00.prob(1, 0), 1.0, abs_tol=1e-6))

    # This state can't be measured, all zeros.
    p, _ = ops.Measure(b00, 1, tostate=1, collapse=False)
    self.assertEqual(p, 0.0)

  def test_measure_order(self):
    """Order of measurement must not make a difference."""

    b00 = bell.bell_state(0, 0)
    _, b00 = ops.Measure(b00, 0, tostate=0)
    _, b00 = ops.Measure(b00, 1, tostate=0)
    self.assertTrue(math.isclose(b00.prob(0, 0), 1.0))
    self.assertTrue(math.isclose(b00.prob(1, 1), 0.0))

    b00 = bell.bell_state(0, 0)
    _, b00 = ops.Measure(b00, 1, tostate=0)
    _, b00 = ops.Measure(b00, 0, tostate=0)
    self.assertTrue(math.isclose(b00.prob(0, 0), 1.0))
    self.assertTrue(math.isclose(b00.prob(1, 1), 0.0))

  def test_ghz(self):
    ghz = bell.ghz_state(3)
    self.assertGreater(ghz.prob(0, 0, 0), 0.49)
    self.assertGreater(ghz.prob(1, 1, 1), 0.49)

    ghz = bell.ghz_state(10)
    self.assertGreater(ghz.prob(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0.49)
    self.assertGreater(ghz.prob(1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 0.49)

  def test_bell_and_pauli(self):
    b00 = bell.bell_state(0, 0)

    bell_xz = ops.PauliX()(b00)
    bell_xz = ops.PauliZ()(bell_xz)

    bell_iy = (1j * ops.PauliY())(b00)

    self.assertTrue(np.allclose(bell_xz, bell_iy))

  def test_w_state(self):
    psi = bell.w_state()
    self.assertGreater(psi.prob(0, 0, 1), 0.3)
    self.assertGreater(psi.prob(0, 1, 0), 0.3)
    self.assertGreater(psi.prob(1, 0, 0), 0.3)


if __name__ == '__main__':
  absltest.main()
