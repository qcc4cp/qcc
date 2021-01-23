# python3
import math

from absl.testing import absltest

from src.lib import ops
from src.lib import state


class MeasureTest(absltest.TestCase):

  def test_measure(self):
    psi = state.zeros(2)
    psi = ops.Hadamard()(psi)
    psi = ops.Cnot(0, 1)(psi)

    p0, psi2 = ops.Measure(psi, 0)
    self.assertTrue(math.isclose(p0, 0.5, abs_tol=1e-5))

    # Measure again - now state should have collapsed.
    p0, _ = ops.Measure(psi2, 0)
    self.assertTrue(math.isclose(p0, 1.0, abs_tol=1e-6))


if __name__ == '__main__':
  absltest.main()
