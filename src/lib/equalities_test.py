# python3
import math
from absl.testing import absltest
import numpy as np

from src.lib import bell
from src.lib import helper
from src.lib import ops
from src.lib import state


class EqualitiesTest(absltest.TestCase):

  def test_reversible_hadamard(self):
    """H*H = I."""

    h2 = ops.Hadamard(2)
    i2 = ops.Identity(2)
    self.assertTrue((h2 @ h2).is_close(i2))

  def test_s_gate(self):
    """S^2 == Z."""

    x = ops.Sgate() @ ops.Sgate()
    self.assertTrue(x.is_close(ops.PauliZ()))

  def test_t_gate(self):
    """T^2 == S."""

    s = ops.Tgate() @ ops.Tgate()
    self.assertTrue(s.is_close(ops.Phase()))

  def test_v_gate(self):
    """V^2 == X."""

    s = ops.Vgate() @ ops.Vgate()
    self.assertTrue(s.is_close(ops.PauliX()))

  def test_swap(self):
    """Swap(Swap) == I."""

    swap = ops.Swap(0, 3)
    self.assertTrue((swap @ swap).is_close(ops.Identity(4)))

  def test_controlled_z(self):
    """Exercise 4.18 in Nielson, Chuang, CZ(0, 1) == CZ(1, 0)."""

    z0 = ops.ControlledU(0, 1, ops.PauliZ())
    z1 = ops.ControlledU(1, 0, ops.PauliZ())
    self.assertTrue(z0.is_close(z1))

    z0 = ops.ControlledU(0, 7, ops.PauliZ())
    z1 = ops.ControlledU(7, 0, ops.PauliZ())
    self.assertTrue(z0.is_close(z1))

  def test_had_cnot_had(self):
    """Exercise 4.20 in Nielson, Chuang, H2.Cnot(0,1).H2==Cnot(1,0)."""

    h2 = ops.Hadamard(2)
    cnot = ops.Cnot(0, 1)
    op = h2(cnot(h2))
    self.assertTrue(op.is_close(ops.Cnot(1, 0)))

  def test_xyx(self):
    """Exercise 4.7 in Nielson, Chuang, XYX == -Y."""

    x = ops.PauliX()
    y = ops.PauliY()
    op = x(y(x))
    self.assertTrue(op.is_close(-1.0 * ops.PauliY()))

  def test_equalities(self):
    """Exercise 4.13 in Nielson, Chuang."""

    # Generate the Pauli and Hadamard matrices.
    _, x, y, z = ops.Pauli()
    h = ops.Hadamard()

    op = h(x(h))
    self.assertTrue(op.is_close(ops.PauliZ()))

    op = h(y(h))
    self.assertTrue(op.is_close(-1.0 * ops.PauliY()))

    op = h(z(h))
    self.assertTrue(op.is_close(ops.PauliX()))

    op = x(z)
    self.assertTrue(op.is_close(1.0j * ops.PauliY()))

  def test_global_phase(self):
    """Exercise 4.14 in Nielson, Chuang, HTH == phase*rotX(pi/4)."""

    h = ops.Hadamard()
    op = h(ops.Tgate()(h))

    # If equal up to a global phase, all values should be equal.
    phase = op / ops.RotationX(math.pi / 4)
    self.assertTrue(
        math.isclose(phase[0, 0].real, phase[0, 1].real, abs_tol=1e-6)
    )
    self.assertTrue(
        math.isclose(phase[0, 0].imag, phase[0, 1].imag, abs_tol=1e-6)
    )
    self.assertTrue(
        math.isclose(phase[0, 0].real, phase[1, 0].real, abs_tol=1e-6)
    )
    self.assertTrue(
        math.isclose(phase[0, 0].imag, phase[1, 0].imag, abs_tol=1e-6)
    )
    self.assertTrue(
        math.isclose(phase[0, 0].real, phase[1, 1].real, abs_tol=1e-6)
    )
    self.assertTrue(
        math.isclose(phase[0, 0].imag, phase[1, 1].imag, abs_tol=1e-6)
    )

  def test_double_rot(self):
    """Make sure rotations add up."""

    rx = ops.RotationX(45.0 / 180.0 * math.pi)
    self.assertTrue((rx @ rx).is_close(ops.RotationX(90.0 / 180.0 * math.pi)))

  def test_v_vdag_v(self):
    """Figure 4.8 Nielson, Chuang."""

    # Make Toffoli out of V = sqrt(X).
    #
    v = ops.Vgate()  # Could be any unitary, in principle!
    ident = ops.Identity()
    cnot = ops.Cnot(0, 1)

    o0 = ident * ops.ControlledU(1, 2, v)
    c2 = cnot * ident
    o2 = ident * ops.ControlledU(1, 2, v.adjoint())
    o4 = ops.ControlledU(0, 2, v)
    final = o4 @ c2 @ o2 @ c2 @ o0

    v2 = v @ v
    cv1 = ops.ControlledU(1, 2, v2)
    cv0 = ops.ControlledU(0, 1, cv1)

    self.assertTrue(final.is_close(cv0))

  def test_control_equalities(self):
    """Exercise 4.31 Nielson, Chung."""

    i, x, y, z = ops.Pauli()
    x1 = x * i
    x2 = i * x
    y1 = y * i
    y2 = i * y
    z1 = z * i
    z2 = i * z
    c = ops.Cnot(0, 1)
    theta = 25.0 * math.pi / 180.0
    rx2 = i * ops.RotationX(theta)
    rz1 = ops.RotationZ(theta) * i

    self.assertTrue(c(x1(c)).is_close(x1(x2)))
    self.assertTrue((c @ x1 @ c).is_close(x1 @ x2))
    self.assertTrue((c @ y1 @ c).is_close(y1 @ x2))
    self.assertTrue((c @ z1 @ c).is_close(z1))
    self.assertTrue((c @ x2 @ c).is_close(x2))
    self.assertTrue((c @ y2 @ c).is_close(z1 @ y2))
    self.assertTrue((c @ z2 @ c).is_close(z1 @ z2))
    self.assertTrue((rz1 @ c).is_close(c @ rz1))
    self.assertTrue((rx2 @ c).is_close(c @ rx2))

  def test_partial(self):
    """Test partial trace."""

    psi = bell.bell_state(0, 0)
    reduced = ops.TraceOut(psi.density(), [0])
    self.assertTrue(math.isclose(np.real(np.trace(reduced)), 1.0, abs_tol=1e-6))
    self.assertTrue(math.isclose(np.real(reduced[0, 0]), 0.5, abs_tol=1e-6))
    self.assertTrue(math.isclose(np.real(reduced[1, 1]), 0.5, abs_tol=1e-6))

    q0 = state.qubit(alpha=0.5)
    q1 = state.qubit(alpha=0.8660254)
    psi = q0 * q1
    reduced = ops.TraceOut(psi.density(), [0])
    self.assertTrue(math.isclose(np.real(np.trace(reduced)), 1.0))
    self.assertTrue(math.isclose(np.real(reduced[0, 0]), 0.75, abs_tol=1e-6))
    self.assertTrue(math.isclose(np.real(reduced[1, 1]), 0.25, abs_tol=1e-6))

    reduced = ops.TraceOut(psi.density(), [1])
    self.assertTrue(math.isclose(np.real(np.trace(reduced)), 1.0))
    self.assertTrue(math.isclose(np.real(reduced[0, 0]), 0.25, abs_tol=1e-6))
    self.assertTrue(math.isclose(np.real(reduced[1, 1]), 0.75, abs_tol=1e-6))

    psi = q1 * q0 * state.qubit(alpha=(0.8660254))
    reduced = ops.TraceOut(psi.density(), [2])
    reduced = ops.TraceOut(reduced, [0])
    self.assertTrue(math.isclose(np.real(np.trace(reduced)), 1.0))
    self.assertTrue(math.isclose(np.real(reduced[0, 0]), 0.25))
    self.assertTrue(math.isclose(np.real(reduced[1, 1]), 0.75))

    psi = q1 * q0 * state.qubit(alpha=(0.8660254))
    reduced = ops.TraceOut(psi.density(), [0, 2])
    self.assertTrue(math.isclose(np.real(np.trace(reduced)), 1.0))
    self.assertTrue(math.isclose(np.real(reduced[0, 0]), 0.25))
    self.assertTrue(math.isclose(np.real(reduced[1, 1]), 0.75))

  def test_bloch_coords(self):
    psi = state.bitstring(1, 1)
    psi = ops.Qft(2)(psi)

    rho0 = ops.TraceOut(psi.density(), [1])
    rho1 = ops.TraceOut(psi.density(), [0])

    x0, _, _ = helper.density_to_cartesian(rho0)
    _, y1, _ = helper.density_to_cartesian(rho1)

    self.assertTrue(math.isclose(-1.0, x0, abs_tol=0.01))
    self.assertTrue(math.isclose(-1.0, y1, abs_tol=0.01))


if __name__ == '__main__':
  absltest.main()
