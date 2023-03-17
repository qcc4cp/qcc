# python3
import math
import random
from absl.testing import absltest
import numpy as np

from src.lib import helper
from src.lib import ops
from src.lib import state


class OpsTest(absltest.TestCase):

  def test_id(self):
    identity = ops.Identity()
    self.assertEqual(identity[0, 0], 1)
    self.assertEqual(identity[0, 1], 0)
    self.assertEqual(identity[1, 0], 0)
    self.assertEqual(identity[1, 1], 1)

  def test_unitary(self):
    self.assertTrue(ops.PauliX().is_unitary())
    self.assertTrue(ops.PauliY().is_unitary())
    self.assertTrue(ops.PauliZ().is_unitary())
    self.assertTrue(ops.Identity().is_unitary())

  def test_double_hadamard(self):
    """Check that Hadamard is fully reversible."""

    psi = state.zeros(2)

    psi2 = ops.Hadamard(2)(ops.Hadamard(2)(psi))
    self.assertEqual(psi2.nbits, 2)
    self.assertTrue(psi.is_close(psi2))

    combo = ops.Hadamard(2) @ ops.Hadamard(2)
    psi3 = combo(psi)
    self.assertEqual(psi3.nbits, 2)
    self.assertTrue(psi.is_close(psi3))
    self.assertTrue(psi.density().is_pure())

  def test_cnot(self):
    """Check implementation of ControlledU via Cnot."""

    psi = state.bitstring(0, 1)
    psi2 = ops.Cnot(0, 1)(psi)
    self.assertTrue(psi.is_close(psi2))

    psi2 = ops.Cnot(0, 1)(state.bitstring(1, 1))
    self.assertTrue(psi2.is_close(state.bitstring(1, 0)))

    psi2 = ops.Cnot(0, 3)(state.bitstring(1, 0, 0, 0, 1))
    self.assertTrue(psi2.is_close(state.bitstring(1, 0, 0, 1, 1)))

    psi2 = ops.Cnot(4, 0)(state.bitstring(1, 0, 0, 0, 1))
    self.assertTrue(psi2.is_close(state.bitstring(0, 0, 0, 0, 1)))

  def test_cnot0(self):
    """Check implementation of ControlledU via Cnot0."""

    # Check operator itself.
    x = ops.PauliX() * ops.Identity()
    self.assertTrue(ops.Cnot0(0, 1).is_close(x @ ops.Cnot(0, 1) @ x))

    # Compute simplest case with Cnot0.
    psi = state.bitstring(1, 0)
    psi2 = ops.Cnot0(0, 1)(psi)
    self.assertTrue(psi.is_close(psi2))

    # Compute via explicit constrution.
    psi2 = (x @ ops.Cnot(0, 1) @ x)(psi)
    self.assertTrue(psi.is_close(psi2))

    # Different offsets.
    psi2 = ops.Cnot0(0, 1)(state.bitstring(0, 1))
    self.assertTrue(psi2.is_close(state.bitstring(0, 0)))

    psi2 = ops.Cnot0(0, 3)(state.bitstring(0, 0, 0, 0, 1))
    self.assertTrue(psi2.is_close(state.bitstring(0, 0, 0, 1, 1)))

    psi2 = ops.Cnot0(4, 0)(state.bitstring(1, 0, 0, 0, 0))
    self.assertTrue(psi2.is_close(state.bitstring(0, 0, 0, 0, 0)))

  def test_controlled_controlled(self):
    """Toffoli gate over 4 qubits to verify that controlling works."""

    cnot = ops.Cnot(0, 3)
    toffoli = ops.ControlledU(0, 1, cnot)
    self.assertTrue(toffoli.is_close(ops.Toffoli(0, 1, 4)))

    psi = toffoli(state.bitstring(0, 1, 0, 0, 1))
    self.assertTrue(psi.is_close(state.bitstring(0, 1, 0, 0, 1)))

    psi = toffoli(state.bitstring(1, 1, 0, 0, 1))
    self.assertTrue(psi.is_close(state.bitstring(1, 1, 0, 0, 0)))

    psi = toffoli(state.bitstring(0, 0, 1, 1, 0, 0, 1), idx=2)
    self.assertTrue(psi.is_close(state.bitstring(0, 0, 1, 1, 0, 0, 0)))

  def test_swap(self):
    """Test swap gate, various indices."""

    swap = ops.Swap(0, 4)
    psi = swap(state.bitstring(1, 0, 1, 0, 0))
    self.assertTrue(psi.is_close(state.bitstring(0, 0, 1, 0, 1)))

    swap = ops.Swap(2, 0)
    psi = swap(state.bitstring(1, 0, 0))
    self.assertTrue(psi.is_close(state.bitstring(0, 0, 1)))

    op_manual = ops.Identity().kpow(2) * swap * ops.Identity()
    psi = op_manual(state.bitstring(1, 1, 0, 1, 1, 0))
    self.assertTrue(psi.is_close(state.bitstring(1, 1, 1, 1, 0, 0)))

    psi = swap(state.bitstring(1, 1, 0, 1, 1, 0), idx=2)
    self.assertTrue(psi.is_close(state.bitstring(1, 1, 1, 1, 0, 0)))

  def test_t_gate(self):
    """Test that T^2 == S."""

    t = ops.Tgate()
    self.assertTrue(t(t).is_close(ops.Phase()))

  def test_v_gate(self):
    """Test that V^2 == X."""

    t = ops.Vgate()
    self.assertTrue(t(t).is_close(ops.PauliX()))
    self.assertTrue(t(t.adjoint()).is_close(ops.Identity()))
    self.assertTrue(t.is_unitary())

  def test_yroot_gate(self):
    """Test that Yroot^2 == Y."""

    t = ops.Yroot()
    self.assertTrue(t(t).is_close(ops.PauliY()))

  def check_rotation(self, angle):
    # Note that RotationZ rotates by theta/2
    psi = ops.RotationZ(math.pi / 180.0 * angle)(state.zeros(1))
    self.assertTrue(math.isclose(-angle / 2, psi.phase(0), abs_tol=1e-5))

  def test_phase(self):
    psi = state.zeros(1)
    psi = ops.RotationZ(math.pi / 2)(psi)
    phase = psi.phase(0)

    # Note that Rotation rotates by theta/2.
    self.assertTrue(math.isclose(phase, -45.0, abs_tol=1e-6))

    # Test all other angles, check for sign flips.
    for i in range(360):
      self.check_rotation(float(i) / 2)
    for i in range(360):
      self.check_rotation(float(-i) / 2)

  def test_rk(self):
    rk0 = ops.Rk(0)
    self.assertTrue(rk0.is_close(ops.Identity()))

    rk1 = ops.Rk(1)
    self.assertTrue(rk1.is_close(ops.PauliZ()))

    rk2 = ops.Rk(2)
    self.assertTrue(rk2.is_close(ops.Sgate()))

    rk3 = ops.Rk(3)
    self.assertTrue(rk3.is_close(ops.Tgate()))

    for idx in range(8):
      psi = state.zeros(2)
      psi = (ops.Rk(idx).kpow(2) @ ops.Rk(-idx).kpow(2))(psi)
      self.assertTrue(psi.is_close(state.zeros(2)))

  def test_qft(self):
    """Build 'manually' a 3 qubit gate, Nielsen/Chuang Box 5.1."""

    h = ops.Hadamard()

    op = ops.Identity(3)
    op = op(h, 0)
    op = op(ops.ControlledU(1, 0, ops.Rk(2)), 0)  # S-gate
    op = op(ops.ControlledU(2, 0, ops.Rk(3)), 0)  # T-gate
    op = op(h, 1)
    op = op(ops.ControlledU(1, 0, ops.Rk(2)), 1)  # S-gate
    op = op(h, 2)
    op = op(ops.Swap(0, 2), 0)

    op3 = ops.Qft(3)
    self.assertTrue(op3.is_close(op))

  def test_qft_adjoint(self):
    bits = [0, 1, 0, 1, 1, 0]
    psi = state.bitstring(*bits)
    psi = ops.Qft(6)(psi)
    psi = ops.Qft(6).adjoint()(psi)
    maxbits, _ = psi.maxprob()
    self.assertEqual(maxbits, bits)

  def test_qft_hadamard(self):
    # For a state |00...0>, applying QFT or applying
    # all Hadamard gates must be identical.
    psi = state.zeros(5)
    psi = ops.Qft(5)(psi)

    phi = state.zeros(5)
    for i in range(5):
      phi = ops.Hadamard()(phi, i)
    for i in range(len(phi)):
      if phi[i] != psi[i]:
        raise AssertionError('Incorrect QFT vs Hadamards.')

  def test_padding(self):
    ident = ops.Identity(3)
    h = ops.Hadamard()

    op = ident(h, 0)
    op_manual = h * ops.Identity(2)
    self.assertTrue(op.is_close(op_manual))
    op = ident(h, 1)
    op_manual = ops.Identity() * h * ops.Identity()
    self.assertTrue(op.is_close(op_manual))
    op = ident(h, 2)
    op_manual = ops.Identity(2) * h
    self.assertTrue(op.is_close(op_manual))

    ident = ops.Identity(4)
    cx = ops.Cnot(0, 1)

    op = ident(cx, 0)
    op_manual = cx * ops.Identity(2)
    self.assertTrue(op.is_close(op_manual))
    op = ident(cx, 1)
    op_manual = ops.Identity(1) * cx * ops.Identity(1)
    self.assertTrue(op.is_close(op_manual))
    op = ident(cx, 2)
    op_manual = ops.Identity(2) * cx
    self.assertTrue(op.is_close(op_manual))

  def test_controlled_rotations(self):
    psi = state.bitstring(1, 1, 1)
    psi02 = ops.ControlledU(0, 2, ops.Rk(1))(psi)
    psi20 = ops.ControlledU(2, 0, ops.Rk(1))(psi)
    self.assertTrue(psi02.is_close(psi20))

  def test_bloch(self):
    psi = state.zeros(1)
    x, y, z = helper.density_to_cartesian(psi.density())
    self.assertEqual(x, 0.0)
    self.assertEqual(y, 0.0)
    self.assertEqual(z, 1.0)

    psi = ops.PauliX()(psi)
    x, y, z = helper.density_to_cartesian(psi.density())
    self.assertEqual(x, 0.0)
    self.assertEqual(y, 0.0)
    self.assertEqual(z, -1.0)

    psi = ops.Hadamard()(psi)
    x, y, z = helper.density_to_cartesian(psi.density())
    self.assertTrue(math.isclose(x, -1.0, abs_tol=1e-6))
    self.assertTrue(math.isclose(y, 0.0, abs_tol=1e-6))
    self.assertTrue(math.isclose(z, 0.0, abs_tol=1e-6))

  def test_rk_u1(self):
    for i in range(10):
      u1 = ops.U1(2 * math.pi / (2**i))
      rk = ops.Rk(i)
      self.assertTrue(u1.is_close(rk))

  def test_hh(self):
    p1 = state.qubit(alpha=random.random())
    x1 = state.qubit(alpha=random.random())
    psi = p1 * x1
    # inner product of full state
    self.assertTrue(np.allclose(np.inner(psi.conj(), psi), 1.0))

    # inner product of the constituents multiplied
    self.assertTrue(
        np.allclose(np.inner(p1.conj(), p1) * np.inner(x1.conj(), x1), 1.0)
    )

  def test_u(self):
    val = random.random()
    self.assertTrue(np.allclose(ops.U3(0, 0, val), ops.U1(val)))
    self.assertTrue(np.allclose(ops.U3(np.pi / 2, 0, np.pi), ops.Hadamard()))
    self.assertTrue(np.allclose(ops.U3(0, 0, 0), ops.Identity()))
    self.assertTrue(np.allclose(ops.U3(np.pi, 0, np.pi), ops.PauliX()))
    self.assertTrue(
        np.allclose(ops.U3(np.pi, np.pi / 2, np.pi / 2), ops.PauliY())
    )
    self.assertTrue(np.allclose(ops.U3(0, 0, np.pi), ops.PauliZ()))
    self.assertTrue(
        np.allclose(ops.U3(val, -np.pi / 2, np.pi / 2), ops.RotationX(val))
    )
    self.assertTrue(np.allclose(ops.U3(val, 0, 0), ops.RotationY(val)))

  def test_diffusion_op(self):
    nbits = 3
    op = ops.Hadamard(nbits)
    op = op @ ops.PauliX(nbits)
    cz = ops.ControlledU(1, 2, ops.PauliZ())
    czz = ops.ControlledU(0, 1, cz)
    op = op @ czz
    op = op @ ops.PauliX(nbits)
    op = op @ ops.Hadamard(nbits)
    self.assertTrue(np.allclose(op[0, 0], 1 - 2 / (2**nbits), atol=0.001))
    self.assertTrue(np.allclose(op[0, 1], -2 / (2**nbits), atol=0.001))

  def test_two_qubit_qft(self):
    for q1 in range(2):
      for q0 in range(2):
        psi = state.bitstring(q1, q0)
        psi = ops.Hadamard()(psi)
        psi = ops.ControlledU(0, 1, ops.Sgate())(psi)
        psi1 = ops.Hadamard()(psi, 1)

        psi = state.bitstring(q1, q0)
        psi = ops.Hadamard()(psi)
        psi = ops.ControlledU(1, 0, ops.Sgate())(psi)
        psi2 = ops.Hadamard()(psi, 1)
        self.assertTrue(psi1.is_close(psi2))

  def test_ctrl_phase(self):
    for gate in [
        ops.Sgate(),
        ops.Tgate(),
        ops.Rk(1),
        ops.Rk(2),
        ops.U1(random.random()),
        ops.U1(random.random()),
    ]:
      op01 = ops.ControlledU(2, 5, gate)
      op10 = ops.ControlledU(5, 2, gate)
      self.assertTrue(op01.is_close(op10))

  def test_rho(self):
    for _ in range(100):
      q = state.qubit(alpha=random.random())
      rho = q.density()
      ident, x, y, z = ops.Pauli()
      u = (rho + x @ rho @ x + y @ rho @ y + z @ rho @ z) / 2
      self.assertTrue(np.allclose(u, ident))


if __name__ == '__main__':
  absltest.main()
