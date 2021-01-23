# python3

from absl.testing import absltest

from lib import tensor

class TensorTest(absltest.TestCase):

  def test_pow(self):
    t = tensor.Tensor([1.0, 1.0])
    self.assertLen(t.shape, 1)
    self.assertEqual(t.shape[0], 2)

    t0 = t ** 0.0
    self.assertEqual(t0, 1.0)

    t1 = t ** 1
    self.assertLen(t1.shape, 1)
    self.assertEqual(t1.shape[0], 2)

    t2 = t ** 2
    self.assertLen(t2.shape, 1)
    self.assertEqual(t2.shape[0], 4)

    m = tensor.Tensor([[1.0, 1.0], [1.0, 1.0]])
    self.assertLen(m.shape, 2)
    self.assertEqual(m.shape[0], 2)
    self.assertEqual(m.shape[1], 2)

    m0 = m ** 0.0
    self.assertEqual(m0, 1.0)

    m1 = m ** 1
    self.assertLen(m1.shape, 2)
    self.assertEqual(m1.shape[0], 2)
    self.assertEqual(m1.shape[1], 2)

    m2 = m ** 2
    self.assertLen(m2.shape, 2)
    self.assertEqual(m2.shape[0], 4)
    self.assertEqual(m2.shape[1], 4)

  def test_hermitian(self):
    t = tensor.Tensor([[2.0, 0.0], [0.0, 2.0]])
    self.assertTrue(t.is_hermitian())
    self.assertFalse(t.is_unitary())


if __name__ == '__main__':
  absltest.main()

