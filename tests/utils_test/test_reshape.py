import unittest
import numpy as np
import nt.testing as tc
from nt.utils.numpy_utils import reshape


T, B, F = 400, 6, 513
A = np.random.uniform(size=(T, B, F))
B = np.random.uniform(size=(T, 1, B, F))


class TestReshape(unittest.TestCase):
    def test_noop_comma(self):
        assert reshape(A, 'T,B,F->T,B,F').shape == (T, B, F)

    def test_noop_space(self):
        assert reshape(A, 'T B F->T B F').shape == (T, B, F)

    def test_noop_mixed(self):
        assert reshape(A, 't b f->t, b f').shape == (T, B, F)

    def test_transpose_comma(self):
        assert reshape(A, 'T,B,F->F,T,B').shape == (F, T, B)

    def test_transpose_mixed(self):
        assert reshape(A, 't, b, f -> f t b').shape == (F, T, B)

    def test_broadcast_axis_0(self):
        assert reshape(A, 'T,B,F->1,T,B,F').shape == (1, T, B, F)

    def test_broadcast_axis_2(self):
        assert reshape(A, 'T,B,F->T,B,1,F').shape == (T, B, 1, F)

    def test_broadcast_axis_3(self):
        assert reshape(A, 'T,B,F->T,B,F,1').shape == (T, B, F, 1)

    def test_reshape_comma(self):
        assert reshape(A, 'T,B,F->T,B*F').shape == (T, B*F)

    def test_reshape_comma_flat(self):
        assert reshape(A, 'T,B,F->T*B*F').shape == (T*B*F,)
        assert reshape(B, 'T, 1, B, F -> T*B*F').shape == (T*B*F,)

    def test_reshape_and_broadcast(self):
        assert reshape(A, 'T,B,F->T,1,B*F').shape == (T, 1, B*F)
        tc.assert_equal(reshape(A, 'T,B,F->T,1,B*F').ravel(), A.ravel())

    def test_reshape_and_broadcast_many(self):
        assert reshape(A, 'T,B,F->1,T,1,B*F,1').shape == (1, T, 1, B*F, 1)

    def test_transpose_and_reshape(self):
        result = reshape(A, 'T,B,F->F,B*T')
        assert result.shape == (F, B*T)
        tc.assert_almost_equal(result, A.transpose(2, 1, 0).reshape(F, B*T))

    def test_all_comma(self):
        assert reshape(A, 'T,B,F->F,1,B*T').shape == (F, 1, B*T)

    def test_all_space(self):
        assert reshape(A, 't b f -> F 1 B*T').shape == (F, 1, B*T)
