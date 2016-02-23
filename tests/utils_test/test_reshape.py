import unittest
import numpy as np
import nt.testing as tc
from nt.utils.numpy_utils import reshape


T, B, F = 400, 6, 513
A = np.random.uniform(size=(T, B, F))


class TestReshape(unittest.TestCase):
    def test_noop(self):
        assert reshape(A, 'T,B,F->T,B,F').shape == (T, B, F)
        assert reshape(A, 'T B F->T B F').shape == (T, B, F)

    def test_transpose(self):
        assert reshape(A, 'T,B,F->F,T,B').shape == (F, T, B)

    def test_broadcast(self):
        assert reshape(A, 'T,B,F->1,T,B,F').shape == (1, T, B, F)
        assert reshape(A, 'T,B,F->T,B,1,F').shape == (T, B, 1, F)
        assert reshape(A, 'T,B,F->T,B,F,1').shape == (T, B, F, 1)

    def test_reshape(self):
        assert reshape(A, 'T,B,F->T,B*F').shape == (T, B*F)
        assert reshape(A, 'T,B,F->T*B*F').shape == (T*B*F,)

    def test_reshape_and_broadcast(self):
        assert reshape(A, 'T,B,F->T,1,B*F').shape == (T, 1, B*F)
        tc.assert_equal(reshape(A, 'T,B,F->T,1,B*F').ravel(), A.ravel())
        assert reshape(A, 'T,B,F->1,T,1,B*F,1').shape == (1, T, 1, B*F, 1)

    def test_transpose_and_reshape(self):
        result = reshape(A, 'T,B,F->F,B*T')
        assert result.shape == (F, B*T)
        tc.assert_almost_equal(result, A.transpose(2, 1, 0).reshape(F, B*T))

    def test_all(self):
        assert reshape(A, 'T,B,F->F,1,B*T').shape == (F, 1, B*T)
