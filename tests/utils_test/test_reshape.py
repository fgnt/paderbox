import unittest
import numpy as np
import nt.testing as tc
from nt.utils.numpy_utils import reshape


T, B, F = 40, 6, 51
A = np.random.uniform(size=(T, B, F))
A2 = np.random.uniform(size=(T, 1, B, F))
A3 = np.random.uniform(size=(T*B*F,))
A4 = np.random.uniform(size=(T, 1, 1, B, 1, F))


class TestReshape(unittest.TestCase):
    def test_noop_comma(self):
        result = reshape(A, 'T,B,F->T,B,F')
        tc.assert_equal(result.shape, (T, B, F))
        tc.assert_equal(result, A)

    def test_noop_space(self):
        result = reshape(A, 'T B F->T B F')
        tc.assert_equal(result.shape, (T, B, F))
        tc.assert_equal(result, A)

    def test_noop_mixed(self):
        result = reshape(A, 'tbf->t, b f')
        tc.assert_equal(result.shape, (T, B, F))
        tc.assert_equal(result, A)

    def test_transpose_comma(self):
        result = reshape(A, 'T,B,F->F,T,B')
        tc.assert_equal(result.shape, (F, T, B))
        tc.assert_equal(result, A.transpose(2, 0, 1))

    def test_transpose_mixed(self):
        result = reshape(A, 't, b, f -> f t b')
        tc.assert_equal(result.shape, (F, T, B))
        tc.assert_equal(result, A.transpose(2, 0, 1))

    def test_broadcast_axis_0(self):
        result = reshape(A, 'T,B,F->1,T,B,F')
        tc.assert_equal(result.shape, (1, T, B, F))
        tc.assert_equal(result, A[None, ...])

    def test_broadcast_axis_2(self):
        result = reshape(A, 'T,B,F->T,B,1,F')
        tc.assert_equal(result.shape, (T, B, 1, F))
        tc.assert_equal(result, A[..., None, :])

    def test_broadcast_axis_3(self):
        result = reshape(A, 'T,B,F->T,B,F,1')
        tc.assert_equal(result.shape, (T, B, F, 1))
        tc.assert_equal(result, A[..., None])

    def test_reshape_comma(self):
        result = reshape(A, 'T,B,F->T,B*F')
        tc.assert_equal(result.shape, (T, B*F))
        tc.assert_equal(result, A.reshape(T, B*F))

    def test_reshape_comma_unflatten(self):
        with tc.assert_raises(NotImplementedError):
            reshape(A3, 't*b*f->t, b, f')

    def test_reshape_comma_unflatten_and_transpose_and_flatten(self):
        with tc.assert_raises(NotImplementedError):
            reshape(A3, 't*b*f->f, t*b')

    def test_reshape_comma_flat(self):
        result = reshape(A, 'T,B,F->T*B*F')
        tc.assert_equal(result.shape, (T*B*F,))
        tc.assert_equal(result, A.ravel())

    def test_reshape_comma_with_singleton_input(self):
        result = reshape(A2, 'T, 1, B, F -> T*B*F')
        tc.assert_equal(result.shape, (T*B*F,))
        tc.assert_equal(result, A2.ravel())

    def test_reshape_comma_with_a_lot_of_singleton_inputs(self):
        result = reshape(A4, 'T, 1, 1, B, 1, F -> T*B*F')
        tc.assert_equal(result.shape, (T*B*F,))
        tc.assert_equal(result, A4.ravel())

    def test_reshape_and_broadcast(self):
        tc.assert_equal(reshape(A, 'T,B,F->T,1,B*F').shape, (T, 1, B*F))
        tc.assert_equal(reshape(A, 'T,B,F->T,1,B*F').ravel(), A.ravel())

    def test_reshape_and_broadcast_many(self):
        result = reshape(A, 'T,B,F->1,T,1,B*F,1')
        tc.assert_equal(result.shape, (1, T, 1, B*F, 1))

    def test_swap_and_reshape(self):
        result = reshape(A, 'T,B,F->T,F*B')
        tc.assert_equal(result.shape, (T, F * B))
        tc.assert_equal(result, A.swapaxes(-1, -2).reshape(T, F * B))

    def test_transpose_and_reshape(self):
        result = reshape(A, 'T,B,F->F,B*T')
        tc.assert_equal(result.shape, (F, B*T))
        tc.assert_equal(result, A.transpose(2, 1, 0).reshape(F, B*T))

    def test_all_comma(self):
        tc.assert_equal(reshape(A, 'T,B,F->F,1,B*T').shape, (F, 1, B*T))

    def test_all_space(self):
        tc.assert_equal(reshape(A, 't b f -> f1b*t').shape, (F, 1, B*T))

    def test_ellipsis_3(self):
        tc.assert_equal(reshape(A, '...->...').shape, (T, B, F))

    def test_ellipsis_2(self):
        tc.assert_equal(reshape(A, '...F->...F').shape, (T, B, F))

    def test_ellipsis_2_begin(self):
        tc.assert_equal(reshape(A, 'T...->T...').shape, (T, B, F))

    def test_ellipsis_2_letter_conflict(self):
        tc.assert_equal(reshape(A, 'a...->a...').shape, (T, B, F))

    def test_ellipsis_1(self):
        tc.assert_equal(reshape(A, '...BF->...FB').shape, (T, F, B))

    def test_ellipsis_1_begin(self):
        tc.assert_equal(reshape(A, 'TB...->BT...').shape, (B, T, F))

    def test_ellipsis_1_mid(self):
        tc.assert_equal(reshape(A, 'T...F->F...T').shape, (F, B, T))

    def test_ellipsis_0(self):
        tc.assert_equal(reshape(A, '...TBF->...TFB').shape, (T, F, B))

    def test_ellipsis_0_begin(self):
        tc.assert_equal(reshape(A, 'TBF...->TFB...').shape, (T, F, B))
