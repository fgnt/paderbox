import unittest
import numpy as np
import nt.testing as tc
from nt.utils.numpy_utils import morph


T, B, F = 40, 6, 50
A = np.random.uniform(size=(T, B, F))
A2 = np.random.uniform(size=(T, 1, B, F))
A3 = np.random.uniform(size=(T*B*F,))
A4 = np.random.uniform(size=(T, 1, 1, B, 1, F))


class TestMorph(unittest.TestCase):
    def test_noop_comma(self):
        result = morph('T,B,F->T,B,F', A)
        tc.assert_equal(result.shape, (T, B, F))
        tc.assert_equal(result, A)

    def test_noop_space(self):
        result = morph('T B F->T B F', A)
        tc.assert_equal(result.shape, (T, B, F))
        tc.assert_equal(result, A)

    def test_noop_mixed(self):
        result = morph('tbf->t, b f', A)
        tc.assert_equal(result.shape, (T, B, F))
        tc.assert_equal(result, A)

    def test_transpose_comma(self):
        result = morph('T,B,F->F,T,B', A)
        tc.assert_equal(result.shape, (F, T, B))
        tc.assert_equal(result, A.transpose(2, 0, 1))

    def test_transpose_mixed(self):
        result = morph('t, b, f -> f t b', A)
        tc.assert_equal(result.shape, (F, T, B))
        tc.assert_equal(result, A.transpose(2, 0, 1))

    def test_broadcast_axis_0(self):
        result = morph('T,B,F->1,T,B,F', A)
        tc.assert_equal(result.shape, (1, T, B, F))
        tc.assert_equal(result, A[None, ...])

    def test_broadcast_axis_2(self):
        result = morph('T,B,F->T,B,1,F', A)
        tc.assert_equal(result.shape, (T, B, 1, F))
        tc.assert_equal(result, A[..., None, :])

    def test_broadcast_axis_3(self):
        result = morph('T,B,F->T,B,F,1', A)
        tc.assert_equal(result.shape, (T, B, F, 1))
        tc.assert_equal(result, A[..., None])

    def test_reshape_comma(self):
        result = morph('T,B,F->T,B*F', A)
        tc.assert_equal(result.shape, (T, B*F))
        tc.assert_equal(result, A.reshape(T, B*F))

    def test_reshape_comma_unflatten(self):
        result = morph('t*b*f->tbf', A3, t=T, b=B)
        tc.assert_equal(result.shape, (T, B, F))
        tc.assert_equal(result, A3.reshape((T, B, F)))

    def test_reshape_comma_unflatten_and_transpose_and_flatten(self):
        result = morph('t*b*f->f, t*b', A3, f=F, t=T)
        tc.assert_equal(result.shape, (F, T*B))
        tc.assert_equal(result, A3.reshape((T*B, F)).transpose((1, 0)))

    def test_reshape_comma_flat(self):
        result = morph('T,B,F->T*B*F', A)
        tc.assert_equal(result.shape, (T*B*F,))
        tc.assert_equal(result, A.ravel())

    def test_reshape_comma_with_singleton_input(self):
        result = morph('T, 1, B, F -> T*B*F', A2)
        tc.assert_equal(result.shape, (T*B*F,))
        tc.assert_equal(result, A2.ravel())

    def test_reshape_and_broadcast(self):
        tc.assert_equal(morph('T,B,F->T,1,B*F', A).shape, (T, 1, B*F))
        tc.assert_equal(morph('T,B,F->T,1,B*F', A).ravel(), A.ravel())

    def test_reshape_and_broadcast_many(self):
        result = morph('T,B,F->1,T,1,B*F,1', A)
        tc.assert_equal(result.shape, (1, T, 1, B*F, 1))

    def test_swap_and_reshape(self):
        result = morph('T,B,F->T,F*B', A)
        tc.assert_equal(result.shape, (T, F * B))
        tc.assert_equal(result, A.swapaxes(-1, -2).reshape(T, F * B))

    def test_transpose_and_reshape(self):
        result = morph('T,B,F->F,B*T', A)
        tc.assert_equal(result.shape, (F, B*T))
        tc.assert_equal(result, A.transpose(2, 1, 0).reshape(F, B*T))

    def test_transpose_capital(self):
        result = morph('tbB->tBb', A)
        tc.assert_equal(result.shape, (T, F, B))
        tc.assert_equal(result, A.transpose(0, 2, 1))

    def test_all_comma(self):
        tc.assert_equal(morph('T,B,F->F,1,B*T', A).shape, (F, 1, B*T))

    def test_all_space(self):
        tc.assert_equal(morph('t b f -> f1b*t', A).shape, (F, 1, B*T))

    def test_ellipsis_3(self):
        tc.assert_equal(morph('...->...', A).shape, (T, B, F))

    def test_ellipsis_2(self):
        tc.assert_equal(morph('...F->...F', A).shape, (T, B, F))

    def test_ellipsis_2_begin(self):
        tc.assert_equal(morph('T...->T...', A).shape, (T, B, F))

    def test_ellipsis_2_letter_conflict(self):
        tc.assert_equal(morph('a...->a...', A).shape, (T, B, F))

    def test_ellipsis_1(self):
        tc.assert_equal(morph('...BF->...FB', A).shape, (T, F, B))

    def test_ellipsis_1_begin(self):
        tc.assert_equal(morph('TB...->BT...', A).shape, (B, T, F))

    def test_ellipsis_1_mid(self):
        tc.assert_equal(morph('T...F->F...T', A).shape, (F, B, T))

    def test_ellipsis_0(self):
        tc.assert_equal(morph('...TBF->...TFB', A).shape, (T, F, B))

    def test_ellipsis_0_begin(self):
        tc.assert_equal(morph('TBF...->TFB...', A).shape, (T, F, B))

    def test_ellipsis_expand_0(self):
        tc.assert_equal(
            morph(
                'a*b...->ab...',
                A,
                a=T // 2,
                b=2
            ).shape, (T // 2, 2, B, F))

    def test_ellipsis_expand_1(self):
        tc.assert_equal(
            morph(
                '...a*b->...ab',
                A,
                a=F // 2,
                b=2
            ).shape, (T, B, F // 2, 2))

    def test_reduce_mean(self):
        tc.assert_equal(
            morph(
                '...F->...',
                A,
                reduce=np.mean
            ).shape, (T, B))
        tc.assert_equal(
            morph(
                '...F->...',
                A,
                reduce=np.mean
            ), np.mean(A, axis=-1))

    def test_reduce_median(self):
        tc.assert_equal(
            morph(
                '...F->...',
                A,
                reduce=np.median
            ).shape, (T, B))
        tc.assert_equal(
            morph(
                '...F->...',
                A,
                reduce=np.median
            ), np.median(A, axis=-1))

    def test_reduce_sum(self):
        tc.assert_equal(
            morph(
                '...F->...',
                A,
                reduce=np.sum
            ).shape, (T, B))
        tc.assert_equal(
            morph(
                '...F->...',
                A,
                reduce=np.sum
            ), np.sum(A, axis=-1))
