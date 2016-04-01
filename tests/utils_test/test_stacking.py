import unittest
import numpy as np
from nt.utils.numpy_utils import stack_context, unstack_context

T, B, F = 400, 6, 513
A = np.random.uniform(size=(T, B, F)) + 1j * np.random.uniform(size=(T, B, F))


class TestUnstackContext(unittest.TestCase):
    def test_identity_operation(self):
        left_context = 2
        right_context = 3
        step_width = 1

        stacked = stack_context(
            A,
            left_context=left_context,
            right_context=right_context,
            step_width=step_width
        )

        unstacked = unstack_context(
            stacked,
            mode='center',
            left_context=left_context,
            right_context=right_context,
            step_width=step_width
        )

        np.testing.assert_allclose(unstacked, A)
