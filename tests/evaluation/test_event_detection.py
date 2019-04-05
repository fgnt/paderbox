import numpy as np
from paderbox.evaluation.event_detection import error_rate, fscore, \
    tune_decision_offset
import paderbox.testing as tc


def test_error_rate():
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]])
    er, substitutions, insertions, deletions = error_rate(
        a, b, substitution_axis=-1, sum_axis=(-2, -1)
    )
    tc.assert_equal(er, 1.)
    tc.assert_equal(insertions, 1.)
    tc.assert_equal(deletions, 1.)
    tc.assert_equal(substitutions, 1.)

    er, substitutions, insertions, deletions = error_rate(
        np.broadcast_to(a, (10, 3, 3)), np.broadcast_to(b, (10, 3, 3)),
        substitution_axis=-1, sum_axis=(-2, -1)
    )
    tc.assert_equal(er, 10*[1.])
    tc.assert_equal(insertions, 10*[1.])
    tc.assert_equal(deletions, 10*[1.])
    tc.assert_equal(substitutions, 10*[1.])

    er, substitutions, insertions, deletions = error_rate(
        a, b, substitution_axis=None, sum_axis=-2
    )
    tc.assert_equal(er, [3., 0., 1.])
    tc.assert_equal(insertions, [2, 0, 0])
    tc.assert_equal(deletions, [1, 0, 1])
    tc.assert_equal(substitutions, [0, 0, 0])

    er, substitutions, insertions, deletions = error_rate(
        np.broadcast_to(a, (10, 3, 3)), np.broadcast_to(b, (10, 3, 3)),
        substitution_axis=None, sum_axis=-2
    )
    tc.assert_equal(er, 10*[[3., 0., 1.]])
    tc.assert_equal(insertions, 10*[[2, 0, 0]])
    tc.assert_equal(deletions, 10*[[1, 0, 1]])
    tc.assert_equal(substitutions, 10*[[0, 0, 0]])


def test_fscore():
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]])
    f, p, r = fscore(
        a, b, sum_axis=(-2, -1)
    )
    tc.assert_equal(p, 1/3)
    tc.assert_equal(r, 1/3)
    tc.assert_equal(f, 1/3)

    f, p, r = fscore(
        np.broadcast_to(a, (10, 3, 3)), np.broadcast_to(b, (10, 3, 3)),
        sum_axis=(-2, -1)
    )
    tc.assert_equal(p, 10*[1/3])
    tc.assert_equal(r, 10*[1/3])
    tc.assert_equal(f, 10*[1/3])


def test_tune_decision_offset():
    a = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    b = np.array([[0.4, 0.1, 0.5],
                  [0.3, 0.2, 0.3],
                  [0.0, 0.1, 0.9]])
    offset = tune_decision_offset(a, b)
    tc.assert_almost_equal(offset, [-.35, -.15, -.7])
    offset = tune_decision_offset(
        a, b,
        metric_fn=lambda x, y, ax: error_rate(
            x, y, substitution_axis=None, sum_axis=ax
        )[0],
        maximize=False
    )
    tc.assert_almost_equal(offset, [-.35, -.15, -.7])

    a = np.tile(a, (20, 1))
    b = np.tile(b, (20, 1))
    offset = tune_decision_offset(
        a, b, max_candidate_offsets=10,
        metric_fn=lambda x, y, ax: error_rate(
            x, y, substitution_axis=None, sum_axis=ax
        )[0],
        maximize=False
    )
    tc.assert_almost_equal(offset, [-.35, -.15, -.7])

    offset = tune_decision_offset(
        a, b, max_candidate_offsets=10,
        metric_fn=lambda x, y, ax: error_rate(
            x, y, substitution_axis=None, sum_axis=ax
        )[0],
        maximize=False,
        micro_averaging=True
    )
    tc.assert_almost_equal(offset, [-.35, -.15, -.7])

    b = np.linspace(0., 1., 1001)
    a = np.zeros_like(b)
    a[490:] = 1.
    offset = tune_decision_offset(
        a[:, None], b[:, None], max_candidate_offsets=11,
        metric_fn=lambda x, y, ax: error_rate(
            x, y, substitution_axis=None, sum_axis=ax
        )[0],
        maximize=False,
        micro_averaging=True)
    tc.assert_almost_equal(offset, [-.4895])

    a[0:490:10] = 1.
    offset = tune_decision_offset(
        a[:, None], b[:, None], max_candidate_offsets=11,
        metric_fn=lambda x, y, ax: error_rate(
            x, y, substitution_axis=None, sum_axis=ax
        )[0],
        maximize=False,
        micro_averaging=True)
    tc.assert_almost_equal(offset, [-.4895])
