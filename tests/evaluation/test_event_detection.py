import numpy as np
from paderbox.evaluation.event_detection import hard_metrics, tune_decision_threshold
import paderbox.testing as tc


def test_hard_metrics():
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]])
    metrics_dict = hard_metrics(a, b)
    tc.assert_equal(metrics_dict['error_rate'], 1.)
    tc.assert_equal(metrics_dict['insertions'], 1.)
    tc.assert_equal(metrics_dict['deletions'], 1.)
    tc.assert_equal(metrics_dict['substitutions'], 1.)

    metrics_dict = hard_metrics(
        np.broadcast_to(a, (10, 3, 3)), np.broadcast_to(b, (10, 3, 3))
    )
    tc.assert_equal(metrics_dict['error_rate'], 10*[1.])
    tc.assert_equal(metrics_dict['insertions'], 10*[1.])
    tc.assert_equal(metrics_dict['deletions'], 10*[1.])
    tc.assert_equal(metrics_dict['substitutions'], 10*[1.])

    metrics_dict = hard_metrics(a, b, event_wise=True)
    tc.assert_equal(metrics_dict['error_rate'], [3., 0., 1.])
    tc.assert_equal(metrics_dict['insertions'], [2, 0, 0])
    tc.assert_equal(metrics_dict['deletions'], [1, 0, 1])
    tc.assert_equal(metrics_dict['substitutions'], [0, 0, 0])

    metrics_dict = hard_metrics(
        np.broadcast_to(a, (10, 3, 3)),
        np.broadcast_to(b, (10, 3, 3)),
        event_wise=True
    )
    tc.assert_equal(metrics_dict['error_rate'], 10*[[3., 0., 1.]])
    tc.assert_equal(metrics_dict['insertions'], 10*[[2, 0, 0]])
    tc.assert_equal(metrics_dict['deletions'], 10*[[1, 0, 1]])
    tc.assert_equal(metrics_dict['substitutions'], 10*[[0, 0, 0]])


def test_tune_decision_threshold():
    a = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    b = np.array([[0.4, 0.1, 0.5],
                  [0.3, 0.2, 0.3],
                  [0.0, 0.1, 0.9]])
    threshold = tune_decision_threshold(a, b)
    tc.assert_almost_equal(threshold, [.35, .15, .7])
    threshold = tune_decision_threshold(a, b, event_wise=True)
    tc.assert_almost_equal(threshold, [.35, .15, .7])

    a = np.tile(a, (20, 1))
    b = np.tile(b, (20, 1))
    threshold = tune_decision_threshold(a, b, max_thresholds=10)
    tc.assert_almost_equal(threshold, [.35, .15, .7])
    threshold = tune_decision_threshold(
        a, b, event_wise=True, max_thresholds=10)
    tc.assert_almost_equal(threshold, [.35, .15, .7])

    b = np.linspace(0., 1., 1001)
    a = np.zeros_like(b)
    a[490:] = 1.
    threshold = tune_decision_threshold(
        a[:, None], b[:, None], event_wise=True, max_thresholds=11)
    tc.assert_almost_equal(threshold, [.4895])

    a[0:490:10] = 1.
    threshold = tune_decision_threshold(
        a[:, None], b[:, None], event_wise=True, max_thresholds=11)
    tc.assert_almost_equal(threshold, [.4895])
