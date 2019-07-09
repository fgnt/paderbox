import numpy as np
from paderbox.evaluation.event_detection import \
    error_rate, binomial_error_rate, fscore, binomial_fscore, lwlrap, \
    tune_decision_offset, get_candidate_thresholds
import paderbox.testing as tc


def test_error_rate():
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]])
    er, substitutions, insertions, deletions = error_rate(a, b)
    tc.assert_equal(er, 1.)
    tc.assert_equal(insertions, 1.)
    tc.assert_equal(deletions, 1.)
    tc.assert_equal(substitutions, 1.)

    er, substitutions, insertions, deletions = error_rate(
        np.broadcast_to(a, (10, 3, 3)), np.broadcast_to(b, (10, 3, 3))
    )
    tc.assert_equal(er, 10*[1.])
    tc.assert_equal(insertions, 10*[1.])
    tc.assert_equal(deletions, 10*[1.])
    tc.assert_equal(substitutions, 10*[1.])

    er, substitutions, insertions, deletions = error_rate(
        a, b, event_wise=True
    )
    tc.assert_equal(er, [3., 0., 1.])
    tc.assert_equal(insertions, [2, 0, 0])
    tc.assert_equal(deletions, [1, 0, 1])
    tc.assert_equal(substitutions, [0, 0, 0])

    er, substitutions, insertions, deletions = error_rate(
        np.broadcast_to(a, (10, 3, 3)), np.broadcast_to(b, (10, 3, 3)),
        event_wise=True
    )
    tc.assert_equal(er, 10*[[3., 0., 1.]])
    tc.assert_equal(insertions, 10*[[2, 0, 0]])
    tc.assert_equal(deletions, 10*[[1, 0, 1]])
    tc.assert_equal(substitutions, 10*[[0, 0, 0]])


def test_fscore():
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]])
    f, p, r = fscore(a, b)
    tc.assert_equal(p, 1/3)
    tc.assert_equal(r, 1/3)
    tc.assert_equal(f, 1/3)

    f, p, r = fscore(
        np.broadcast_to(a, (10, 3, 3)), np.broadcast_to(b, (10, 3, 3))
    )
    tc.assert_equal(p, 10*[1/3])
    tc.assert_equal(r, 10*[1/3])
    tc.assert_equal(f, 10*[1/3])


def test_get_thresholds():
    scores = np.linspace(0., 1., 1001)
    targets = np.zeros_like(scores)
    targets[490:] = 1.
    thresholds = get_candidate_thresholds(targets[:, None], scores[:, None])
    tc.assert_almost_equal(thresholds[0], [.4895])

    targets = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ])
    scores = np.array([
        [0.4, 0.1, 0.5], [0.3, 0.2, 0.7], [0.0, 0.1, 0.6]
    ])
    candidate_offsets = get_candidate_thresholds(targets, scores)
    for actual, expected in zip(
            candidate_offsets,
            [np.array([0.35]), np.array([0.15]), np.array([0.55, 0.8])]
    ):
        tc.assert_array_almost_equal(actual, expected)


def test_tune_decision_offset():
    targets = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ])
    scores = np.array([
        [0.4, 0.1, 0.5], [0.3, 0.2, 0.7], [0.0, 0.1, 0.6]
    ])
    candidate_offsets = [
        -thres for thres in get_candidate_thresholds(targets, scores)
    ]
    offset = tune_decision_offset(
        targets, scores, candidate_offsets=candidate_offsets
    )
    tc.assert_almost_equal(offset, np.array([-.35, -.15, -.55]))
    offset = tune_decision_offset(
        targets, scores, candidate_offsets=candidate_offsets,
        metric_fn=binomial_error_rate, maximize=False
    )
    tc.assert_almost_equal(offset, [-.35, -.15, -.55])

    targets = np.tile(targets, (20, 1))
    scores = np.tile(scores, (20, 1))
    offset = tune_decision_offset(
        targets, scores, candidate_offsets=candidate_offsets,
        metric_fn=binomial_error_rate, maximize=False
    )
    tc.assert_almost_equal(offset, [-.35, -.15, -.55])

    offset = tune_decision_offset(
        targets, scores, candidate_offsets=3*[np.linspace(-1., 1., 101)],
        metric_fn=lwlrap, maximize=True,
        fine_tune_iterations=5
    )
    tc.assert_almost_equal(offset, [0., 0.22, -0.28])


def test_lwlrap():

    num_samples = 100
    num_labels = 20

    truth = np.random.rand(num_samples, num_labels) > 0.5
    truth[0:1, :] = False  # Ensure at least some samples with no truth labels.

    scores = np.random.rand(num_samples, num_labels)
    lwlrap_ = lwlrap(truth, scores)
    per_class_lwlrap_ref, weight_per_class_ref = calculate_per_class_lwlrap(
        truth, scores
    )
    assert lwlrap_ == np.sum(per_class_lwlrap_ref * weight_per_class_ref)


# Reference implementation taken from https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=Xffu7w5t0YFa

# Core calculation of label precisions for one test sample.
def one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


# All-in-one calculation of per-class lwlrap.
def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros(
        (num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            one_sample_positive_class_precisions(scores[sample_num, :],
                                                 truth[sample_num, :]))
        precisions_for_samples_by_classes[
            sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class
