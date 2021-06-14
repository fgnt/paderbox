import pytest
from paderbox.array import interval

from paderbox.array.interval.util import (
    cy_non_intersection,
    cy_intersection,
    cy_parse_item,
    cy_str_to_intervals,
    cy_invert_intervals,
)


def test_cy_invert_intervals():
    assert cy_invert_intervals(((1, 2), (3, 4)), 5) == ((0, 1), (2, 3), (4, 5))
    assert cy_invert_intervals(((0, 3), (4, 5)), 5) == ((3, 4),)
    assert cy_invert_intervals(((0, 3), (4, 5)), 6) == ((3, 4), (5, 6))
    assert cy_invert_intervals(((1, 3), (4, 5)), 5) == ((0, 1), (3, 4))
    assert cy_invert_intervals((), 10) == ((0, 10),)


def test_cy_non_intersection():
    assert cy_non_intersection((0, 3), ((1, 2),)) == ()
    assert cy_non_intersection((1, 4), ((0, 2), (3, 5))) == ((0, 1), (4, 5))
    assert cy_non_intersection((1, 2), ((0, 3),)) == ((0, 1), (2, 3))


def test_cy_intersection():
    assert cy_intersection((0, 3), ((1, 2),)) == ((1, 2),)
    assert cy_intersection((1, 4), ((0, 2), (3, 5))) == ((1, 2), (3, 4))
    assert cy_intersection((1, 2), ((0, 3),)) == ((1, 2),)
    assert cy_intersection((4, 5), ((0, 3),)) == ()


def test_shape():
    ai = interval.zeros(1)
    assert isinstance(ai.shape, tuple)
    assert isinstance(ai.shape[0], int)
    interval.zeros((1,))
    assert isinstance(ai.shape, tuple)
    assert isinstance(ai.shape[0], int)
    interval.zeros([1, ])
    assert isinstance(ai.shape, tuple)
    assert isinstance(ai.shape[0], int)
    with pytest.raises(TypeError):
        interval.zeros('a')
    with pytest.raises(TypeError):
        interval.zeros({'num_samples': 42})
    with pytest.raises(TypeError):
        interval.zeros(('asdf', ))
    with pytest.raises(ValueError):
        interval.zeros((1, 2))
