"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
from inflammation.models import daily_max, daily_mean, daily_min

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

@pytest.mark.parametrize("test, expected", [
                                            ([[1,2],[5,9],[4,4],[8,2]], [8,9]),
                                            ([[12,2],[5,9],[4,4],[8,2]], [12,9])
                                              ])
def test_daily_max(test, expected):
    """Test that max function works for an array of positive integers."""
    npt.assert_array_equal(daily_max(test), expected)

@pytest.mark.parametrize("test, expected", [
                                            ([[1,2],[5,9],[4,4],[8,2]], [1,2]),
                                            ([[1,6],[2,7],[4,9],[1,4]], [1,4])
                                            ])
def test_daily_min(test, expected):
    """Test that daily min function works for an array of postive integers."""

    npt.assert_array_equal(daily_min(test), expected)


def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])
