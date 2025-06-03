"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
from inflammation.models import daily_max, daily_mean, daily_min

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

@pytest.mark.parametrize("test, expected", [
                                            ([[1,2],[5,9],[4,4],[8,2]], [8,9])
                                              ])
def test_daily_max(test, expected):
    """Test that max function works for an array of positive integers."""
    npt.assert_array_equal(daily_max(test), expected)

@pytest.mark.parametrize("test, expected", [
                                            ([[1,2],[5,9],[4,4],[8,2]], [1,2])
                                            ])
def test_daily_min(test, expected):
    """Test that daily min function works for an array of postive integers."""

    npt.assert_array_equal(daily_min(test), expected)


def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])
