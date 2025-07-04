"""Tests for statistics functions within the Model layer."""

import math
import numpy as np
import numpy.testing as npt
import pytest
from inflammation.models import daily_max, daily_mean, daily_min
from inflammation.models import patient_normalise

@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [

        (
            'hello',
            None,
            TypeError,
        ),
        (
            3,
            None,
            TypeError,
        ),
        (
            [4, 5, 6],
            None,
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        )
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""
    if isinstance(test, list):
        test = np.array(test)
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            patient_normalise(test)

    else:
        result = patient_normalise(test)
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)

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
