from utils.array_validation import (_assert_two_dimensional, _assert_one_dimensional)

import pytest
import pandas as pd
import numpy as np


@pytest.mark.parametrize(
    "iterable, expected_exception",
    [
        (pd.DataFrame({'col1': [1, 2, 3]}), None),
        (pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}), ValueError),
        (np.array([1, 2, 3]), None),
        (np.array([[1, 2], [3, 4]]), ValueError),
        (pd.Series([[1, 2], [3, 4]]), ValueError),
        (pd.Series([1, [2, 3], 4]), ValueError),
        (pd.Series([1, 2, 3]), None),
        ([1, 2, 3], None),
        ([[1, 2], [3, 4]], ValueError)
    ]
)
def test_assert_one_dimensional(iterable, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            _assert_one_dimensional(iterable)
    else:
        _assert_one_dimensional(iterable)


@pytest.mark.parametrize(
    "iterable, expected_exception",
    [
        (pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), ValueError),
        (np.array([1, 2, 3]), ValueError),
        (np.array([[1, 2], [3, 4]]), None),
        ([1, 2, 3], ValueError),
        ([[1, 2], [3, 4]], None),
        (pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}), None)
    ],
)
def test_assert_two_dimensional(iterable, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            _assert_two_dimensional(iterable)
    else:
        _assert_two_dimensional(iterable)

