import numpy as np
import pandas as pd
from typing import Union

from core.constants import MONTAGE_1020

from numpy import ndarray


def validate_input(func):
    """ Decorator function """
    def wrapper(self, signals, channels, annotations, fs, age, id, gender):
        if not isinstance(signals, ndarray) or signals.ndim != 2:
            raise ValueError("Signals must be a 2D numpy array")
        if not isinstance(channels, ndarray) or not all(isinstance(c, str) for c in channels):
            raise ValueError("Channels must be a array of strings")
        if not set(channels) == set(MONTAGE_1020):
            raise ValueError("Channels must match the MONTAGE_1020")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling frequency must be a positive number")
        if id is not None and not isinstance(id, str):
            raise ValueError("Subject ID must be a string")
        if gender is not None and gender not in ['m', 'f']:
            raise ValueError("Gender must be either 'm' or 'f'")
        if age is not None and not isinstance(age, int):
            raise ValueError("Age must be an int")
        if not isinstance(annotations, ndarray) or not all(isinstance(a, str) for a in annotations):
            raise ValueError("Annotations must be an array of strings")
        if len(channels) != signals.shape[1]:
            raise ValueError("Number of channels must match the second dimension of signals")
        if len(annotations) != signals.shape[0]:
            raise ValueError("Number of annotations must match the first dimension of signals")
        if signals.shape[0] < 4 * fs:
            raise ValueError("Signals length must be at least four times the sampling frequency")

        return func(self, signals, channels, annotations, fs, age, id, gender)

    return wrapper


def _assert_two_dimensional(iterable: Union[pd.DataFrame, np.ndarray, list]) -> None:
    """
    Ensures the argument is two-dimensional.

    :param iterable: Any list-like type of which we want to see whether it is two-dimensional.

    :raises: ValueError when the given `iterable` is not two-dimensional.
    :raises: TypeError when the type of `iterable` is not expected.

    """

    error_msg = "The given argument is {}-dimensional instead of 2-dimensional."

    if isinstance(iterable, pd.DataFrame):
        if len(iterable.columns) != 2:
            raise ValueError(error_msg.format(len(iterable.columns)))
    elif isinstance(iterable, np.ndarray) or isinstance(iterable, pd.DatetimeIndex):
        if len(iterable.shape) != 2:
            raise ValueError(error_msg.format(len(iterable.shape)))
    elif isinstance(iterable, list):
        if not any(isinstance(i, list) or isinstance(i, np.ndarray) for i in iterable):
            raise ValueError(error_msg.format(1))
    else:
        raise TypeError(
            "Wrong type: {}. Cannot assess number of dimensions.".format(type(iterable))
        )


def _assert_one_dimensional(iterable: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> None:
    """
    Ensures the argument is one-dimensional.

    :param iterable: Any list-like type of which we want to see whether it
        is one-dimensional.

    :raises: ValueError when the given `iterable` is not one-dimensional.
    :raises: TypeError when the type of the given `iterable` is not expected.

    """
    error_msg = "The given argument is {}-dimensional instead of 1-dimensional."

    if isinstance(iterable, pd.DataFrame):
        if len(iterable.columns) != 1:
            raise ValueError(error_msg.format(len(iterable.columns)))
    elif isinstance(iterable, np.ndarray) or isinstance(iterable, pd.DatetimeIndex):
        if len(iterable.shape) != 1:
            raise ValueError(error_msg.format(len(iterable.shape)))
    elif isinstance(iterable, list):
        if any(isinstance(i, list) or isinstance(i, np.ndarray) for i in iterable):
            raise ValueError(error_msg.format(2))
    elif isinstance(iterable, pd.Series):
        length = iterable.size
        new_length = iterable.copy().explode().size
        if length != new_length:
            raise ValueError(error_msg.format(2))
    else:
        raise TypeError(
            "Wrong type: {}. Cannot assess number of columns.".format(type(iterable))
        )
