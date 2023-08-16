from numpy import (ndarray, diff, abs, max, min, sum, divide, log, var)
from numpy import mean as _mean


def bounded_variation(coefficients: ndarray, axis: int) -> ndarray:
    """
    Calculate the bounded variation of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Bounded variation of the signal.
    """
    diffs = diff(coefficients, axis=axis)
    abs_sums = sum(abs(diffs), axis=axis)
    max_c = max(coefficients, axis=axis)
    min_c = min(coefficients, axis=axis)
    return divide(abs_sums, max_c - min_c)


def maximum(coefficients: ndarray, axis: int) -> ndarray:
    """
    Calculate the maximum of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Maximum of the signal.
    """
    return max(coefficients, axis=axis)


def mean(coefficients: ndarray, axis: int) -> ndarray:
    """
    Calculate the mean of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Mean of the signal.
    """
    return _mean(coefficients, axis=axis)


def minimum(coefficients: ndarray, axis: int) -> ndarray:
    """
    Calculate the minimum of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Minimum of the signal.
    """
    return min(coefficients, axis=axis)


def power(coefficients: ndarray, axis: int) -> ndarray:
    """
    Calculate the power of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Power of the signal.
    """
    return sum(coefficients * coefficients, axis=axis)


def power_ratio(coefficients: ndarray, axis: int = -2) -> ndarray:
    """
    Calculate the power ratio of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Power ratio of the signal.
    """
    return coefficients / sum(coefficients, axis=axis, keepdims=True)


def entropy(coefficients: ndarray, axis: int) -> ndarray:
    """
    Computes the entropy of the given coefficients.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Entropy of the input coefficients.
    """
    return -1 * coefficients * log(coefficients)


def variance(coefficients: ndarray, axis: int) -> ndarray:
    """
    Computes the variance of the given coefficients.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Variance of the input coefficients.
    """
    return var(coefficients, axis=axis)
