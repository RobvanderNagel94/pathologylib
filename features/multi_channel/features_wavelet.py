import numpy as np


def bounded_variation(coefficients: np.ndarray,
                      axis: int) -> np.ndarray:
    """
    Calculate the bounded variation of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Bounded variation of the signal.
    """
    diffs = np.diff(coefficients, axis=axis)
    abs_sums = np.sum(np.abs(diffs), axis=axis)
    max_c = np.max(coefficients, axis=axis)
    min_c = np.min(coefficients, axis=axis)
    return np.divide(abs_sums, max_c - min_c)


def maximum(coefficients: np.ndarray,
            axis: int) -> np.ndarray:
    """
    Calculate the maximum of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Maximum of the signal.
    """
    return np.max(coefficients, axis=axis)


def mean(coefficients: np.ndarray,
         axis: int) -> np.ndarray:
    """
    Calculate the mean of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Mean of the signal.
    """
    return np.mean(coefficients, axis=axis)


def minimum(coefficients: np.ndarray,
            axis: int) -> np.ndarray:
    """
    Calculate the minimum of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Minimum of the signal.
    """
    return np.min(coefficients, axis=axis)


def power(coefficients: np.ndarray,
          axis: int) -> np.ndarray:
    """
    Calculate the power of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Power of the signal.
    """
    return np.sum(coefficients * coefficients, axis=axis)


def power_ratio(coefficients: np.ndarray,
                axis: int = -2) -> np.ndarray:
    """
    Calculate the power ratio of a signal.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Power ratio of the signal.
    """
    ratios = coefficients / np.sum(coefficients, axis=axis, keepdims=True)
    return ratios


def entropy(coefficients: np.ndarray, axis: int):
    """
    Computes the entropy of the given coefficients.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Entropy of the input coefficients.
    """
    return -1 * coefficients * np.log(coefficients)


def variance(coefficients: np.ndarray, axis: int):
    """
    Computes the variance of the given coefficients.

    :param coefficients: Signal coefficients.
    :param axis: Axis along which the operation is performed.

    :return: Variance of the input coefficients.
    """
    return np.var(coefficients, axis=axis)
