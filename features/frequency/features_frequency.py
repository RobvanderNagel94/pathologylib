from numpy import (ndarray, max, min, sum, log, ptp, var)
from numpy import mean as _mean


def maximum(power_spectrum: ndarray, axis=-1):
    """
    This function returns the maximum value along the specified axis in the given power spectrum array.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Maximum value along the specified axis.
    """
    return max(power_spectrum, axis=axis)


def mean(power_spectrum: ndarray, axis=-1):
    """
    This function returns the mean value along the specified axis in the given power spectrum array.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Mean value along the specified axis.
    """
    return _mean(power_spectrum, axis=axis)


def minimum(power_spectrum: ndarray, axis=-1):
    """
    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Minimum value along the specified axis.
    """
    return min(power_spectrum, axis=axis)


def peak_frequency(power_spectrum: ndarray, axis=-1):
    """
    This function returns the index of the peak frequency along the specified axis in the given power spectrum array.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Peak frequency value along the specified axis.
    """
    return power_spectrum.argmax(axis=axis)


def power(power_spectrum: ndarray, axis=-1):
    """
    This function returns the sum of the squares along the specified axis in the given power spectrum array.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Sum of squares along the specified axis.
    """
    return sum(power_spectrum ** 2, axis=axis)


def power_ratio(power_spectrum: ndarray, axis=-1):
    """
    Compute the power ratio of a given power spectrum along the specified axis.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Power ratios of the power spectrum along the specified axis.
    """
    return power_spectrum / sum(power_spectrum, axis=axis, keepdims=True)


def spectral_entropy(power_spectrum: ndarray):
    """
    Compute the spectral entropy of a given power spectrum.

    :param power_spectrum: Input power spectrum

    :return: Spectral entropy of the power spectrum.
    """
    return -1 * power_spectrum * log(power_spectrum)


def value_range(power_spectrum: ndarray, axis=-1):
    """
    Compute the range of values of a given power spectrum along the specified axis.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Range of values of the power spectrum along the specified axis.
    """
    return ptp(power_spectrum, axis=axis)


def entropy(power_spectrum: ndarray, axis=-1) -> ndarray:
    """
    Computes the entropy of the given coefficients.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the entropy value is computed along the last axis.

    :return: Entropy of the input coefficients.
    """
    return -1 * power_spectrum * log(power_spectrum)


def variance(power_spectrum: ndarray, axis=-1):
    """
    Compute the variance of a given power spectrum along the specified axis.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Variance of the power spectrum along the specified axis.
    """
    return var(power_spectrum, axis=axis)
