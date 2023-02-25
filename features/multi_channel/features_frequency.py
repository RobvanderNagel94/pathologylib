import numpy as np


def maximum(power_spectrum: np.ndarray, axis=-1):
    """
    This function returns the maximum value along the specified axis in the given power spectrum array.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Maximum value along the specified axis.
    """
    return np.max(power_spectrum, axis=axis)


def mean(power_spectrum: np.ndarray, axis=-1):
    """
    This function returns the mean value along the specified axis in the given power spectrum array.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Mean value along the specified axis.
    """
    return np.mean(power_spectrum, axis=axis)


def minimum(power_spectrum: np.ndarray, axis=-1):
    """
    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Minimum value along the specified axis.
    """
    return np.min(power_spectrum, axis=axis)


def peak_frequency(power_spectrum: np.ndarray, axis=-1):
    """
    This function returns the index of the peak frequency along the specified axis in the given power spectrum array.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Peak frequency value along the specified axis.
    """
    return power_spectrum.argmax(axis=axis)


def power(power_spectrum: np.ndarray, axis=-1):
    """
    This function returns the sum of the squares along the specified axis in the given power spectrum array.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Sum of squares along the specified axis.
    """
    return np.sum(power_spectrum ** 2, axis=axis)


def power_ratio(power_spectrum: np.ndarray, axis=-1):
    """
    Compute the power ratio of a given power spectrum along the specified axis.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Power ratios of the power spectrum along the specified axis.
    """
    ratios = power_spectrum / np.sum(power_spectrum, axis=axis, keepdims=True)
    return ratios


def spectral_entropy(power_spectrum: np.ndarray):
    """
    Compute the spectral entropy of a given power spectrum.

    :param power_spectrum: Input power spectrum

    :return: Spectral entropy of the power spectrum.
    """
    return -1 * power_spectrum * np.log(power_spectrum)


def value_range(power_spectrum: np.ndarray, axis=-1):
    """
    Compute the range of values of a given power spectrum along the specified axis.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Range of values of the power spectrum along the specified axis.
    """
    return np.ptp(power_spectrum, axis=axis)


def variance(power_spectrum: np.ndarray, axis=-1):
    """
    Compute the variance of a given power spectrum along the specified axis.

    :param power_spectrum: Input power spectrum
    :param axis: Axis along which the operation is performed.
        Default is -1, which means the maximum value is computed along the last axis.

    :return: Variance of the power spectrum along the specified axis.
    """
    return np.var(power_spectrum, axis=axis)
