from utils.number_validation import _assert_nfft
from numpy import (ndarray, asarray)
from typing import Union
from scipy.signal import welch


def estimate_welch(x: Union[ndarray, list],
                   fs: float = 250,
                   window: str = 'hann',
                   noverlap: int = 256,
                   nfft: int = 512,
                   nperseg: int = 512,
                   fmin: int = 1,
                   fmax: int = 100
                   ) -> (ndarray, ndarray):

    """
    Estimates power spectral density using Welch's averaged periodogram method.
    The method splits the series into overlapping segments, computes periodograms for each segment,
    and then averages the periodograms.

    :param x: Array of signal values with shape (n,).
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param window: Window function, default 'hann'.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.
    :param fmin: Minimum frequency bound for power spectrum, default 1.
    :param fmax: Maximum frequency bound for power spectrum, default 100.

    :return: (bounded frequency spectrum, bounded power spectral density).

    Raises
    ----------
    AssertionError
        if input x is not of shape (n,)
        if fmin and fmax are between 1 and 100
        if fmax < fmin

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    """

    x = asarray(x)

    if not (1 <= fmin <= 100) and (1 <= fmax <= 100):
        raise ValueError(f"fmin and fmax must be between 1 and 100.")
    if not fmax > fmin:
        raise ValueError(f"fmax = {fmax} must be greater than fmin = {fmin}.")
    if not x.shape == (len(x),):
        raise ValueError(f"input must have shape (n,) but (n,m) was given.")

    _assert_nfft(n=len(x),
                 nfft=nfft,
                 nperseg=nperseg,
                 noverlap=noverlap)

    f, Sxx = welch(x,
                   fs=fs,
                   window=window,
                   noverlap=noverlap,
                   nfft=nfft,
                   nperseg=nperseg
                   )

    f_lim = f[(f >= fmin) & (f <= fmax)]
    start = f.tolist().index(f_lim[0])
    end = f.tolist().index(f_lim[-1]) + 1

    return f[start:end], Sxx[start:end]
