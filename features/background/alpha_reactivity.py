from features.frequency.welch import estimate_welch
from utils.number_validation import _assert_nfft

from numpy import (ndarray, asarray)
from typing import Union


def AlphaReactivity(x_open: Union[ndarray, list],
                             x_closed: Union[ndarray, list],
                             fs: int = 250,
                             window: str = 'hann',
                             noverlap: int = 256,
                             nfft: int = 512,
                             nperseg: int = 512,
                             pkf: float = 10) -> float:

    """
    Computes reactivity for the dominant peak frequency between the eyes closed and open states.

    :param x_open: Array of signal data from the O1 channel, annotated with the eyes open state of shape (n,).
    :param x_closed: Array of signal data from the O1 channel, annotated with the eyes closed state of shape (n,).
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param window: Window function, default 'hann'.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.
    :param pkf: Estimated peak frequency.

    :return: Quantified and normalised value for alpha power reactivity.

    Notes
    --------

    Using the estimated PDR peak value, the reactivity is calculated by
    constructing a 0.5 Hz frequency band on the estimated dominant frequency
    when the eyes are open and when the eyes are closed. Based on these values,
    a normalised value is found which quantifies the reactivity of the PDR.

    If Qreac > 0.5: Substantial reactivity.
    If 0.1 < Qreac < 0.5: low reactivity.
    If Qreac < 0.1: absent reactivity.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    x_open = asarray(x_open)
    x_closed = asarray(x_closed)

    if not (x_open.shape == (len(x_open),)) or (x_closed.shape == (len(x_closed),)):
        raise ValueError(f"input must have shape (n,) but (n,m) was given.")
    if (len(x_open) < 4 * fs) or (len(x_closed) < 4 * fs):
        raise ValueError(f"signal length for x_open = {x_open} and x_closed = {x_closed} "
                         f"must be at least four times the sampling frequency")
    if not (2. < pkf < 18.):
        raise ValueError(f"pkf = {pkf} must be between 2 and 18.")

    for array in [x_open, x_closed]:
        _assert_nfft(n=array.shape[1],
                     nfft=nfft,
                     nperseg=nperseg,
                     noverlap=noverlap)

    fmin = int(pkf - 0.5)
    fmax = int(pkf + 0.5)

    Sxx_closed = estimate_welch(x_closed, fs, window, noverlap, nfft, nperseg, fmin, fmax)[1]
    Sxx_open = estimate_welch(x_open, fs, window, noverlap, nfft, nperseg, fmin, fmax)[1]

    return float(1 - (Sxx_open.sum() / Sxx_closed.sum()))
