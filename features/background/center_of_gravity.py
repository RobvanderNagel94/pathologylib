from features.frequency.welch import estimate_welch
from utils.dict_validation import _assert_valid_dict

from numpy import (ndarray, asarray, array, transpose)


def COG(eeg_dict: dict,
        fs: int = 250,
        window: str = 'hann',
        noverlap: int = 256,
        nfft: int = 512,
        nperseg: int = 512,
        fmin: int = 1,
        fmax: int = 18
        ) -> (ndarray, ndarray):
    """
    Computes left-right and anterior-posterior power distributions along the scalp using
    van Putten's center-of-gravity feature.

    :param eeg_dict: Dictionary of channels (keys) and arrays of signal values (values), passed per segment.
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param window: Window function, default 'hann'.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.
    :param fmin: Minimum frequency bound for power spectrum, default 1.
    :param fmax: Maximum frequency bound for power spectrum, default 12.

    :return: Power in left-right direction, weighted by channel positions.
    :return: Power in anterior-posterior direction, weighted by channel positions.

    References
    ----------
    [1] Van Putten (2007), The colorful brain: compact visualisation of routine EEG recordings.
    .. 10.1007/978-3-540-73044-6_127

    """

    _assert_valid_dict(eeg_dict=eeg_dict, fs=fs, noverlap=noverlap, nfft=nfft, nperseg=nperseg)

    channels = array(list(eeg_dict.keys()))

    Sxx = transpose(

        [estimate_welch(eeg_dict[ch], fs, window, noverlap, nfft, nperseg, fmin, fmax)[1] for ch in channels]

    )
    # Order channels = [Fp2 F8 T4 T6 O2 F4 C4 P4 Fz Cz Pz F3 C3 P3 Fp1 F7 T3 T5 O1]
    x = array([0, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, -2, -2, -2, -1])
    y = array([0, 1, 0, -1, -2, 1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, -1, -2])

    # Compute euclidean distances from center location Cz
    x_cog = (Sxx @ x) / Sxx.sum(axis=1)
    y_cog = (Sxx @ y) / Sxx.sum(axis=1)

    return x_cog, y_cog
