from utils.number_validation import _assert_nfft
from core.constants import MONTAGE_1020

from numpy import asarray


def _assert_valid_dict(eeg_dict: dict,
                       fs: int = 250,
                       noverlap: int = 256,
                       nfft: int = 512,
                       nperseg: int = 512,
                       ) -> None:
    """
    Validates if dictionary of channels and signals satisfy basic criteria.

    :param eeg_dict: Dictionary of channels (keys) and arrays of signal values (values)
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.

    """

    channels = asarray(list(eeg_dict.keys()))
    signals = asarray(list(eeg_dict.values()))

    if channels != MONTAGE_1020:
        raise ValueError(f"Channels do not match MONTAGE_1020")
    if len(signals) < 4 * fs:
        raise ValueError("Signals length must be at least four times the sampling frequency")

    _assert_nfft(n=signals.shape[1],
                 nfft=nfft,
                 nperseg=nperseg,
                 noverlap=noverlap)
