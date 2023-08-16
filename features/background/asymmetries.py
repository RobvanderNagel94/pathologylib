from utils.dict_validation import _assert_valid_dict
from features.frequency.welch import estimate_welch

from numpy import (ndarray, asarray)


def InterhemisphericAsymmetries(eeg_dict: dict,
                                fs: int = 250,
                                window: str = 'hann',
                                noverlap: int = 256,
                                nfft: int = 512,
                                nperseg: int = 512,
                                fmin: int = 2,
                                fmax: int = 15) -> ndarray:

    """
    Computes asymmetrical background patterns by comparing rhythmic activity
    between the two hemispheres in corresponding left and right channel pairs.

    :param eeg_dict: Dictionary of channels (keys) and arrays of signal values (values)
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param window: Window function, default 'hann'.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.
    :param fmin: Minimum frequency bound for power spectrum, default 2.
    :param fmax: Maximum frequency bound for power spectrum, default 15.

    :return: Quantified and normalised asymmetry values for each left and right channel pair.

    Notes
    ---------

    If Qasym > 0.5: non-pathological asymmetry.
    If Qasym < 0.5: pathological asymmetry.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    _assert_valid_dict(eeg_dict=eeg_dict, fs=fs, noverlap=noverlap, nfft=nfft, nperseg=nperseg)

    left = ['fp1', 'f7', 'f3', 't3', 'c3', 't5', 'p3', 'o1']
    right = ['fp2', 'f8', 'f4', 't4', 'c4', 't6', 'p4', 'o2']

    Sxx_left = asarray(
        [estimate_welch(eeg_dict[ch], fs, window, noverlap, nfft, nperseg, fmin=fmin, fmax=fmax)[1] for ch in
         left])

    Sxx_right = asarray(
        [estimate_welch(eeg_dict[ch], fs, window, noverlap, nfft, nperseg, fmin=fmin, fmax=fmax)[1] for ch in
         right])

    lr_asymmetries = [(Sxx_right[i].sum() - Sxx_left[i].sum()) / (Sxx_right[i].sum() + Sxx_left[i].sum()) for i in range(len(Sxx_left))]

    return asarray(lr_asymmetries)
