from utils.dict_validation import _assert_valid_dict
from features.frequency.welch import estimate_welch

from numpy import asarray


def DiffuseSlowing(eeg_dict: dict,
                   fs: int = 250,
                   window: str = 'hann',
                   noverlap: int = 256,
                   nfft: int = 512,
                   nperseg: int = 512,
                   fmin: int = 2,
                   fmax_low: int = 8,
                   fmax_wide: int = 25) -> float:
    """
    Computes diffuse slow-wave activity for segments with the eyes closed state.

    :param eeg_dict: Dictionary of channels (keys) and arrays of signal values (values)
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param window: Window function, default 'hann'.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.
    :param fmin: Minimum frequency bound for power spectrum, default 2.
    :param fmax_low: Maximum frequency bound for lower power spectrum, default 8.
    :param fmax_wide: Maximum frequency bound for upper power spectrum, default 25.

    :return: Quantified and normalised value for diffused slow-wave activity.

    Notes
    ---------
    Diffused slowing results in an increased power over the Delta [0-4]Hz and Theta [4-8]Hz bands
    and decreased power in the Alpha [8-12]Hz and Beta bands [13-25]Hz.
    Using the guideline above, the mean spectrum of the EEG is calculated and the power ratio between
    Plow = {2. . .8} Hz and Pwide = {2. . .25} Hz

    If Qslow > 0.6: EEG is categorized as pathological.
    If Qslow < 0.6: EEG is categorized as non-pathological.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    _assert_valid_dict(eeg_dict=eeg_dict, fs=fs, noverlap=noverlap, nfft=nfft, nperseg=nperseg)

    channels = asarray(list(eeg_dict.keys()))

    Plow = asarray(
        [estimate_welch(eeg_dict[ch], fs, window, noverlap, nfft, nperseg, fmin=fmin, fmax=fmax_low)[1] for ch in
         channels]).sum(axis=0).sum()

    Pwide = asarray(
        [estimate_welch(eeg_dict[ch], fs, window, noverlap, nfft, nperseg, fmin=fmin, fmax=fmax_wide)[1] for ch in
         channels]).sum(axis=0).sum()

    return float(Plow / Pwide)
