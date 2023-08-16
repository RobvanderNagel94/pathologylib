from features.frequency.welch import estimate_welch
from utils.dict_validation import _assert_valid_dict
from numpy import sum


def APG(eeg_dict: dict,
        fs: int = 250,
        window: str = 'hann',
        noverlap: int = 256,
        nfft: int = 512,
        nperseg: int = 512,
        fmin: int = 8,
        fmax: int = 12) -> float:

    """
    Computes the anterior to posterior ratio (gradient) of alpha power from segments
    annotated with the eyes closed state.

    :param eeg_dict: Dictionary of channels (keys) and arrays of signal values (values)
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param window: Window function, default 'hann'.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.
    :param fmin: Minimum frequency bound for power spectrum, default 8.
    :param fmax: Maximum frequency bound for power spectrum, default 12.

    :return: Quantified and normalised value for the anterio-posterior gradient.

    Notes
    --------

    If Qapg < 0.4: Alpha power gradient is categorized as normal.
    If 0.4 < Qapg < 0.6: Alpha power gradient is considered moderately differentiated.
    If Qapg > 0.6: Alpha power gradient is considered pathological.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    _assert_valid_dict(eeg_dict=eeg_dict, fs=fs, noverlap=noverlap, nfft=nfft, nperseg=nperseg)

    anterior = ['fp1', 'fp2', 'f7', 'f8', 'f3', 'fz', 'f4']
    posterior = ['t5', 't6', 'p3', 'p4', 'pz', 'o1', 'o2']

    Pxx_ant = [sum(sum(estimate_welch(eeg_dict[ch], fs, window, noverlap, nfft, nperseg, fmin, fmax)[1])) for ch in
               anterior]
    Pxx_pos = [sum(sum(estimate_welch(eeg_dict[ch], fs, window, noverlap, nfft, nperseg, fmin, fmax)[1])) for ch in
               posterior]

    return float(sum(Pxx_ant) / (sum(Pxx_ant) + sum(Pxx_pos)))
