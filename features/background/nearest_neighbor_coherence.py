from utils.dict_validation import _assert_valid_dict

from scipy.signal import coherence
from numpy import ndarray, asarray


def mNNC(eeg_dict: dict,
         fs: int = 250,
         window: str = 'hann',
         noverlap: int = 256,
         nfft: int = 512,
         nperseg: int = 512
         ) -> ndarray:
    """
    Computes signal coherence's for each neighboring channel
    (mNNC = maximum nearest neighbor coherence).

    :param eeg_dict: Dictionary of channels (keys) and arrays of signal values (values), passed per segment.
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param window: Window function, default 'hann'.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.

    :return: Outputs maximum coherence values from all 19 channels.

    Notes
    ----------
    The input signals and channels must satisfy the output of the 1020 system with 19 channels:
    MONTAGE_1020 = [FP2, F8, T4, T6, O2, F4, C4, P4, FZ, CZ, PZ, F3, C3, P3, FP1, F7, T3, T5, O1]

    References
    ----------
    [1] Van Putten (2007), The colorful brain: compact visualisation of routine EEG recordings.
    10.1007/978-3-540-73044-6_127
    """

    _assert_valid_dict(eeg_dict=eeg_dict, fs=fs, noverlap=noverlap, nfft=nfft, nperseg=nperseg)

    neighbor_pairs = [
        ('fp1', 'fz'), ('fp1', 'f3'), ('fp1', 'f7'),
        ('fp2', 'fz'), ('fp2', 'f4'), ('fp2', 'f8'),
        ('f4', 'fz'), ('f4', 'f8'), ('f4', 'c4'),
        ('f3', 'fz'), ('f3', 'f7'), ('f3', 'c3'),
        ('fz', 'f3'), ('fz', 'f4'), ('fz', 'cz'),
        ('cz', 'fz'), ('cz', 'c4'), ('cz', 'c3'), ('cz', 'pz'),
        ('pz', 'p4'), ('pz', 'p3'), ('pz', 'o2'), ('pz', 'o1'),
        ('c4', 'cz'), ('c4', 'f4'), ('c4', 'p4'), ('c4', 't4'),
        ('c3', 'cz'), ('c3', 'f3'), ('c3', 'p3'), ('c3', 't3'),
        ('p4', 'c4'), ('p4', 't6'), ('p4', 'pz'), ('p4', 'o2'),
        ('p3', 'c3'), ('p3', 't5'), ('p3', 'pz'), ('p3', 'o1'),
        ('o2', 'pz'), ('o2', 'p4'), ('o2', 't6'),
        ('o1', 'pz'), ('o1', 'p3'), ('o1', 't5'),
        ('f8', 't4'), ('f8', 'c4'), ('f8', 'f4'),
        ('f7', 't3'), ('f7', 'c3'), ('f7', 'f3'),
        ('t4', 'f8'), ('t4', 'c4'), ('t4', 't6'),
        ('t3', 'f7'), ('t3', 'c3'), ('t3', 't5'),
        ('t6', 't4'), ('t6', 'p4'), ('t6', 'o2'),
        ('t5', 't3'), ('t5', 'p3'), ('t5', 'o1')
    ]


    coh_dict = {}
    for ch1, ch2 in neighbor_pairs:
        _, coh = coherence(eeg_dict[ch1], eeg_dict[ch2], fs, window, nperseg, noverlap, nfft)
        coh_dict[ch1] = coh_dict.get(ch1, []) + [coh]

    coh_values = []
    for ch, coh_list in coh_dict.items():
        if len(coh_list) > 1:
            coh_mean = sum(coh_list) / len(coh_list)
        else:
            coh_mean = coh_list[0]
        coh_values.append(coh_mean)

    return asarray(coh_values)
