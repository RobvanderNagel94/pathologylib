from features.multi_channel.features_background import estimate_welch

import numpy as np


def COG(x: dict,
        fs: float = 250,
        window: str = 'hann',
        noverlap: int = 256,
        nfft: int = 512,
        nperseg: int = 512,
        ) -> (np.ndarray, np.ndarray):
    
    """
    Computes left-right and anterior-posterior power distributions along the scalp using
    van Putten's center-of-gravity feature.

    :param x: Dictionary of EEG data, key=channels, values=signals, passed as single segment
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param window: Window function, default 'hann'.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.

    :return: Power in left-right direction, weighted by channel positions.
    :return: Power in anterior-posterior direction, weighted by channel positions.

    References
    ----------
    [1] Van Putten (2007), The colorful brain: compact visualisation of routine EEG recordings.
    .. 10.1007/978-3-540-73044-6_127

    """

    # Transpose power spectrum
    Sxx = np.transpose(

        [estimate_welch(x[ch], fs, window, noverlap, nfft, nperseg)[1] for ch in x]

    )
    # Order channels = [Fp2 F8 T4 T6 O2 F4 C4 P4 Fz Cz Pz F3 C3 P3 Fp1 F7 T3 T5 O1]
    # Compute euclidean distances for each channel location from center location Cz
    x = np.array([0, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, -2, -2, -2, -1])
    y = np.array([0, 1, 0, -1, -2, 1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, -1, -2])

    # Normalise power in X and Y directions
    x_cog = (Sxx @ x) / Sxx.sum(axis=1)
    y_cog = (Sxx @ y) / Sxx.sum(axis=1)

    return x_cog, y_cog
