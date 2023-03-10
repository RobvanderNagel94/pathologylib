from typing import Union


def _assert_nfft(n: int,
                 nfft: int,
                 nperseg: Union[int, None],
                 noverlap: Union[int, None]) -> None:
    """
    Check if nfft, nperseg, and noverlap make sense given n samples.

    :param n: Number of data points.
    :param nfft: Number of data points used for the FFT.
    :param nperseg: Number of data points in each epoch.
    :param noverlap: Number of data points between epochs.
        If None, noverlap = nperseg // 2

    """

    error_msg = (f"If nperseg is None nfft is not allowed to be > n_times. "
                 f"If you want zero-padding, you have to set nperseg to relevant length. "
                 f"Got nfft of {nfft} while signal length is {n}.")

    if nperseg is None and nfft > n:
        raise ValueError(error_msg)

    nperseg = nfft if nperseg is None or nperseg > nfft else nperseg
    nperseg = n if nperseg > n else nperseg

    error_msg = (f"noverlap cannot be greater than nperseg (or 'nfft'). "
                 f"Got noverlap of {noverlap} while nperseg is {nperseg}.")

    if noverlap >= nperseg:
        raise ValueError(error_msg)
