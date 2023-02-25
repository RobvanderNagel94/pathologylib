from mne.filter import filter_data
from scipy import signal
import numpy as np


def split_into_epochs(signals: np.ndarray,
                      sfreq: int,
                      epoch_duration_s: int
                      ) -> np.ndarray:
    """
    Splits the signals into non-overlapping epochs.

    :param signals: Signals to be split into epochs.
    :param sfreq: Sample frequency of the signals.
    :param epoch_duration_s: Desired duration of each epoch in seconds.

    :return: Splitted epochs.

    Example:
    --------
    >>> import numpy as np
    >>> signals = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> sfreq = 2
    >>> epoch_duration_s = 3
    >>> split_into_epochs(signals, sfreq, epoch_duration_s)
    array([[[1, 2, 3],
            [3, 4, 5],
            [5, 6, 7],
            [7, 8, 9]]])
    """

    n_samples = signals.shape[-1]
    n_samples_in_epoch = int(epoch_duration_s * sfreq)
    epochs = []

    # +1 for last window when n_samples is perfectly dividable
    for i in range(0, n_samples-n_samples_in_epoch+1, n_samples_in_epoch):
        epoch = np.take(signals, range(i, i + n_samples_in_epoch), axis=-1)
        epochs.append(epoch)

    return np.stack(epochs)


def reject_windows_with_outliers(epochs: np.ndarray, outlier_value: int = 800) -> np.ndarray:
    """
    Reject windows that contain outliers/clipped values.

    :param epochs: Epochs to be checked for outliers.
    :param outlier_value: Threshold for what value is considered an outlier, by default 800.

    :return: Array of booleans indicating which epochs contain outliers.

    Examples
    --------
    >>> import numpy as np
    >>> epochs = np.array([[[-1000, 200, 300], [400, 500, 600]], [[-200, 300, 400], [500, 600, 900]]])
    >>> reject_windows_with_outliers(epochs, outlier_value=800)
    array([False, True])

    In this example, the epochs array has shape (2, 2, 3) and contains 2 epochs.
    The first epoch has values below the outlier threshold of -800 and above the threshold of 800, but the second epoch has no values outside the threshold.
    Therefore, the function returns a boolean array with shape (2,), indicating that both epochs do not contain any outliers.

    """
    pos_outliers = np.sum(epochs >= outlier_value, axis=(1, 2))
    neg_outliers = np.sum(epochs <= -1 * outlier_value, axis=(1, 2))
    outliers = np.logical_or(pos_outliers, neg_outliers)
    return outliers


def apply_window_function(epochs: np.ndarray, window_name: str = "blackmanharris") -> np.ndarray:
    """
    Apply window function to the given epochs.

    :param epochs: Epochs to be applied for window function.
    :param window_name: The name of the window function to apply.
        Supports "boxcar", "hamming", "hann", "blackmanharris", "flattop".

    :return: windowed epochs.

    Raises
    ------
    AssertionError
        If the `window_name` is not one of the supported window functions.
    """
    assert window_name in ["boxcar", "hamming", "hann", "blackmanharris", "flattop"], \
        "cannot handle window {}".format(window_name)

    n_samples_in_epoch = epochs.shape[-1]
    method_to_call = getattr(signal, window_name)
    window_function = method_to_call(n_samples_in_epoch)
    return epochs * window_function


def filter_to_frequency_band(signals: np.ndarray, sfreq: int, lower: int, upper: int) -> np.ndarray:
    """
    Filter signals to the frequency range defined by `lower` and `upper`.

    :param signals: Input signals.
    :param sfreq: Sampling frequency of the signals.
    :param lower: Lower frequency boundary.
    :param upper: Upper frequency boundary.

    :return: Filtered signals.
    """
    return filter_data(data=signals, sfreq=sfreq, l_freq=lower, h_freq=upper, verbose='error')


def filter_to_frequency_bands(signals: np.ndarray, bands: list, sfreq: int) -> np.ndarray:
    """
    Filter signals to the frequency ranges defined in `bands`.

    :param signals: Input signals.
    :param bands: List of frequency ranges, represented as tuples of lower and upper frequencies.
    :param sfreq: Sampling frequency of the signals.

    :return: Filtered signals.
    """
    signals = signals.astype(np.float64)
    (n_signals, n_times) = signals.shape
    band_signals = np.ndarray(shape=(len(bands), n_signals, n_times))
    for band_id, band in enumerate(bands):
        lower, upper = band
        if upper >= sfreq / 2:
            upper = None
        curr_band_signals = filter_to_frequency_band(signals, sfreq, lower, upper)
        band_signals[band_id] = curr_band_signals
    return band_signals


def assemble_overlapping_band_limits(non_overlapping_bands: list) -> np.ndarray:
    """
    Create 50% overlapping frequency bands from non-overlapping bands.

    :param non_overlapping_bands: List of non-overlapping frequency bands.

    :return: array of overlapping frequency bands.
    """
    overlapping_bands = []
    for i in range(len(non_overlapping_bands) - 1):
        band_i = non_overlapping_bands[i]
        overlapping_bands.append(band_i)
        band_j = non_overlapping_bands[i + 1]
        overlapping_bands.append([int((band_i[0] + band_j[0]) / 2),
                                  int((band_i[1] + band_j[1]) / 2)])
    overlapping_bands.append(non_overlapping_bands[-1])
    return np.array(overlapping_bands)