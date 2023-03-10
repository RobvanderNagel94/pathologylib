import numpy as np
from mne.filter import filter_data
from typing import Union


def split_into_epochs(signals: Union[np.ndarray, list],
                      fs: int,
                      epoch_duration_s: int
                      ) -> np.ndarray:
    """
    Splits the signals into non-overlapping epochs.
    :param signals: Signals to be split into epochs.
    :param fs: Sample frequency of the signals.
    :param epoch_duration_s: Desired duration of each epoch in seconds.

    :return: Array of epochs (n_epochs, n_channels, epoch_samples)

    Example:
    --------
    >>> import numpy as np
    >>> signals = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> fs = 2
    >>> epoch_duration_s = 3
    >>> split_into_epochs(signals, fs, epoch_duration_s)
    array([[[1, 2, 3],
            [3, 4, 5],
            [5, 6, 7],
            [7, 8, 9]]])
    """

    signals = np.asarray(signals)

    n_samples = signals.shape[-1]
    n_samples_in_epoch = int(epoch_duration_s * fs)
    epochs = []

    error_msg = f"ValueError: n_samples must be larger than n_samples_in_epoch"
    assert n_samples > n_samples_in_epoch, error_msg

    # +1 for last window when n_samples is perfectly dividable
    for i in range(0, n_samples - n_samples_in_epoch + 1, n_samples_in_epoch):
        epoch = np.take(signals, range(i, i + n_samples_in_epoch), axis=-1)
        epochs.append(epoch)

    return np.stack(epochs)


def replace_outliers(crops: Union[np.ndarray, list],
                     outlier_threshold: float = 800.,
                     replacement_threshold: float = 800.) -> np.ndarray:
    """
    Replace epochs that contain outliers based on a threshold.

    :param crops: Iterable of shape (n_channels, n_samples).
    :param outlier_threshold: Threshold for what value is considered an outlier, by default 800 microVolt.
    :param replacement_threshold: Threshold for the value to be replaced with, by default 800 microVolt.

    :return: Array with replaced outliers.

    Example
    ----------
    >>> import numpy as np
    >>> crops = np.array([[-1000, 200, 300],
    >>>                    [400, 500, 900]])
    >>> replace_outliers(crops, replacement_threshold= 800., outlier_threshold = 800.)
    array([[-800, 200, 300],
           [400, 500, 800]])

    """
    crops = np.asarray(crops)

    crops[crops >= outlier_threshold] = replacement_threshold
    crops[crops <= -1 * outlier_threshold] = -replacement_threshold

    return crops


def apply_window_function(epochs: Union[np.ndarray, list],
                          window_name: str = "blackman") -> np.ndarray:
    """
    Apply window function to the given epochs.

    :param epochs: Array of shape (n_channels, n_samples).
    :param window_name: The name of the window function to apply.
        Supports "hamming", "hanning", "blackman", "bartlett", "kaiser".

    :return: windowed epochs.

    Raises
    ----------
    AssertionError
        If the `window_name` is not one of the supported window functions.
    """

    window_functions = ["hamming", "hanning", "blackman", "bartlett", "kaiser"]

    if window_name not in window_functions:
        raise ValueError(
            f"window_name = {window_name} is not supported"
            f" Supported window function are "
            f" 'hamming', 'hanning', 'blackman', "
            f" 'bartlett', 'kaiser' "
        )

    epochs = np.asarray(epochs)
    n_samples_in_epoch = epochs.shape[-1]
    window_function = getattr(np, window_name)(n_samples_in_epoch)

    return epochs * window_function


def filter_to_frequency_band(signals: Union[np.ndarray, list],
                             sfreq: int,
                             lower: int,
                             upper: int) -> np.ndarray:
    """
    Filter signals to the frequency range defined by `lower` and `upper`.

    :param signals: Input signals.
    :param sfreq: Sampling frequency of the signals.
    :param lower: Lower frequency boundary.
    :param upper: Upper frequency boundary.

    :return: Filtered signals.

    References
    ----------
    mne.filter.filter_data

    """
    signals = np.asarray(signals)
    return filter_data(data=signals, sfreq=sfreq, l_freq=lower, h_freq=upper, verbose='error')


def filter_to_frequency_bands(signals: Union[np.ndarray, list],
                              bands: Union[np.ndarray, list],
                              fs: int) -> np.ndarray:
    """
    Filter signals to the frequency ranges defined in `bands`.

    :param signals: Input signals.
    :param bands: List of frequency ranges, represented as tuples of lower and upper frequencies.
    :param fs: Sampling frequency of the signals.

    :return: Filtered signals.
    """

    signals = np.asarray(signals)
    bands = np.asarray(bands)

    signals = signals.astype(np.float64)
    (n_signals, n_times) = signals.shape
    band_signals = np.ndarray(shape=(len(bands), n_signals, n_times))
    for band_id, band in enumerate(bands):
        lower, upper = band
        if upper >= fs / 2:
            upper = None
        curr_band_signals = filter_to_frequency_band(signals, fs, lower, upper)
        band_signals[band_id] = curr_band_signals
    return band_signals


def assemble_overlapping_band_limits(non_overlapping_bands: Union[np.ndarray, list]) -> np.ndarray:
    """
    Create 50% overlapping frequency bands from non-overlapping bands.

    :param non_overlapping_bands: List of non-overlapping frequency bands.
    :return: array of overlapping frequency bands.
    """
    non_overlapping_bands = np.asarray(non_overlapping_bands)

    overlapping_bands = []
    for i in range(len(non_overlapping_bands) - 1):
        band_i = non_overlapping_bands[i]
        overlapping_bands.append(band_i)
        band_j = non_overlapping_bands[i + 1]
        overlapping_bands.append([int((band_i[0] + band_j[0]) / 2),
                                  int((band_i[1] + band_j[1]) / 2)])
    overlapping_bands.append(non_overlapping_bands[-1])
    return np.array(overlapping_bands)
