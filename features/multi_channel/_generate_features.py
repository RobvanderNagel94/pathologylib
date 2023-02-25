from utils.preprocessing import (split_into_epochs,
                                 apply_window_function,
                                 reject_windows_with_outliers,
                                 assemble_overlapping_band_limits)

from .generator_wavelet import (generate_cwt_features, generate_dwt_features)
from .generator_time import generate_time_features
from .generator_frequency import generate_dft_features
from .generator_phase import generate_phase_features
import numpy as np

sfreq = 250
epoch_duration_s = 10
max_abs_val = 800
window_name = "blackmanharris"
agg_mode = "median"
continuous_wavelet = "morl"
discrete_wavelet = "db4"
band_overlap = True
domains = "all"

band_limits = [[0, 2], [2, 4], [4, 8], [8, 13], [13, 18], [18, 24], [24, 30], [30, 49.9]]

default_feature_generation_params = {
    "epoch_duration_s": epoch_duration_s,
    "max_abs_val": max_abs_val,
    "window_name": window_name,
    "band_limits": band_limits,
    "agg_mode": agg_mode,
    "discrete_wavelet": discrete_wavelet,
    "continuous_wavelet": continuous_wavelet,
    "band_overlap": band_overlap
}


def generate_features_of_one_file(signals: np.ndarray,
                                  sfreq: int,
                                  epoch_duration_s: int,
                                  max_abs_val: int,
                                  window_name: str,
                                  band_limits: list,
                                  agg_mode: any,
                                  discrete_wavelet: str,
                                  continuous_wavelet: str,
                                  band_overlap: any) -> np.ndarray:
    """
    Generates various types of features for one file.

    :param signals: Array representing the data.
    :param sfreq: Sampling frequency of the crops.
    :param epoch_duration_s: Duration of each epoch in seconds.
    :param max_abs_val: The maximum absolute value used to reject outliers.
    :param epoch_duration_s: The length of the epochs used
    :param band_limits: The frequency limits of the bands to compute the features.
    :param agg_mode: Aggregation function applied to the features.
    :param discrete_wavelet: The name of the discrete wavelet transformer.
    :param continuous_wavelet: The name of the continuous wavelet transformer.
    :param band_overlap: Flag indicating whether the bands should overlap.

    :return: Multi-channel features of the signals.
    """

    non_overlapping_bands = band_limits

    if band_overlap:
        band_limits = assemble_overlapping_band_limits(band_limits)

    crops = split_into_epochs(
        signals=signals, sfreq=sfreq, epoch_duration_s=epoch_duration_s)

    outlier_mask = reject_windows_with_outliers(
        outlier_value=max_abs_val, epochs=crops)
    crops = crops[outlier_mask == False]

    if crops.size == 0:
        raise ValueError("Removed all crops due to outliers")

    weighted_crops = apply_window_function(
        epochs=crops, window_name=window_name)

    print('>> Extract CWT features ..')
    cwt_features = generate_cwt_features(
        crops=weighted_crops, wavelet=continuous_wavelet, sfreq=sfreq,
        band_limits=non_overlapping_bands, agg_func=agg_mode)
    print('\n>> Extract DWT features ..')
    dwt_features = generate_dwt_features(
        crops=weighted_crops, wavelet=discrete_wavelet, sfreq=sfreq,
        agg_func=agg_mode)
    print('\n>> Extract DFT features ..')
    dft_features = generate_dft_features(
        crops=weighted_crops, sfreq=sfreq, band_limits=band_limits,
        agg_func=agg_mode)
    print('\n>> Extract Synchrony features ..')
    phase_features = generate_phase_features(
        signals=signals, agg_func=agg_mode, band_limits=band_limits,
        outlier_mask=outlier_mask, sfreq=sfreq,
        epoch_duration_s=epoch_duration_s)
    print('\n>> Extract Time features ..')
    time_features = generate_time_features(
        crops=crops, agg_func=agg_mode, sfreq=sfreq)

    all_features = [features for features in
                    [cwt_features, dwt_features, dft_features,
                     phase_features, time_features] if features is not None]

    print('\n>> Done!')
    axis = 1 if agg_mode in ["none", "None", None] else 0
    return np.concatenate(all_features, axis=axis)
