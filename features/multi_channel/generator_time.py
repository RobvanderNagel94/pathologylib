from features.multi_channel import features_time as ft
import numpy as np


def generate_time_features(crops: np.ndarray,
                           sfreq: int,
                           agg_func: any,
                           axis=-1):
    """
    Generates time features of `crops` using the given parameters.

    :param crops: Array of shape (n_crops, n_elecs, n_samples_in_epoch) representing the data.
    :param sfreq: Sampling frequency of the crops.
    :param agg_func: Aggregation function to apply to the CWT features.
    :param axis: Axis along which the aggregation function should be applied.,
        axis=-1 means that the features are computed at the last index

    :return: Time features
    """

    Kmax = 3
    n = 4
    T = 1
    Tau = 4
    DE = 10
    W = None

    energy_ = ft.energy(epochs=crops, axis=axis)
    fisher_info = ft.svd_fisher(epochs=crops, axis=axis)
    higuchi_fractal_dim = ft.higuchi_fractal_dimension(epochs=crops, axis=axis, Kmax=Kmax)
    [activity, mobility, complexity] = ft._hjorth_parameters(epochs=crops, axis=axis)
    hurst_exp = ft.hurst_exponent(epochs=crops, axis=axis)
    kurt = ft.kurtosis(epochs=crops, axis=axis)
    line_length = ft.line_length(epochs=crops, axis=axis)
    lumpiness = ft.lumpiness(epochs=crops, freq=sfreq, axis=axis)
    flat_spots = ft.flat_spots(epochs=crops, axis=axis)
    lyapunov_exp = ft.largest_lyapunov_exponent(epochs=crops, axis=axis, Tau=Tau, n=n, T=T, fs=sfreq)
    max_ = ft.maximum(epochs=crops, axis=axis)
    mean_ = ft.mean(epochs=crops, axis=axis)
    median_ = ft.median(epochs=crops, axis=axis)
    min_ = ft.minimum(epochs=crops, axis=axis)
    non_lin_energy = ft.non_linear_energy(epochs=crops, axis=axis)
    petrosian_fractal_dim = ft.petrosian_fractal_dimension(epochs=crops, axis=axis)
    skew = ft.skewness(epochs=crops, axis=axis)
    svd_entropy_ = ft.svd_entropy(epochs=crops, axis=axis, Tau=Tau, DE=DE, W=W)
    zero_crossings = ft.zero_crossing(epochs=crops, axis=axis)
    zero_crossings_dev = ft.zero_crossing_derivative(epochs=crops, axis=axis)

    time_features = np.hstack((
        energy_, fisher_info, higuchi_fractal_dim, activity,
        complexity, mobility, hurst_exp, kurt, line_length, lumpiness, flat_spots,
        lyapunov_exp, max_, mean_, median_, min_, non_lin_energy, petrosian_fractal_dim,
        skew, svd_entropy_, zero_crossings, zero_crossings_dev
    ))

    time_features = time_features.reshape(len(crops), -1)
    if agg_func is not None:
        time_features = agg_func(time_features, axis=0)

    return time_features
