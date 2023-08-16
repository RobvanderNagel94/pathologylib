from features.time import features_time as ft
from features.time.features_time import time_params
from numpy import ndarray, asarray, hstack
import numpy as np


def generate_time_features(crops: ndarray,
                           fs: int,
                           agg_func: any = np.median,
                           default_params: dict = time_params,
                           axis: int = -1) -> ndarray:
    """
    Generates time features of `crops` using the given parameters.

    :param crops: Array of shape (n_crops, n_elecs, n_samples_in_epoch).
    :param fs: Sampling frequency of the crops.
    :param agg_func: Aggregation function to apply.
    :param default_params: Dictionary of parameters for specific time features
    :param axis: Axis along which the aggregation function should be applied.
        axis=-1 means that the features are computed at the last index.

    :return: Array of time features
    """

    time_params = {
        
        "fs": 250,
        "Kmax": 3,
        "n": 4,
        "T": 1,
        "Tau": 4,
        "DE": 10,
        "W": None
    }

    Kmax = default_params['Kmax']
    n = default_params['n']
    T = default_params['T']
    Tau = default_params['Tau']
    DE = default_params['DE']
    W = default_params['W']

    print('>> Extracting statistical features')
    activity = ft.hjorth_activity(crops=crops, axis=axis)
    mobility = ft.hjorth_mobility(crops=crops, axis=axis)
    complexity = ft.hjorth_complexity(crops=crops, axis=axis)
    energy = ft.energy(crops=crops, axis=axis)
    non_lin_energy = ft.non_linear_energy(crops=crops, axis=axis)
    line_length = ft.line_length(crops=crops, axis=axis)
    lumpiness = ft.lumpiness(crops=crops, fs=fs, axis=axis)
    flat_spots = ft.flat_spots(crops=crops, axis=axis)
    stability = ft.stability(crops=crops, axis=axis, fs=fs)
    max_ = ft.maximum(crops=crops, axis=axis)
    min_ = ft.minimum(crops=crops, axis=axis)
    mean_ = ft.mean(crops=crops, axis=axis)
    median_ = ft.median(crops=crops, axis=axis)
    skew_ = ft.skewness(crops=crops, axis=axis)
    kurt_ = ft.kurtosis(crops=crops, axis=axis)
    zero_crossings = ft.zero_crossing(crops=crops, axis=axis)
    zero_crossings_dev = ft.zero_crossing_derivative(crops=crops, axis=axis)

    print('>> Extracting time-complexity features')
    lyapunov_exp = ft.largest_lyapunov_exponent(crops=crops, axis=axis, Tau=Tau, n=n, T=T, fs=fs)
    hurst_exp = ft.hurst_exponent(crops=crops, axis=axis)
    higuchi_fractal_dim = ft.higuchi_fractal_dimension(crops=crops, axis=axis, Kmax=Kmax)
    petrosian_fractal_dim = ft.petrosian_fractal_dimension(crops=crops, axis=axis)
    svd_entropy = ft.svd_entropy(crops=crops, axis=axis, Tau=Tau, DE=DE, W=W)
    svd_fisher = ft.svd_fisher(crops=crops, axis=axis, Tau=Tau, DE=DE)

    time_features = hstack(

        (activity, mobility, complexity, energy, non_lin_energy, line_length,
         lumpiness, flat_spots, stability, max_, min_, mean_, median_, skew_,
         kurt_, zero_crossings, zero_crossings_dev, lyapunov_exp, hurst_exp,
         higuchi_fractal_dim, petrosian_fractal_dim, svd_entropy, svd_fisher)

    )

    time_features = time_features.reshape(len(crops), -1)

    if agg_func is not None:
        time_features = agg_func(time_features, axis=0)

    return asarray(time_features)
