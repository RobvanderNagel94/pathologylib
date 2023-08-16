from pywt._extensions._pywt import (DiscreteContinuousWavelet, ContinuousWavelet, Wavelet, _check_dtype)
from pywt._functions import (integrate_wavelet, scale2frequency, central_frequency)
from pywt import wavedec, dwt_max_level, wavelist
from features.frequency import features_wavelet as fw
from typing import Union
from numpy import (array, asarray, ndarray, isscalar,
                   size, take, abs, fft, zeros, sum,
                   log, arange, apply_along_axis, delete,
                   sqrt, floor, max, where, diff, ceil, swapaxes, convolve)


class WaveletTransformer:
    def __init__(self):
        pass

    def freq_to_scale(self, freq, wavelet, sfreq):
        central_freq = central_frequency(wavelet)
        assert freq > 0, "freq smaller or equal to zero!"
        scale = central_freq / freq
        return scale * sfreq

    def freqs_to_scale(self, freqs, wavelet, sfreq):
        scales = []
        for freq in freqs:
            scale = self.freq_to_scale(freq[1], wavelet, sfreq)
            scales.append(scale)
        return scales

    def pywt_cwt(self, data, scales, wavelet):
        dt = _check_dtype(data)
        data = array(data, dtype=dt)
        if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
            wavelet = DiscreteContinuousWavelet(wavelet)
        if isscalar(scales):
            scales = array([scales])
        if data.ndim == 1:
            if wavelet.complex_cwt:
                out = zeros((size(scales), data.size), dtype=complex)
            else:
                out = zeros((size(scales), data.size))
            for i in arange(size(scales)):
                precision = 10
                int_psi, x = integrate_wavelet(wavelet, precision=precision)
                step = x[1] - x[0]
                j = floor(arange(
                    scales[i] * (x[-1] - x[0]) + 1) / (scales[i] * step))
                if max(j) >= size(int_psi):
                    j = delete(j, where((j >= size(int_psi)))[0])
                coef = - sqrt(scales[i]) * diff(
                    convolve(data, int_psi[j.astype(int)][::-1]))
                d = (coef.size - data.size) / 2.
                out[i, :] = coef[int(floor(d)):int(-ceil(d))]
            return out
        else:
            raise ValueError("Only dim == 1 supported")

    def cwt_transform(self, crops, wavelet, band_limits, fs):
        scales = self.freqs_to_scale(
            freqs=band_limits, wavelet=wavelet, sfreq=fs)
        coefficients = apply_along_axis(
            func1d=self.pywt_cwt, axis=2, arr=crops, scales=scales,
            wavelet=wavelet)
        coefficients = swapaxes(a=coefficients, axis1=1, axis2=2)
        return coefficients

    def dwt_transform(self, crops, wavelet, fs):
        (n_windows, n_elecs, n_samples_in_window) = crops.shape
        max_level = dwt_max_level(
            data_len=n_samples_in_window, filter_len=wavelet)
        pseudo_freqs = [fs / 2 ** i for i in range(1, max_level)]
        pseudo_freqs = [pseudo_freq for pseudo_freq in pseudo_freqs
                        if pseudo_freq >= 2]
        n_levels = len(pseudo_freqs)
        multi_level_coeffs = wavedec(
            data=crops, wavelet=wavelet, level=n_levels - 1, axis=2)
        return multi_level_coeffs


def generate_cwt_features(crops: Union[ndarray, list],
                          wavelet: str,
                          band_limits: Union[ndarray, list],
                          agg_func: any,
                          fs: int) -> ndarray:
    """
    Generate Continuous Wavelet Transform (CWT) features for input crops.

    :param crops: Array of shape (n_crops, n_elecs, n_samples_in_epoch) representing the data.
    :param wavelet: Wavelet to use for the CWT transform. ("morl" or "db4")
    :param band_limits: List of band limits for the CWT
    :param agg_func: Aggregation function to apply to the CWT features. ("median")
    :param fs: Sampling frequency of the crops.

    :return: CWT features with shape (n_crops, n_features) where n_features is determined by the aggregation function.
    """
    crops = asarray(crops)

    (n_crops, n_elecs, n_samples_in_epoch) = crops.shape

    wt = WaveletTransformer()

    cwt_coefficients = wt.cwt_transform(crops=crops,
                                        wavelet=wavelet,
                                        band_limits=band_limits,
                                        fs=fs)

    cwt_feats = ndarray(shape=(n_crops, 7, len(cwt_coefficients), n_elecs))
    print(cwt_feats.shape)

    cwt_feats[:, 0, :, :] = fw.bounded_variation(coefficients=cwt_coefficients, axis=-1)
    cwt_feats[:, 1, :, :] = fw.entropy(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 2, :, :] = fw.maximum(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 3, :, :] = fw.mean(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 4, :, :] = fw.minimum(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 5, :, :] = fw.power(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 6, :, :] = fw.variance(coefficients=cwt_coefficients, axis=3)

    powers = cwt_feats[:, 5, :, :]
    ratios = powers / sum(powers, axis=1, keepdims=True)
    cwt_feats[:, 7, :, :] = ratios

    cwt_features = cwt_feats.reshape(n_crops, -1)
    if agg_func is not None:
        cwt_features = agg_func(cwt_features, axis=0)

    return cwt_features


def generate_dwt_features(crops: Union[ndarray, list],
                          wavelet: str,
                          agg_func: any,
                          fs: int) -> ndarray:
    """
    Generate Discrete Wavelet Transform (DWT) features for input crops.

    :param crops: Array of shape (n_crops, n_elecs, n_samples_in_epoch) representing the data.
    :param wavelet: Wavelet to use for the CWT transform. ("morl" or "db4")
    :param agg_func: Aggregation function to apply to the CWT features. ("median")
    :param fs: Sampling frequency of the crops.

    :return: DWT features with shape (n_crops, n_features) where n_features is determined by the aggregation function.

    """
    crops = asarray(crops)

    (n_crops, n_elecs, n_samples_in_epoch) = crops.shape

    wt = WaveletTransformer()

    dwt_coefficients = wt.dwt_transform(crops=crops, wavelet=wavelet, fs=fs)

    dwt_feats = ndarray(shape=(n_crops, 7, len(dwt_coefficients), n_elecs))

    for level_id, level_coeffs in enumerate(dwt_coefficients):
        level_coeffs = abs(level_coeffs)

        dwt_feats[:, 0, level_id, :] = fw.bounded_variation(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 1, level_id, :] = fw.entropy(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 2, level_id, :] = fw.maximum(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 3, level_id, :] = fw.mean(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 4, level_id, :] = fw.minimum(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 5, level_id, :] = fw.power(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 6, level_id, :] = fw.variance(coefficients=level_coeffs, axis=2)

    powers = dwt_feats[:, 5, :, :]
    ratios = powers / sum(powers, axis=1, keepdims=True)
    dwt_feats[:, 7, :, :] = ratios

    dwt_features = dwt_feats.reshape(n_crops, -1)
    if agg_func is not None:
        dwt_features = agg_func(dwt_features, axis=0)

    return dwt_features
