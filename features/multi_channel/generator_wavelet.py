from pywt._extensions._pywt import (DiscreteContinuousWavelet, ContinuousWavelet, Wavelet, _check_dtype)
from pywt._functions import (integrate_wavelet, central_frequency)
from pywt import (wavedec, dwt_max_level)
from features.multi_channel import features_wavelet as fw
import numpy as np


class WaveletTransformer(object):
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

    # taken from pywt and adapted to not compute and return frequencies
    # this is available in new, separate function
    # like this, it can be applied using numpy.apply_along_axis
    def pywt_cwt(self, data, scales, wavelet):
        dt = _check_dtype(data)
        data = np.array(data, dtype=dt)
        if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
            wavelet = DiscreteContinuousWavelet(wavelet)
        if np.isscalar(scales):
            scales = np.array([scales])
        if data.ndim == 1:
            if wavelet.complex_cwt:
                out = np.zeros((np.size(scales), data.size), dtype=complex)
            else:
                out = np.zeros((np.size(scales), data.size))
            for i in np.arange(np.size(scales)):
                precision = 10
                int_psi, x = integrate_wavelet(wavelet, precision=precision)
                step = x[1] - x[0]
                j = np.floor(np.arange(
                    scales[i] * (x[-1] - x[0]) + 1) / (scales[i] * step))
                if np.max(j) >= np.size(int_psi):
                    j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
                coef = - np.sqrt(scales[i]) * np.diff(
                    np.convolve(data, int_psi[j.astype(np.int)][::-1]))
                d = (coef.size - data.size) / 2.
                out[i, :] = coef[int(np.floor(d)):int(-np.ceil(d))]
            return out
        else:
            raise ValueError("Only dim == 1 supported")

    def cwt_transform(self, crops, wavelet, band_limits, sfreq):
        scales = self.freqs_to_scale(
            freqs=band_limits, wavelet=wavelet, sfreq=sfreq)
        coefficients = np.apply_along_axis(
            func1d=self.pywt_cwt, axis=2, arr=crops, scales=scales,
            wavelet=wavelet)
        coefficients = np.swapaxes(a=coefficients, axis1=1, axis2=2)
        return coefficients

    def dwt_transform(self, crops, wavelet, sfreq):
        (n_windows, n_elecs, n_samples_in_window) = crops.shape
        max_level = dwt_max_level(data_len=n_samples_in_window,
                                  filter_len=wavelet)
        pseudo_freqs = [sfreq / 2 ** i for i in range(1, max_level)]
        pseudo_freqs = [pseudo_freq for pseudo_freq in pseudo_freqs if pseudo_freq >= 2]
        n_levels = len(pseudo_freqs)
        multi_level_coeffs = wavedec(data=crops, wavelet=wavelet, level=n_levels - 1, axis=2)
        return multi_level_coeffs


def generate_cwt_features(crops: np.ndarray,
                          wavelet: str,
                          band_limits: list,
                          agg_func: any,
                          sfreq: int) -> np.ndarray:
    """
    Generate Continuous Wavelet Transform (CWT) features for input crops.

    :param crops: Array of shape (n_crops, n_elecs, n_samples_in_epoch) representing the data.
    :param wavelet: Wavelet to use for the CWT transform. ("morl" or "db4")
    :param band_limits: List of band limits for the CWT
    :param agg_func: Aggregation function to apply to the CWT features. ("median")
    :param sfreq: Sampling frequency of the crops.

    :return: CWT features with shape (n_crops, n_features) where n_features is determined by the aggregation function.
    """
    (n_crops, n_elecs, n_samples_in_epoch) = crops.shape

    wt = WaveletTransformer()

    cwt_coefficients = wt.cwt_transform(crops=crops,
                                        wavelet=wavelet,
                                        band_limits=band_limits,
                                        sfreq=sfreq)

    cwt_feats = np.ndarray(shape=(n_crops, 7, len(cwt_coefficients), n_elecs))

    cwt_feats[:, 0, :, :] = fw.bounded_variation(coefficients=cwt_coefficients, axis=2)
    cwt_feats[:, 1, :, :] = fw.entropy(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 2, :, :] = fw.maximum(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 3, :, :] = fw.mean(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 4, :, :] = fw.minimum(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 5, :, :] = fw.power(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 6, :, :] = fw.variance(coefficients=cwt_coefficients, axis=3)

    powers = cwt_feats[:, 5, :, :]
    ratios = powers / np.sum(powers, axis=1, keepdims=True)
    cwt_feats[:, 7, :, :] = ratios

    cwt_feats = cwt_feats.reshape(n_crops, -1)
    if agg_func is not None:
        cwt_feats = agg_func(cwt_feats, axis=0)

    return cwt_feats


def generate_dwt_features(crops: np.ndarray,
                          wavelet: str,
                          agg_func: any,
                          sfreq: int) -> np.ndarray:
    """
    Generate Discrete Wavelet Transform (DWT) features for input crops.

    :param crops: Array of shape (n_crops, n_elecs, n_samples_in_epoch) representing the data.
    :param wavelet: Wavelet to use for the CWT transform. ("morl" or "db4")
    :param agg_func: Aggregation function to apply to the CWT features. ("median")
    :param sfreq: Sampling frequency of the crops.

    :return: DWT features with shape (n_crops, n_features) where n_features is determined by the aggregation function.

    """
    (n_crops, n_elecs, n_samples_in_epoch) = crops.shape
    wt = WaveletTransformer()
    dwt_coefficients = wt.dwt_transform(crops=crops, wavelet=wavelet, sfreq=sfreq)

    dwt_feats = np.ndarray(shape=(n_crops, 7, len(dwt_coefficients), n_elecs))

    for level_id, level_coeffs in enumerate(dwt_coefficients):
        level_coeffs = np.abs(level_coeffs)
        dwt_feats[:, 0, level_id, :] = fw.bounded_variation(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 1, level_id, :] = fw.entropy(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 2, level_id, :] = fw.maximum(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 3, level_id, :] = fw.mean(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 4, level_id, :] = fw.minimum(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 5, level_id, :] = fw.power(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 6, level_id, :] = fw.variance(coefficients=level_coeffs, axis=2)

    powers = dwt_feats[:, 5, :, :]
    ratios = powers / np.sum(powers, axis=1, keepdims=True)
    dwt_feats[:, 7, :, :] = ratios

    dwt_feats = dwt_feats.reshape(n_crops, -1)
    if agg_func is not None:
        dwt_feats = agg_func(dwt_feats, axis=0)

    return dwt_feats
