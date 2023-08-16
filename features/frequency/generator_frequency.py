from features.abstract_feature_generator import AbstractFeatureGenerator
from features.frequency import features_frequency as ff

from numpy import (asarray, ndarray, take, abs, fft, zeros, sum, log)
from typing import Union


class FrequencyFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, elecs, agg, bands, fs, domain="fft"):
        super().__init__(domain=domain, electrodes=elecs, agg_mode=agg)
        self.freq_feats = sorted(feat_func for feat_func in dir(ff) if not feat_func.startswith('_'))
        self.bands = bands
        self.fs = fs

    def get_feature_labels(self):
        feature_labels = []
        for freq_feat in self.freq_feats:
            freq_feat = freq_feat.replace("_", "-")
            for band_id, band in enumerate(self.bands):
                lower, upper = band
                for electrode in self.electrodes:
                    label = '_'.join([
                        self.domain,
                        freq_feat,
                        f"{lower}-{upper}Hz",
                        str(electrode)
                    ])
                    feature_labels.append(label)
        return feature_labels

    def convert_with_fft(self, weighted_epochs):
        epochs_amplitudes = abs(fft.rfft(weighted_epochs, axis=2))
        epochs_amplitudes /= weighted_epochs.shape[-1]
        return epochs_amplitudes

    def generate_features(self, weighted_epochs):
        n_epochs, n_elecs, n_samples_in_epoch = weighted_epochs.shape
        epochs_psds = self.convert_with_fft(weighted_epochs)
        freq_bin_size = self.fs / n_samples_in_epoch
        freqs = fft.fftfreq(int(n_samples_in_epoch), 1. / self.fs)
        freq_feats = zeros((n_epochs, len(self.freq_feats), len(self.bands), n_elecs))

        for freq_feat_id, freq_feat_name in enumerate(self.freq_feats):
            if freq_feat_name == "power_ratio":
                powers = freq_feats[:, self.freq_feats.index("power"), :, :]
                ratio = ff.power_ratio(powers, axis=-2)
                freq_feats[:, freq_feat_id, :, :] = ratio

            elif freq_feat_name == "entropy":
                ratios = freq_feats[:, self.freq_feats.index("power_ratio"), :, :]
                spec_entropy = ff.entropy(ratios)
                freq_feats[:, freq_feat_id, :, :] = spec_entropy
            else:
                func = getattr(ff, freq_feat_name)

                band_psd_features = zeros((n_epochs, len(self.bands), n_elecs))
                for band_id, (lower, upper) in enumerate(self.bands):
                    lower_bin, upper_bin = (int(lower / freq_bin_size), int(upper / freq_bin_size))
                    upper_bin = min(upper_bin, len(freqs) - 1)
                    band_psds = take(epochs_psds, range(lower_bin, upper_bin), axis=-1)
                    band_psd_features[:, band_id, :] = func(band_psds, axis=-1)

                freq_feats[:, freq_feat_id, :, :] = band_psd_features

        freq_feats = freq_feats.reshape(n_epochs, -1)
        if self.agg_mode:
            freq_feats = self.agg_mode(freq_feats, axis=0)

        return freq_feats


class FourierTransformer:
    def __init__(self):
        pass

    def convert_with_fft(self, crops):
        epochs_amplitudes = abs(fft.rfft(crops, axis=2))
        epochs_amplitudes /= crops.shape[-1]
        return epochs_amplitudes

    def dft_transform(self, crops, fs, n_samples_in_epoch):
        crop_psds = self.convert_with_fft(crops=crops)
        freq_bin_size = fs / n_samples_in_epoch
        freqs = fft.fftfreq(int(n_samples_in_epoch), 1. / fs)
        return crop_psds, freqs, freq_bin_size


def generate_dft_features(crops: Union[ndarray, list],
                          fs: int,
                          band_limits: Union[ndarray, list],
                          agg_func: any) -> ndarray:
    """
    Generates the DFT features from the crops of signals.

    :param crops: Array of shape (n_crops, n_elecs, n_samples_in_epoch) representing the data.
    :param fs: Sampling frequency of the crops.
    :param band_limits: List of band limits for the CWT.
    :param agg_func: Aggregation function to apply to the CWT features.

    :return: DFT features. Shape: (n_crops, n_freq_feats, n_bands, n_elecs)
    """

    crops = asarray(crops)

    (n_crops, n_elecs, n_samples_in_epoch) = crops.shape

    ftr = FourierTransformer()

    crops_amplitude_spectrum, fs, freq_bin_size = ftr.dft_transform(
        crops=crops, fs=fs, n_samples_in_epoch=n_samples_in_epoch)

    dft_feats = ndarray(shape=(len(crops), 7 + 2, len(band_limits), n_elecs))

    for band_id, (lower, upper) in enumerate(band_limits):
        lower_bin, upper_bin = int(lower / freq_bin_size), int(upper / freq_bin_size)
        # if upper_bin corresponds to nyquist frequency or higher,
        # take last available frequency
        if upper_bin >= len(fs):
            upper_bin = len(fs) - 1

        band_amplitude_spectrum = take(crops_amplitude_spectrum, range(lower_bin, upper_bin), axis=-1)

        dft_feats[:, 0, band_id, :] = ff.maximum(power_spectrum=band_amplitude_spectrum)
        dft_feats[:, 1, band_id, :] = ff.mean(power_spectrum=band_amplitude_spectrum)
        dft_feats[:, 2, band_id, :] = ff.minimum(power_spectrum=band_amplitude_spectrum)
        dft_feats[:, 3, band_id, :] = ff.power(power_spectrum=band_amplitude_spectrum)
        dft_feats[:, 4, band_id, :] = ff.value_range(power_spectrum=band_amplitude_spectrum)
        dft_feats[:, 5, band_id, :] = ff.variance(power_spectrum=band_amplitude_spectrum)
        dft_feats[:, 6, band_id, :] = fs[lower_bin + ff.peak_frequency(power_spectrum=band_amplitude_spectrum)]

    powers = dft_feats[:, 3, :, :]
    ratios = powers / sum(powers, axis=1, keepdims=True)
    dft_feats[:, 7, :, :] = ratios

    dft_features = dft_feats.reshape(n_crops, -1)
    if agg_func is not None:
        dft_features = agg_func(dft_features, axis=0)

    return dft_features
