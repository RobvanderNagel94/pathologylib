from features.multi_channel.abstract_feature_generator import AbstractFeatureGenerator
from features.multi_channel import features_frequency as ff, features_frequency
import numpy as np


class FrequencyFeatureGenerator(AbstractFeatureGenerator):

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
                        str(lower) + '-' + str(upper) + 'Hz',
                        str(electrode)])
                    feature_labels.append(label)
        return feature_labels

    def convert_with_fft(self, weighted_epochs):
        epochs_amplitudes = np.abs(np.fft.rfft(weighted_epochs, axis=2))
        epochs_amplitudes /= weighted_epochs.shape[-1]
        return epochs_amplitudes

    def generate_features(self, weighted_epochs):
        (n_epochs, n_elecs, n_samples_in_epoch) = weighted_epochs.shape
        epochs_psds = self.convert_with_fft(weighted_epochs)
        freq_bin_size = self.sfreq / n_samples_in_epoch
        freqs = np.fft.fftfreq(int(n_samples_in_epoch), 1. / self.sfreq)
        freq_feats = np.ndarray(shape=(n_epochs, len(self.freq_feats),
                                       len(self.bands), n_elecs))

        for freq_feat_id, freq_feat_name in enumerate(self.freq_feats):
            if freq_feat_name == "power_ratio":
                powers = freq_feats[:, self.freq_feats.index("power"), :, :]
                func = getattr(features_frequency, freq_feat_name)
                ratio = func(powers, axis=-2)
                freq_feats[:, freq_feat_id, :, :] = ratio

            elif freq_feat_name == "entropy":
                func = getattr(features_frequency, freq_feat_name)
                ratios = freq_feats[:, self.freq_feats.index("power_ratio"), :, :]
                spec_entropy = func(ratios)
                freq_feats[:, freq_feat_id, :, :] = spec_entropy
            else:
                func = getattr(features_frequency, freq_feat_name)

                band_psd_features = np.ndarray(shape=(n_epochs, len(self.bands),
                                                      n_elecs))
                for band_id, (lower, upper) in enumerate(self.bands):
                    lower_bin, upper_bin = (int(lower / freq_bin_size),
                                            int(upper / freq_bin_size))

                    if upper_bin >= len(freqs):
                        upper_bin = len(freqs) - 1
                    band_psds = np.take(epochs_psds, range(lower_bin, upper_bin), axis=-1)
                    band_psd_features[:, band_id, :] = func(band_psds, axis=-1)

                freq_feats[:, freq_feat_id, :, :] = band_psd_features

        freq_feats = freq_feats.reshape(n_epochs, -1)
        if self.agg_mode:
            freq_feats = self.agg_mode(freq_feats, axis=0)

        return freq_feats

    def __init__(self, elecs, agg, bands, sfreq, domain="fft"):
        super(FrequencyFeatureGenerator, self).__init__(
            domain=domain, electrodes=elecs, agg_mode=agg)
        self.freq_feats = sorted([
            feat_func
            for feat_func in dir(features_frequency)
            if not feat_func.startswith('_')])
        self.bands = bands
        self.sfreq = sfreq


class FourierTransformer(object):
    def __init__(self):
        pass

    def convert_with_fft(self, crops):
        epochs_amplitudes = np.abs(np.fft.rfft(crops, axis=2))
        epochs_amplitudes /= crops.shape[-1]
        return epochs_amplitudes

    def dft_transform(self, crops, sfreq, n_samples_in_epoch):
        crop_psds = self.convert_with_fft(crops=crops)
        freq_bin_size = sfreq / n_samples_in_epoch
        freqs = np.fft.fftfreq(int(n_samples_in_epoch), 1. / sfreq)
        return crop_psds, freqs, freq_bin_size


def generate_dft_features(crops: np.ndarray,
                          sfreq: int,
                          band_limits: list,
                          agg_func: any) -> np.ndarray:
    """
    Generates the DFT features from the crops of signals.

    :param crops: Array of shape (n_crops, n_elecs, n_samples_in_epoch) representing the data.
    :param sfreq: Sampling frequency of the crops.
    :param band_limits: List of band limits for the CWT.
    :param agg_func: Aggregation function to apply to the CWT features.

    :return: DFT features. Shape: (n_crops, n_freq_feats, n_bands, n_elecs)
    """

    (n_crops, n_elecs, n_samples_in_epoch) = crops.shape

    ftr = FourierTransformer()
    crops_amplitude_spectrum, freqs, freq_bin_size = ftr.dft_transform(
        crops=crops, sfreq=sfreq, n_samples_in_epoch=n_samples_in_epoch)

    freq_feats = np.ndarray(shape=(len(crops), 7 + 2, len(band_limits), n_elecs))
    for band_id, (lower, upper) in enumerate(band_limits):
        lower_bin, upper_bin = int(lower / freq_bin_size), int(upper / freq_bin_size)
        # if upper_bin corresponds to nyquist frequency or higher,
        # take last available frequency
        if upper_bin >= len(freqs):
            upper_bin = len(freqs) - 1
        band_amplitude_spectrum = np.take(crops_amplitude_spectrum, range(lower_bin, upper_bin), axis=-1)

        freq_feats[:, 0, band_id, :] = ff.maximum(power_spectrum=band_amplitude_spectrum)
        freq_feats[:, 1, band_id, :] = ff.mean(power_spectrum=band_amplitude_spectrum)
        freq_feats[:, 2, band_id, :] = ff.minimum(power_spectrum=band_amplitude_spectrum)
        freq_feats[:, 3, band_id, :] = ff.power(power_spectrum=band_amplitude_spectrum)
        freq_feats[:, 4, band_id, :] = ff.value_range(power_spectrum=band_amplitude_spectrum)
        freq_feats[:, 5, band_id, :] = ff.variance(power_spectrum=band_amplitude_spectrum)
        freq_feats[:, 6, band_id, :] = freqs[lower_bin + ff.peak_frequency(power_spectrum=band_amplitude_spectrum)]

    powers = freq_feats[:, 3, :, :]
    ratios = powers / np.sum(powers, axis=1, keepdims=True)
    freq_feats[:, 7, :, :] = ratios

    spectral_entropy = np.sum([ratio * np.log(ratio) for ratio in ratios], axis=-1)
    spectral_entropy = -1 * spectral_entropy / np.log(ratios.shape[1])
    print(np.sum(spectral_entropy, axis=-1))
    freq_feats[:, 8, :, :] = spectral_entropy

    freq_feats = freq_feats.reshape(n_crops, -1)
    if agg_func is not None:
        freq_feats = agg_func(freq_feats, axis=0)

    return freq_feats