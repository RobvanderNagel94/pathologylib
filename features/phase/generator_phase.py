from features.phase import features_phase as fp
from features.abstract_feature_generator import AbstractFeatureGenerator
from features.preprocessing import (split_into_epochs, filter_to_frequency_bands)
from numpy import ndarray


class PhaseFeatureGenerator(AbstractFeatureGenerator):
    """Class that generates phase features."""

    def __init__(self, elecs, agg, bands, domain="phase"):
        super().__init__(domain=domain, electrodes=elecs, agg_mode=agg)
        self.sync_feats = ["plv"]
        self.bands = bands

    def get_feature_labels(self):
        feature_labels = []
        for sync_feat in self.sync_feats:
            for band_id, band in enumerate(self.bands):
                lower, upper = band
                for electrode_id, electrode in enumerate(self.electrodes):
                    for electrode_id2 in range(electrode_id + 1, len(self.electrodes)):
                        label = '_'.join([
                            self.domain,
                            sync_feat,
                            f"{lower}-{upper}Hz",
                            str(electrode),
                            str(self.electrodes[electrode_id2]),
                        ])
                        feature_labels.append(label)
        return feature_labels

    def generate_features(self, band_epochs):
        inst_phases = fp.instantaneous_phases(band_epochs, axis=-1)
        plv_values = fp.phase_locking_values(inst_phases)
        if self.agg_mode:
            plv_values = self.agg_mode(plv_values, axis=0)
        return plv_values


def generate_phase_features(signals: ndarray,
                            band_limits: list,
                            fs: int,
                            epoch_duration_s: int,
                            outlier_mask: ndarray,
                            agg_func: any) -> ndarray:
    """
    Computes phase locking values from a given set of signals.

    :param signals: Array of shape (n_crops, n_elecs, n_samples_in_epoch) representing the data.
    :param band_limits: List of band limits for the CWT.
    :param fs: Sampling frequency of the crops.
    :param epoch_duration_s: Desired duration of each epoch in seconds.
    :param outlier_mask: Boolean array indicating the epochs that are considered outliers.
    :param agg_func: Aggregation function to apply to the features.

    :return: Phase locking values of shape (n_bands, n_electrodes*(n_electrodes-1)//2).
    """

    band_signals = filter_to_frequency_bands(
        signals=signals, bands=band_limits, fs=fs)

    band_crops = split_into_epochs(signals=band_signals, fs=fs, epoch_duration_s=epoch_duration_s)
    band_crops = band_crops[outlier_mask == False]

    epochs_instantaneous_phases = fp.instantaneous_phases(band_signals=band_crops, axis=-1)
    phase_locking_values = fp.phase_locking_values(inst_phases=epochs_instantaneous_phases)

    if agg_func is not None:
        phase_locking_values = agg_func(phase_locking_values, axis=0)
    return phase_locking_values
