from features.multi_channel.abstract_feature_generator import AbstractFeatureGenerator
from utils.preprocessing import (split_into_epochs, filter_to_frequency_bands)
from . import features_phase as fp, features_phase

import numpy as np


class PhaseFeatureGenerator(AbstractFeatureGenerator):

    def get_feature_labels(self):
        feature_labels = []
        for sync_feat in self.sync_feats:
            for band_id, band in enumerate(self.bands):
                lower, upper = band
                for electrode_id, electrode in enumerate(self.electrodes):
                    for electrode_id2 in range(electrode_id + 1,
                                               len(self.electrodes)):
                        label = '_'.join([
                            self.domain,
                            sync_feat,
                            '-'.join([str(lower), str(upper) + 'Hz',
                                      str(electrode),
                                      str(self.electrodes[electrode_id2])])
                        ])
                        feature_labels.append(label)
        return feature_labels

    def generate_features(self, band_epochs):

        epochs_instantaneous_phases = features_phase.instantaneous_phases(band_signals=band_epochs, axis=-1)
        phase_locking_values = features_phase.phase_locking_values(inst_phases=epochs_instantaneous_phases)

        if self.agg_mode:
            phase_locking_values = self.agg_mode(phase_locking_values, axis=0)

        return phase_locking_values

    def __init__(self, elecs, agg, bands, domain="phase"):
        super(PhaseFeatureGenerator, self).__init__(
            domain=domain, electrodes=elecs, agg_mode=agg)
        self.sync_feats = ["plv"]
        self.bands = bands


def generate_phase_features(signals: np.ndarray,
                            band_limits: list,
                            sfreq: int,
                            epoch_duration_s: int,
                            outlier_mask: np.ndarray,
                            agg_func: any) -> np.ndarray:
    """
    Computes phase locking values from a given set of signals.

    :param signals: Array of shape (n_crops, n_elecs, n_samples_in_epoch) representing the data.
    :param band_limits: List of band limits for the CWT.
    :param sfreq: Sampling frequency of the crops.
    :param epoch_duration_s: Desired duration of each epoch in seconds.
    :param outlier_mask: Boolean array indicating the epochs that are considered outliers.
    :param agg_func: Aggregation function to apply to the CWT features. ("median")

    :return: Phase locking values of shape (n_bands, n_electrodes*(n_electrodes-1)//2).
    """

    band_signals = filter_to_frequency_bands(
        signals=signals, bands=band_limits, sfreq=sfreq)
    band_crops = split_into_epochs(band_signals, sfreq=sfreq,
                                   epoch_duration_s=epoch_duration_s)
    band_crops = band_crops[outlier_mask == False]

    epochs_instantaneous_phases = fp.instantaneous_phases(
        band_signals=band_crops, axis=-1)

    phase_locking_values = fp.phase_locking_values(
        inst_phases=epochs_instantaneous_phases)

    if agg_func is not None:
        # n_bands * n_signals*(n_signals-1)/2
        phase_locking_values = agg_func(phase_locking_values, axis=0)
    return phase_locking_values
