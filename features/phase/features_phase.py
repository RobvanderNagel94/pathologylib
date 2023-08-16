from numpy import (ndarray, array)
import numpy as np
import scipy


def instantaneous_phases(band_signals: ndarray, axis: int) -> ndarray:
    """
    Computes the instantaneous phase of a band signal.

    :param band_signals: Signal values,
        Shape of array should be (n_windows, n_bands, n_signals, n_samples).
    :param axis: Axis along which the operation is performed.

    :return: Instantaneous phase of the band signal.
        Shape of array is (n_windows, n_bands, n_signals, n_samples).

    References
    ----------
    https://en.wikipedia.org/wiki/Instantaneous_phase_and_frequency
    """

    analytical_signal = scipy.signal.hilbert(band_signals, axis=axis)
    return np.unwrap(np.angle(analytical_signal), axis=axis)


def phase_locking_value(inst_phases: ndarray) -> ndarray:
    """
    Computes the Phase Locking Values for each channel pair.

    :param inst_phases: Instantaneous phase of the band signal.
        Shape of array should be (n_windows, n_bands, n_signals, n_samples).

    :return: Phase Locking Values for each channel pair.
        Shape of array is (n_windows, n_bands * (n_signals * (n_signals - 1)) // 2).

    References
    ----------
    https://arxiv.org/ftp/arxiv/papers/1710/1710.08037.pdf
    """

    (n_windows, n_bands, n_signals, n_samples) = inst_phases.shape
    plvs = []
    for electrode_id1 in range(n_signals):
        for electrode_id2 in range(electrode_id1 + 1, n_signals):
            for band_id in range(n_bands):
                plv = phase_locking_value2(
                    theta1=inst_phases[:, band_id, electrode_id1],
                    theta2=inst_phases[:, band_id, electrode_id2]
                )
                plvs.append(plv)

    return array(plvs).T


def phase_locking_value2(theta1: ndarray, theta2: ndarray) -> ndarray:
    """
    Computes the intermediate normalised Phase Locking Value (PLV) between two instantaneous phases.

    :param theta1: Instantaneous phases of signal 1.
    :param theta2: Instantaneous phases of signal 2.

    :return: Normalised phase locking values between the two signals.
    """
    delta = np.subtract(theta1, theta2)
    xs_mean = np.mean(np.cos(delta), axis=-1)
    ys_mean = np.mean(np.sin(delta), axis=-1)

    return np.linalg.norm([xs_mean, ys_mean], axis=0)
