import numpy as np
from sklearn.cluster import KMeans
import scipy.signal as signal
from scipy.signal import welch
from scipy.optimize import least_squares


def estimate_welch(x: np.ndarray,
                   fs: float = 250,
                   window: str = 'hann',
                   noverlap: int = 256,
                   nfft: int = 512,
                   nperseg: int = 512,
                   fmin: int = 1,
                   fmax: int = 49
                   ) -> (np.ndarray, np.ndarray):
    """
    Estimates power spectral density using Welch's averaged periodogram method.
    The method splits the series into overlapping segments, computes periodograms for each segment,
    and then averages the periodograms.

    :param x: 1D array of signal values.
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param window: Window function, default 'hann'.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.
    :param fmin: Minimum frequency bound for power spectrum, default 0.
    :param fmax: Maximum frequency bound for power spectrum, default 40.

    :return: Bounded frequency spectrum.
    :return: Power spectral density.

    Raises
    ------
    Exception
        if input length is less than nperseg
        if fmin is less than 1 or greater than 50
        if fmax is less than 1 or greater than 50
        if fmax is not greater than fmin

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    """

    assert len(x) >= nperseg, "Exception: input length = {} must be greater than window = {}.".format(len(x), nperseg)
    assert 1 <= fmin <= 50, "Exception: fmin = {} must be between 1 and 50.".format(fmin)
    assert 1 <= fmax <= 50, "Exception: fmax = {} must be between 1 and 50.".format(fmax)
    assert fmax > fmin, "Exception: fmax = {} must be greater than fmin = {}.".format(fmax, fmin)

    f, Sxx = welch(x,
                   fs=fs,
                   window=window,
                   noverlap=noverlap,
                   nfft=nfft,
                   nperseg=nperseg
                   )

    # bound spectrum
    f_lim = f[(f >= fmin) & (f <= fmax)]
    start = f.tolist().index(f_lim[0])
    end = f.tolist().index(f_lim[-1]) + 1

    return f[start:end], Sxx[start:end]


def estimate_alpha_frequency_components(x: np.ndarray,
                                        fs: float = 250,
                                        window: str = 'hann',
                                        noverlap: int = 256,
                                        nfft: int = 512,
                                        nperseg: int = 512,
                                        fmin: int = 2,
                                        fmax: int = 18):
    """
    Finds dominant peak frequencies in segments of EEG data.

    :param x: 1D array of signal values, passed as single segment.
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param window: Window function, default 'hann'.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.
    :param fmin: Minimum frequency bound for power spectrum, default 2.
    :param fmax: Maximum frequency bound for power spectrum, default 18.

    :return: Alpha rhythm frequency components.

    Notes
    -------
    The function first uses `scipy.signal.find_peaks` to guess an initial peak location,
    then performs a least-squares fit to localize the two most dominant peak locations, including their amplitudes and widths.
    The spectral curve is defined as:

        Plog(f) ≈ Pcurve(f) = Ppk1(f) + Ppk2(f) + Pbg(f)
        Ppk1(f) = A1 * exp((f − f1)^2 / Δ1^2)
        Ppk2(f) = A2 * exp((f − f2)^2 / Δ2^2)
        Pbg(f) = B − C * log(f)

    where `A1` and `A2` are the amplitudes, `f1` and `f2` the center frequencies,
    `Δ1` and `Δ2` the widths, `C` is a power-law approximation, and `B` a normalization factor.

    References
    ----------
    Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    DOI 10.1016/j.clinph.2012.07.007
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    """

    def bg_func(x, t, y):
        bg = x[0] - x[1] * np.log10(t)
        return y - bg

    def pk1_func(x, t, y):
        bg = x[0] - C * np.log10(t)
        pk1 = x[1] * np.exp(-np.power(t - x[2], 2) / np.power(x[3], 2))
        return y - pk1 - bg

    def pk2_func(x, t, y):
        bg = x[0] - C * np.log10(t)
        pk1 = A1 * np.exp(-np.power(t - f1, 2) / np.power(d1, 2))
        pk2 = x[1] * np.exp(-np.power(t - x[2], 2) / np.power(x[3], 2))
        return y - pk1 - pk2 - bg

    def find_two_highest_peaks(x):
        """ Find the two highest peaks in the log transformed power spectrum `x`. """

        peaks, _ = signal.find_peaks(x)
        prominences = signal.peak_prominences(x, peaks)[0]

        # Sort the peaks and prominences by their prominences in descending order
        sorted_idx = np.argsort(-prominences)
        sorted_peaks = peaks[sorted_idx]
        sorted_prominences = prominences[sorted_idx][::-1]

        # Extract the first and second most significant peak’s locations and amplitudes
        if len(sorted_peaks) >= 2:
            f1, f2 = sorted_peaks[:2]
            A1, A2 = sorted_prominences[:2]
        elif len(sorted_peaks) == 1:
            f1 = sorted_peaks[0]
            A1 = sorted_prominences[0]
            f2, A2 = 0, 0
        else:
            raise ValueError("Could not find any peaks in X = {}".format(x))

        return f1, f2, A1, A2

    def optimize_components(f, x, f1, f2, A1, A2, fmin):
        """ Optimize alpha rhythm frequency components using Levenberg–Marquardt."""

        B, C = 0, 0
        approx_bg = least_squares(bg_func,
                                  np.array([B, C]),
                                  args=(f, x),
                                  method='lm')
        B, _ = approx_bg.x  # keep C fixed

        approx_pk1 = least_squares(pk1_func,
                                   np.array([B, A1, f1, 1]),
                                   args=(f, x),
                                   method='lm')
        B, A1, f1, d1 = approx_pk1.x

        approx_pk2 = least_squares(pk2_func,
                                   np.array([B, A2, f2, 1]),
                                   args=(f, x),
                                   method='lm')
        B, A2, f2, d2 = approx_pk2.x

        # Check if A2 is zero, and set f2 and d2 to zero
        if A2 == 0.:
            f2, d2 = 0, 0
        else:
            # Adjust for frequency components
            f1, f2 = f1 + fmin, f2 + fmin

        return B, C, A1, A2, f1, f2, d1, d2

    # Estimate Welch's discrete coefficients
    f, Sxx = estimate_welch(x, fs, window, noverlap, nfft, nperseg, fmin, fmax)

    # Approximate the two highest peaks and extract components
    f1, f2, A1, A2 = find_two_highest_peaks(np.log(Sxx))

    # Optimize components using Levenberg–Marquardt's least-squares method
    B, C, A1, A2, f1, f2, d1, d2 = optimize_components(f, np.log(Sxx), f1, f2, A1, A2)

    return B, C, A1, A2, f1, f2, d1, d2


def find_correlation_coefficient(Sxx: np.ndarray,
                                 B: float,
                                 C: float,
                                 f1: float,
                                 f2: float,
                                 d1: float,
                                 d2: float) -> float:
    """
    Calculates the correlation coefficient between a spectrum with and without the extracted peaks.

    :param Sxx: Spectrum of the signals.
    :param B: Normalization factor.
    :param C: Power-law approximation.
    :param f1: Center frequency of the first data point.
    :param f2: Center frequency of the second data point.
    :param d1: Width of the first data point.
    :param d2: Width of the second data point.

    :return: Correlation coefficient between the original and processed spectrums.
    """

    x = Sxx.tolist()

    Pexcl = np.copy(x)

    idx = ((f1 - d1 <= x.index) & (x.index <= f1 + d1)) | ((f2 - d2 <= x.index) & (x.index <= f2 + d2))
    Pexcl[idx] = 0
    Pres = np.copy(Pexcl)

    Pincl = np.exp(B - C * np.log(x[idx].tolist()))
    Pres[idx] = Pincl

    return 1 - np.corrcoef(Pexcl, Pres)[0, 1]


def estimate_alpha_rhythm(A: np.ndarray,
                          f: np.ndarray,
                          c: np.ndarray,
                          d: np.ndarray) -> float:
    """
    Estimates the alpha rhythm frequency from four arrays of inputs.

    :param A: Amplitude values.
    :param f: Frequency values.
    :param c: Correlation values.
    :param d: Distance values.

    :return: Estimated alpha rhythm frequency.
    """

    combined_components = [[A[i], f[i], c[i], d[i]] for i in range(len(A))]

    TH = 1.5
    combined_components = [x for x in combined_components if 2 <= x[1] <= 18 and x[3] <= TH]
    X = np.array(combined_components)

    if len(combined_components) <= 2:
        raise ValueError("Could not find clusters for length of {} components".format(len(combined_components)))

    clusters = [1, 2]
    SSE = []
    for cluster in clusters:
        kmeans = KMeans(n_clusters=cluster, random_state=0).fit(X)
        SSE.append(kmeans.inertia_)

    optimal_clusters = clusters[np.argmin(SSE)]
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(X)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    if len(labels) < 2:
        largest_cluster = labels[np.argmax(counts)]
        largest_cluster_elements = X[kmeans.labels_ == largest_cluster]
    else:
        largest_clusters = np.argsort(counts)[::-1][:2]
        largest_cluster_elements = np.concatenate([X[kmeans.labels_ == lc] for lc in largest_clusters])

    ck = np.mean(largest_cluster_elements[:, 2])
    w = largest_cluster_elements[:, 2] / ck

    return np.sum(largest_cluster_elements[:, 1] * w)


def alpha_rhythm_anterio_posterior_gradient(x: dict, fmin: int = 8, fmax: int = 12) -> float:
    """
    Computes the anterior to posterior ratio (gradient) of alpha power from segments
    annotated with the eyes closed state.

    :param x: Dictionary of EEG data, key=channels, values=signals
        (requires output of a common reference montage, annotated with the eyes closed state)

    :return: Quantified and normalised value for the anterio-posterior gradient.

    Notes
    --------

    If Qapg < 0.4: Alpha power gradient is categorized as normal.
    If 0.4 < Qapg < 0.6: Alpha power gradient is considered moderately differentiated.
    If Qapg > 0.6: Alpha power gradient is considered pathological.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    channels_anterior = ['fp1', 'fp2', 'f7', 'f8', 'f3', 'fz', 'f4']
    channels_posterior = ['t5', 't6', 'p3', 'p4', 'pz', 'o1', 'o2']

    channels = channels_anterior + channels_posterior
    error_msg = f"Exception: channel names = {list(x.keys())} must be in = {channels}."
    assert all(ch in channels for ch in x), error_msg

    Pxx_ant = [np.sum(np.sum(estimate_welch(x[ch], fmin, fmax)[1])) for ch in channels_anterior]
    Pxx_pos = [np.sum(np.sum(estimate_welch(x[ch], fmin, fmax)[1])) for ch in channels_posterior]

    Pant = np.sum(Pxx_ant)
    Ppos = np.sum(Pxx_pos)

    return float(Pant / (Pant + Ppos))


def diffuse_slow_wave_activity(x: dict, fmin: int = 2, fmax_low: int = 8, fmax_wide: int = 25) -> float:
    """
    Computes diffuse slow-wave activity for segments with the eyes closed state.

    :param x: Dictionary of EEG data, key=channels, values=signals
        (requires output of a common reference montage, annotated with the eyes closed state)

    :return: Quantified and normalised value for diffused slow-wave activity.

    Notes
    -------
    Diffused slowing results in an increased power over the Delta [0-4]Hz and Theta [4-8]Hz bands
    and decreased power in the Alpha [8-12]Hz and Beta bands [13-25]Hz.
    Using the guideline above, the mean spectrum of the EEG is calculated and the power ratio between
    Plow = {2. . .8} Hz and Pwide = {2. . .25} Hz

    If Qslow > 0.6: EEG is categorized as pathological.
    If Qslow < 0.6: EEG is categorized as non-pathological.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    channels = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz',
                'c4', 't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2']

    error_msg = "Exception: channel names = {} must be in = {}."
    assert all(ch in channels for ch in x), error_msg.format(x.keys(), channels)

    Pxx_low = []
    Pxx_wide = []
    for ch in channels:
        _, Pxx = estimate_welch(x[ch], fmin=fmin, fmax=fmax_low)
        Pxx_low.append(Pxx)
        _, Pxx = estimate_welch(x[ch], fmin=fmin, fmax=fmax_wide)
        Pxx_wide.append(Pxx)

    Pwide = np.array(Pxx_wide).sum(axis=0).sum()
    Plow = np.array(Pxx_low).sum(axis=0).sum()

    return float(Plow / Pwide)


def interhemispheric_asymmetries(x: dict, fmin: int = 0.5, fmax: int = 15) -> np.ndarray:
    """
    Computes asymmetrical background patterns by comparing rhythmic activity
    between the two hemispheres in corresponding left and right channel pairs.

    :param x: Dictionary of EEG data, key=channels, values=signals
        (requires output of a common reference montage, annotated with the eyes closed state)

    :return: Quantified and normalised asymmetry values for each left and right channel pair.

    Notes
    -------

    If Qasym > 0.5: non-pathological asymmetry.
    If Qasym < 0.5: pathological asymmetry.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    left = ['fp1', 'f7', 'f3', 't3', 'c3', 't5', 'p3', 'o1']
    right = ['fp2', 'f8', 'f4', 't4', 'c4', 't6', 'p4', 'o2']

    channels = [it for sub in [left, right] for it in sub]
    error_msg = "Exception: channel names = {} must be in = {}."
    assert all(ch in channels for ch in x), error_msg.format(x.keys(), channels)

    Pxx_left = []
    Pxx_right = []
    for ch in left:
        _, Pxx = estimate_welch(x[ch], fmin=fmin, fmax=fmax)
        Pxx_left.append(Pxx)
    for ch in right:
        _, Pxx = estimate_welch(x[ch], fmin=fmin, fmax=fmax)
        Pxx_right.append(Pxx)

    asymmetries = []
    for i in range(len(Pxx_left)):
        L = np.array(Pxx_left)[i].sum()
        R = np.array(Pxx_right)[i].sum()
        asymmetries.append((R - L) / (R + L))

    return np.array(asymmetries)


def alpha_rhythm_reactivity(x_open: np.ndarray, x_closed: np.ndarray, pkf: float) -> float:
    """
    Computes reactivity for the dominant peak frequency between the eyes closed and open states.

    :param x_open: Dictionary of EEG data, key=channels, values=signals
        (requires output of a common reference montage from the O1 channel, annotated with the eyes open state)
    :param x_closed: Dictionary of EEG data, key=channels, values=signals
        (requires output of a common reference montage from the O1 channel, annotated with the eyes closed state)
    :param pkf: Estimated peak frequency.

    :return: Quantified and normalised value for alpha power reactivity.

    Notes
    -------
    Using the estimated PDR peak value, the reactivity is calculated by
    constructing a 0.5 Hz frequency band on the estimated dominant frequency
    when the eyes are open and when the eyes are closed. Based on these values,
    a normalised value is found which quantifies the reactivity of the PDR.

    If Qreac > 0.5: Substantial reactivity.
    If 0.1 < Qreac < 0.5: low reactivity.
    If Qreac < 0.1: absent reactivity.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    assert 2 < pkf < 18, "Exception: pkf = {} must be between 2 and 18.".format(pkf)

    # Compute narrow frequency bound
    fmin = pkf - 0.5
    fmax = pkf + 0.5

    _, Pxx_closed = estimate_welch(x_closed, fmin=fmin, fmax=fmax)
    _, Pxx_open = estimate_welch(x_open, fmin=fmin, fmax=fmax)

    Pec = Pxx_closed.sum()
    Peo = Pxx_open.sum()

    return float(1 - (Peo / Pec))
