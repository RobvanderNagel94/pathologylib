from scipy.signal import (find_peaks, peak_prominences)
from scipy.optimize import least_squares
from sklearn.cluster import KMeans

from numpy import (ndarray, asarray)
import numpy as np


def peak_components(Sxx_log: ndarray) -> (float, float, float, float):
    """
    Find dominant peak locations in the log transformed spectrum `Sxx_log`.
    Returns peak components: peak amplitudes `A1` and `A2` and
    center frequencies `f1` and `f2`.

    :param Sxx_log: Array of the log transformed spectrum of shape (n,)
        bounded between [fmin, fmax] = [1, 18].
    :return: f1, f2, A1, A2

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    """

    peaks, _ = find_peaks(Sxx_log)
    prominences = peak_prominences(Sxx_log, peaks)[0]

    sorted_idx = np.argsort(-prominences)
    sorted_peaks = peaks[sorted_idx]
    sorted_prominences = prominences[sorted_idx][::-1]

    if len(sorted_peaks) >= 2:
        f1, f2 = sorted_peaks[:2]
        A1, A2 = Sxx_log[sorted_peaks[0]], Sxx_log[sorted_peaks[1]]
    elif len(sorted_peaks) == 1:
        f1 = sorted_peaks[0]
        A1 = Sxx_log[sorted_peaks[0]]
        f2, A2 = 0, 0
    else:
        raise ValueError(f"Could not find any peaks in Sxx_log")

    return f1, f2, A1, A2


def fit_curve(Sxx_log: ndarray,
              f: ndarray,
              f1: float,
              f2: float,
              A1: float,
              A2: float,
              fmin: int = 1
              ) -> (float, float, float, float, float, float, float, float):


    """
    Performs a least-squares fit to localize the most dominant peak locations,
    including their amplitudes and widths.

    :param Sxx_log: Array of the log transformed power spectrum of shape (n,)
        bounded between [fmin, fmax] = [1, 18].
    :param f: Array of the transformed frequency spectrum of shape (n,)
        bounded between [fmin, fmax] = [1, 18].
    :param f1: Center frequency of the most significant peak.
    :param f2: Center frequency of the second most significant peak.
    :param A1: Peak amplitude of the most significant peak.
    :param A2: Peak amplitude of the second most significant peak.
    :param fmin: Minimum frequency bound for power spectrum, default 1.

    :return: B, C, A1, A2, f1, f2, d1, d2


    Notes
    ----------
    The spectral curve is defined as:

    Plog(f) ≈ Pcurve(f) = Ppk1(f) + Ppk2(f) + Pbg(f)
    Ppk1(f) = A1 * exp((f − f1)^2 / d1^2)
    Ppk2(f) = A2 * exp((f − f2)^2 / d2^2)
    Pbg(f) = B − C * log(f)

    where `A1` and `A2` are the amplitudes, `f1` and `f2` the center frequencies,
    `d1` and `d2` the widths, `C` is a power-law approximation, and `B` a normalization factor.


    References
    ----------
    Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    DOI 10.1016/j.clinph.2012.07.007
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

    # Optimize using Levenberg–Marquardt least-squares method.
    B, C = 0, 0
    approx_bg = least_squares(bg_func,
                              asarray([B, C]),
                              args=(f, Sxx_log),
                              method='lm')
    B, _ = approx_bg.x  # keep C fixed

    approx_pk1 = least_squares(pk1_func,
                               asarray([B, A1, f1, 1]),
                               args=(f, Sxx_log),
                               method='lm')
    B, A1, f1, d1 = approx_pk1.x

    approx_pk2 = least_squares(pk2_func,
                               asarray([B, A2, f2, 1]),
                               args=(f, Sxx_log),
                               method='lm')
    B, A2, f2, d2 = approx_pk2.x

    # Check if A2 is zero, set f2 and d2 to zero
    if A2 == 0.:
        f2, d2 = 0., 0.

    f1, f2 = f1 + fmin, f2 + fmin

    return B, C, A1, A2, f1, f2, d1, d2


def signal_to_noise(Sxx_log: ndarray,
                    B: float,
                    C: float,
                    f1: float,
                    f2: float,
                    d1: float,
                    d2: float) -> float:
    """
    Calculates the correlation coefficient between a power spectrum with and without extracted peaks.

    :param Sxx_log: Array of the log transformed power spectrum of shape (n,)
        bounded between [fmin, fmax] = [1, 18].
    :param B: Normalization factor.
    :param C: Power-law approximation.
    :param f1: Center frequency of the most significant peak.
    :param f2: Center frequency of the second most significant peak.
    :param d1: Peak width of the most significant peak.
    :param d2: Peak width of the second most significant peak.

    :return: Correlation coefficient between the original and processed power spectra.

    Notes
    ----------
    Having updated the peak parameters, a power ratio is found between the estimated peak
    components and other, less dominant, peaks found in the localized EEG
    segments. The center frequencies f1 and f2 are allocated, and a
    frequency range is found around them based on their center widths
    d1 and d2. A new spectrum Pres is found excluding these frequency ranges:

    Pres(f) = {P(f)                  , f /∈ Rpeaks
               exp(B − C · log(f))   , f ∈  Rpeaks}


    with Rpeaks ⊆ [fmin, fmax] being the frequency range around
    the dominant frequency components. Given the original spectrum P
    and new spectrum Pres, a correlation coefficient was computed:

    c = 1 − corr(P, Pres)

    If Pres is noisy or multiple significant peaks are present, the correlation
    parameter is low. Conversely, if single peaks contributed
    most of the power in Plog, a high value for c is obtained.
    """

    x = Sxx_log.tolist()

    Pexcl = np.copy(x)

    idx = ((f1 - d1 <= x.index) & (x.index <= f1 + d1)) | ((f2 - d2 <= x.index) & (x.index <= f2 + d2))
    Pexcl[idx] = 0

    Pres = np.copy(Pexcl)

    Pincl = np.exp(B - C * np.log10(x[idx].tolist()))
    Pres[idx] = Pincl

    return 1 - np.corrcoef(Pexcl, Pres)[0, 1]


def peak_estimation(A1: ndarray,
                    A2: ndarray,
                    f1: ndarray,
                    f2: ndarray,
                    d1: ndarray,
                    d2: ndarray,
                    c1: ndarray,
                    c2: ndarray) -> float:
    """
    Estimates the alpha rhythm frequency from arrays of estimated components per segment.

    :param A1: Array of amplitude components of each first extracted peak.
    :param A2: Array of amplitude components of each second extracted peak.
    :param f1: Array of frequency components of each first extracted peak.
    :param f2: Array of frequency components of each second extracted peak.
    :param d1: Array of width components of each first extracted peak.
    :param d2: Array of width components of each second extracted peak.
    :param c1: Array of computed correlation coefficients for the first extracted peak.
    :param c2: Array of computed correlation coefficients for the second extracted peak.

    :return: Estimated alpha rhythm frequency.

    Notes
    ----------

    Let all the clusters formed be denoted as {Cj}j∈{1,...,N}
    with cluster Cj = {Ai, fi, ci}i∈{1,...,Mj} containing the amplitude,
    frequency and correlation components of the Mj peaks in the cluster.

    The components were clustered with kmeans.
    The number of clusters k was found using the elbow method which
    involves plotting the sum of squared distances between the data
    points and their cluster centroids (WCSS) against the number of
    clusters. The optimal number of clusters is the value of k at the
    "elbow" point, where the decrease in WCSS begins to level off.

    Parameters based on the two largest clusters formed in the set were used to estimate
    alpha rhythm frequency. In the event that only one cluster was available,
    we assumed that the recording contained one rhythm.
    Based on a given cluster, a weighted average of the frequency was calculated:

      wi = ci / Sum{M}{k} (ck)

      Qf (j) = Sum{M}{i} fi*wi


    with fi and wi presenting the estimated frequency and normalized
    correlation components of the Mj peaks found in a recording.

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

    combined_components = [[A1[i], A2[i], f1[i], f2[i], d1[i], d2[i], c1[i], c2[i]] for i in range(len(A1))]

    TH = 1.5
    combined_components = [x for x in combined_components if 1 <= x[1] <= 18 and x[3] <= TH]
    X = asarray(combined_components)

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

    return sum(largest_cluster_elements[:, 1] * w)
