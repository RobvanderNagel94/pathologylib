from scipy.stats import kurtosis as _kurt
from scipy.stats import skew as _skew

from numpy import ndarray
import numpy as np

time_params = {
    "fs": 250,
    "Kmax": 3,
    "n": 4,
    "T": 1,
    "Tau": 4,
    "DE": 10,
    "W": None,
}


def _embed_seq(x: ndarray, Tau: int, DE: int) -> ndarray:
    """
    Builds a set of embedding sequences from given time series x with lag Tau
    and embedding dimension de.

    :param x: Array of shape (n,).
    :param Tau: Lag or delay when building embedding sequence.
    :param DE: Embedding dimension.

    :return: 2D array with the embedding sequences.

    Notes
    ------
    Let X = [x(1), x(2), ... , x(N)], then for each
    i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
    Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding
    sequence are placed in a matrix.

    """
    shape = (x.size - Tau * (DE - 1), DE)
    strides = (x.itemsize, Tau * x.itemsize)

    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def hjorth_activity(crops: ndarray, axis: int, **kwargs):
    """
    Computes Hjorth activity.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Hjorth activity.

    References
    ----------
    https://en.wikipedia.org/wiki/Hjorth_parameters
    """

    return np.var(crops, axis=axis)


def hjorth_mobility(crops: ndarray, axis: int, **kwargs):
    """
    Computes Hjorth mobility.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Hjorth mobility.

    References
    ----------
    https://en.wikipedia.org/wiki/Hjorth_parameters
    """

    diff = np.diff(crops, axis=axis)
    sigma0 = np.std(crops, axis=axis)
    sigma1 = np.std(diff, axis=axis)

    return np.divide(sigma1, sigma0)


def hjorth_complexity(crops: ndarray, axis: int, **kwargs):
    """
    Computes Hjorth complexity.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Hjorth complexity.

    References
    ----------
    https://en.wikipedia.org/wiki/Hjorth_parameters
    """

    def _hjorth_mobility(crops, axis, **kwargs):
        diff = np.diff(crops, axis=axis)
        sigma0 = np.std(crops, axis=axis)
        sigma1 = np.std(diff, axis=axis)
        return np.divide(sigma1, sigma0)

    diff1 = np.diff(crops, axis=axis)
    diff2 = np.diff(diff1, axis=axis)
    sigma0 = np.std(crops, axis=axis)
    sigma1 = np.std(diff1, axis=axis)
    sigma2 = np.std(diff2, axis=axis)
    return np.divide(np.divide(sigma2, sigma1), _hjorth_mobility(crops, axis))


def hurst_exponent(crops: ndarray, axis: int):
    """
    Compute the Hurst exponent of a time series.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Hurst exponent.

    Raises
    ------
    ValueError
        If the first 10 differences of the series are zero.

    Notes
    -----
    The Hurst exponent measures the long-term persistence of a time series.
    If the output H=0.5, the behavior of the time-series is similar to random walk.
    If H<0.5, the time-series cover less "distance" than a random walk, vice versa.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hurst_exponent
    """

    def hurst_1d(X):
        N = X.size
        T = np.arange(1, N + 1)
        Y = np.cumsum(X)
        Ave_T = Y / T

        S_T = np.zeros(N)
        R_T = np.zeros(N)
        for i in range(N):
            S_T[i] = np.std(X[:i + 1])
            X_T = Y - T * Ave_T[i]
            R_T[i] = np.ptp(X_T[:i + 1])

        # Check for indifferent measurements at the start of the time series
        for i in range(1, len(S_T)):
            if np.diff(S_T)[i - 1] != 0:
                break
        for j in range(1, len(R_T)):
            if np.diff(R_T)[j - 1] != 0:
                break

        k = max(i, j)
        if k >= 10:
            return 0

        R_S = R_T[k:] / S_T[k:]
        R_S = np.log(R_S)

        n = np.log(T)[k:]
        A = np.column_stack((n, np.ones(n.size)))
        [H, _] = np.linalg.lstsq(A, R_S, rcond=None)[0]
        return H

    return np.apply_along_axis(hurst_1d, axis, crops)


def higuchi_fractal_dimension(crops: ndarray, axis: int, **kwargs):
    """
    Computes the Fractal Dimension using Higuchi's method.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Higuchi's fractal dimension.

    References
    ----------
    .. https://en.wikipedia.org/wiki/Higuchi_dimension
    """

    def hfd_1d(X, Kmax):
        L, x = [], []
        N = len(X)
        for k in range(1, Kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)

            L.append(np.log(np.mean(Lk)))
            x.append([np.log(1. / k), 1])

        [alpha, _] = np.linalg.lstsq(x, L, rcond=None)[0]
        hfd = alpha
        return hfd

    Kmax = kwargs["Kmax"]

    return np.apply_along_axis(hfd_1d, axis, crops, Kmax)


def petrosian_fractal_dimension(crops: ndarray, axis: int, **kwargs):
    """
    Computes the Petrosian Fractal Dimension.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Petrosian fractal dimension.

    References
    ----------
    .. [1] Comparison of Fractal Dimension Algorithms for the Computation of EEG
           Biomarkers for Dementia, ResearchGate
           https://www.researchgate.net/publication/40902603_Comparison_of_Fractal_Dimension_Algorithms_for_the_Computation_of_EEG_Biomarkers_for_Dementia
    """

    def pfd_1d(X):
        delta = np.diff(X)
        N_delta = np.sum(np.diff(np.sign(delta)) != 0)
        n = len(X)
        return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))

    return np.apply_along_axis(pfd_1d, axis, crops)


def svd_entropy(crops: ndarray, axis: int, **kwargs):
    """
    Computes entropy of the singular values retrieved from a
    singular value decomposition from the original series.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Total entropy of all the singular values.

    References
    ----------
    .. https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """

    def svd_entropy_1d(X, Tau, DE, W):
        if W is None:
            Y = _embed_seq(X, Tau, DE)
            W = np.linalg.svd(Y, compute_uv=False)
            W /= sum(W)

        return -1 * sum(W * np.log(W))

    Tau = kwargs["Tau"]
    DE = kwargs["DE"]
    W = kwargs["W"]

    return np.apply_along_axis(svd_entropy_1d, axis, crops, Tau, DE, W)


def svd_fisher(crops: ndarray, axis: int, **kwargs):
    """
    Computes the Fisher information of the singular values retrieved from the original series.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Fisher information of the singular values.

    References
    ----------
    https://en.wikipedia.org/wiki/Fisher_information
    """

    def fisher_info_1d(X, Tau, DE):
        mat = _embed_seq(X, Tau, DE)
        W = np.linalg.svd(mat, compute_uv=False)
        W /= sum(W)

        return np.sum((W[1:] - W[:-1]) ** 2 / W[:-1])

    Tau = kwargs["Tau"]
    DE = kwargs["DE"]

    return np.apply_along_axis(fisher_info_1d, axis, crops, Tau, DE)


def largest_lyapunov_exponent(crops: ndarray, axis: int, **kwargs):
    """
    Computes the largest Lyapunov exponent using Rosenstein's method.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Largest Lyapunov Exponent.

    Notes
    ----------

    A n-dimensional trajectory is first reconstructed from the observed data by
    use of embedding delay of `tau`, using `embed_seq(x, tau, n)`.

    The algorithm searches for nearest neighbor of each point on the
    reconstructed trajectory; temporal separation of nearest neighbors must be
    greater than the mean period of the time series: the mean period can be
    estimated as the reciprocal of the mean frequency in power spectrum.

    Each pair of nearest neighbors is assumed to diverge exponentially at a
    rate given by the largest Lyapunov exponent. Now having a collection of
    neighbors, a least-squares fit to the average exponential divergence is
    calculated. The slope of this line gives an accurate estimate of the
    largest Lyapunov exponent.

    References
    ----------
    .. Rosenstein, Michael T., James J. Collins, and Carlo J. De Luca. "A
       practical method for calculating largest Lyapunov exponents from small data
       sets." Physica D: Nonlinear Phenomena 65.1 (1993): 117-134.
    """

    def LLE_1d(x, tau, n, T, fs):
        Em = _embed_seq(x, tau, n)
        M = len(Em)
        A = np.tile(Em, (len(Em), 1, 1))
        B = np.transpose(A, [1, 0, 2])

        square_dists = (A - B) ** 2
        D = np.sqrt(square_dists[:, :, :].sum(axis=2))

        band = np.tri(D.shape[0], k=T) - np.tri(D.shape[0], k=-T - 1)
        band[band == 1] = np.inf
        neighbors = (D + band).argmin(axis=0) 

        inc = np.tile(np.arange(M), (M, 1))
        row_inds = (np.tile(np.arange(M), (M, 1)).T + inc)
        col_inds = (np.tile(neighbors, (M, 1)) + inc.T)
        in_bounds = np.logical_and(row_inds <= M - 1, col_inds <= M - 1)

        row_inds[~in_bounds] = 0
        col_inds[~in_bounds] = 0

        neighbor_dists = np.ma.MaskedArray(D[row_inds, col_inds], ~in_bounds)
        J = (~neighbor_dists.mask).sum(axis=1)
        neighbor_dists[neighbor_dists == 0] = 1

        neighbor_dists.data[neighbor_dists.data == 0] = 1

        d_ij = np.sum(np.log(neighbor_dists.data), axis=1)
        mean_d = d_ij[J > 0] / J[J > 0]

        x = np.arange(len(mean_d))
        X = np.vstack((x, np.ones(len(mean_d)))).T
        [alpha, _] = np.linalg.lstsq(X, mean_d, rcond=None)[0]
        return fs * alpha

    Tau = kwargs["Tau"]
    n = kwargs["n"]
    T = kwargs["T"]
    fs = kwargs["fs"]

    return np.apply_along_axis(LLE_1d, axis, crops, Tau, n, T, fs)


def lumpiness(crops: ndarray, axis: int, **kwargs):
    """
    Computes the variance of the variances based on a division of the series in non-overlapping portions.
    The size of the portions if the frequency of the series.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Variance of the variances of the data.

    References
    ----------
    .. arXiv:2010.10742v2 [cs.LG] 29 Oct 2020
    """

    def lump_1d(X, fs):
        if fs == 1:
            width = 10
        else:
            width = fs

        nr = len(X)
        lo = np.arange(0, nr, width)
        up = lo + width
        nsegs = nr / width
        varx = [np.nanvar(X[lo[idx]:up[idx]], ddof=1) for idx in np.arange(int(nsegs))]

        if len(X) < 2 * width:
            lumpiness = 0
        else:
            lumpiness = np.nanvar(varx, ddof=1)
        return lumpiness

    fs = kwargs["fs"]

    return np.apply_along_axis(lump_1d, axis, crops, fs)


def stability(crops: ndarray, axis: int, **kwargs):
    """
    Calculates the variance of the means based on a division of the series in non-overlapping portions.
    The size of the portions is the frequency of the series, or 10 is the series has frequency 1.

    :param x (np.array): time series
    :param fs (int): frequency
    """

    def stability_1d(X, fs):

        if fs == 1:
            width = 10
        else:
            width = fs

        nr = len(X)
        lo = np.arange(0, nr, width)
        up = lo + width
        nsegs = nr / width
        meanx = [np.nanmean(X[lo[idx]:up[idx]]) for idx in np.arange(int(nsegs))]

        if len(X) < 2 * width:
            stability = 0
        else:
            stability = np.nanvar(meanx, ddof=1)

        return stability

    fs = kwargs["fs"]

    return np.apply_along_axis(stability_1d, axis, crops, fs)


def flat_spots(crops: ndarray, axis: int, **kwargs):
    """
    Computes the number of flat spots in the series, calculated by discretizing the series
    into 10 equal sized intervals and counting the maximum run length within any single interval.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Number of flat spots.

    References
    ----------
    .. arXiv:2010.10742v2 [cs.LG] 29 Oct 2020
    """

    def flatspots_1d(X, len):
        try:
            cut_x = np.histogram_bin_edges(X, bins=len)
            hist_x = np.histogram(X, bins=cut_x)[0]
            nofs = np.amax(hist_x)
            return nofs
        except:
            return 0

    len = np.asarray(crops).shape[0]

    return np.apply_along_axis(flatspots_1d, axis, crops, len)


def zero_crossing(crops: ndarray, axis: int, **kwargs):
    """
    Computes the number of zero crossings in the given epochs.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Number of zero crossings in the given epochs.
    """
    e = 0.01
    norm = crops - crops.mean()
    return np.apply_along_axis(lambda crops: np.sum((crops[:-5] <= e) & (crops[5:] > e)), axis, norm)


def zero_crossing_derivative(crops: ndarray, axis: int, **kwargs):
    """
    Computes the number of zero crossings in the derivative of the given epochs.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Number of zero crossings in the derivative of the given epochs.
    """
    e = 0.01
    diff = np.diff(crops)
    norm = diff - diff.mean()
    return np.apply_along_axis(lambda crops: np.sum(((crops[:-5] <= e) & (crops[5:] > e))), axis, norm)


def energy(crops: ndarray, axis: int, **kwargs):
    """
    Computes the energy of the given epochs.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Energy of the given epochs.
    """
    return np.mean(crops * crops, axis=axis)


def non_linear_energy(crops: ndarray, axis: int, **kwargs):
    """
    Computes the non-linear energy of the given epochs.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Non-linear energy of the given epochs.
    """
    return np.apply_along_axis(lambda crops: np.mean((np.square(crops[1:-1]) - crops[2:] * crops[:-2])), axis, crops)


def skewness(crops: ndarray, axis: int, **kwargs):
    """
    Compute the skewness of the input epochs.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Skewness of the epochs.
    """
    return _skew(crops, axis=axis, bias=False)


def kurtosis(crops: ndarray, axis: int, **kwargs):
    """
    Compute the kurtosis of the input epochs.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Kurtosis of the input `epochs` along the specified `axis`.
    """
    return _kurt(crops, axis=axis, bias=False)


def line_length(crops: ndarray, axis: int, **kwargs):
    """
    Compute the line length of the input epochs.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Line length of the input `epochs` along the specified `axis`.
    """
    return np.sum(np.abs(np.diff(crops)), axis=axis)


def maximum(crops: ndarray, axis: int, **kwargs):
    """
    Compute the maximum value of the input epochs.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Maximum of the input `epochs` along the specified `axis`.
    """
    return np.max(crops, axis=axis)


def minimum(crops: ndarray, axis: int, **kwargs):
    """
    Compute the minimum value of the input epochs.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Minimum of the input `epochs` along the specified `axis`.
    """
    return np.min(crops, axis=axis)


def mean(crops: ndarray, axis: int, **kwargs):
    """
    Calculates the mean of the input ndarray along the specified axis.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Mean of the input `epochs` along the specified `axis`.
    """
    return np.mean(crops, axis=axis)


def median(crops: ndarray, axis: int, **kwargs):
    """
    Calculates the median of the input ndarray along the specified axis.

    :param crops: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Median of the input `epochs` along the specified `axis`.
    """
    return np.median(crops, axis=axis)
