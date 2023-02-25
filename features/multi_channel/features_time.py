from scipy.stats import kurtosis as _kurt
from scipy.stats import skew as _skew
import numpy as np

Kmax = 3
n = 4
T = 1
Tau = 4
DE = 10
W = None
sfreq = 250


def _embed_seq(x: np.ndarray, Tau: int, DE: int) -> np.ndarray:
    """
    Builds a set of embedding sequences from given time series x with lag Tau
    and embedding dimension de.

    :param x: 1D array of time series data.
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


def _hjorth_parameters(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes Hjorth parameters.

    :param epochs: Embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: activity, mobility, complexity.

    References
    ----------
    https://en.wikipedia.org/wiki/Hjorth_parameters
    """

    def _hjorth_mobility(epochs, axis, **kwargs):
        diff = np.diff(epochs, axis=axis)
        sigma0 = np.std(epochs, axis=axis)
        sigma1 = np.std(diff, axis=axis)
        return np.divide(sigma1, sigma0)

    activity = np.var(epochs, axis=axis)
    diff1 = np.diff(epochs, axis=axis)
    diff2 = np.diff(diff1, axis=axis)
    sigma0 = np.std(epochs, axis=axis)
    sigma1 = np.std(diff1, axis=axis)
    sigma2 = np.std(diff2, axis=axis)
    mobility = np.divide(sigma1, sigma0)
    complexity = np.divide(np.divide(sigma2, sigma1), _hjorth_mobility(epochs, axis))
    return activity, complexity, mobility


def hurst_exponent(epochs: np.ndarray, axis: int):
    """
    Compute the Hurst exponent of a time series.

    :param epochs: 2D embedding sequence.
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

        # Compute the standard deviation "St" and the range of
        # cumulative deviate series "Rt"
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
            raise ValueError("Cannot difference series for k = {}", k)

        R_S = R_T[k:] / S_T[k:]
        R_S = np.log(R_S)

        # Fit a straight line with y = ax + b,
        # where the slope "a" is the estimated Hurst exponent.
        n = np.log(T)[k:]
        A = np.column_stack((n, np.ones(n.size)))
        [H, _] = np.linalg.lstsq(A, R_S, rcond=None)[0]
        return H

    return np.apply_along_axis(hurst_1d, axis, epochs)


def higuchi_fractal_dimension(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes the Fractal Dimension using Higuchi's method.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Higuchi's fractal dimension.

    References
    ----------
    .. https://en.wikipedia.org/wiki/Higuchi_dimension
    """

    def hfd_1d(X, Kmax):
        L, x = [],[]
        N = len(X)
        for k in range(1, Kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                # create x_m^k reconstructed sequences
                for i in range(1, int(np.floor((N - m) / k))):
                    # for each produced sequence x_m^k
                    # calculate the average box length Lmk
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k

                # add the average box length of the sequences
                Lk.append(Lmk)

            # take the log of the average box length which behaves
            # proportionally to the fractal dimension D times
            # log of the reciprocal of k
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(1. / k), 1])

        # fit a straight line with y = ax + b,
        # where the slope "a" is the estimated exponent.
        [alpha, _] = np.linalg.lstsq(x, L, rcond=None)[0]
        hfd = alpha
        return hfd

    Kmax = kwargs["Kmax"]

    return np.apply_along_axis(hfd_1d, axis, epochs, Kmax)


def petrosian_fractal_dimension(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes the Petrosian Fractal Dimension.

    :param epochs: 2D embedding sequence.
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

    return np.apply_along_axis(pfd_1d, axis, epochs)


def svd_entropy(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes entropy of the singular values retrieved from a
    singular value decomposition from the original series.

    :param epochs: 2D embedding sequence.
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

    Tau = 4
    DE = 10
    W = None

    return np.apply_along_axis(svd_entropy_1d, axis, epochs, Tau, DE, W)


def svd_fisher(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes the Fisher information of the singular values retrieved from the original series.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Fisher information of the singular values.

    References
    ----------
    https://en.wikipedia.org/wiki/Fisher_information
    """

    def fisher_info_1d(a, tau, de):
        mat = _embed_seq(a, tau, de)
        W = np.linalg.svd(mat, compute_uv=False)
        W /= sum(W)

        return np.sum((W[1:] - W[:-1]) ** 2 / W[:-1])

    Tau = 4
    DE = 10

    return np.apply_along_axis(fisher_info_1d, axis, epochs, Tau, DE)


def largest_lyapunov_exponent(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes the largest Lyapunov exponent using Rosenstein's method.

    :param epochs: 2D embedding sequence.
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

        # square_dists[i,j,k] = (Em[i][k]-Em[j][k])^2
        square_dists = (A - B) ** 2

        # D[i,j] = ||Em[i]-Em[j]||_2
        D = np.sqrt(square_dists[:, :, :].sum(axis=2))

        # Exclude elements within T of the diagonal
        band = np.tri(D.shape[0], k=T) - np.tri(D.shape[0], k=-T - 1)
        band[band == 1] = np.inf
        neighbors = (D + band).argmin(axis=0)  # nearest neighbors more than T steps away

        # Locate nearest neighbors of each point on the trajectory
        # in_bounds[i,j] = (i+j <= M-1 and i+neighbors[j] <= M-1)
        inc = np.tile(np.arange(M), (M, 1))
        row_inds = (np.tile(np.arange(M), (M, 1)).T + inc)
        col_inds = (np.tile(neighbors, (M, 1)) + inc.T)
        in_bounds = np.logical_and(row_inds <= M - 1, col_inds <= M - 1)

        row_inds[~in_bounds] = 0
        col_inds[~in_bounds] = 0

        # Nearest neighbor, Xˆj , is found by searching for the point that minimizes
        # The distance to the particular reference point, Xj.
        # neighbor_dists[i,j] = ||Em[i+j]-Em[i+neighbors[j]]||_2
        neighbor_dists = np.ma.MaskedArray(D[row_inds, col_inds], ~in_bounds)
        J = (~neighbor_dists.mask).sum(axis=1)
        neighbor_dists[neighbor_dists == 0] = 1

        # Handle division by zero cases
        neighbor_dists.data[neighbor_dists.data == 0] = 1

        # Impose the additional constraint that nearest neighbors need to have a temporal separation
        # greater than the mean period of the time series.
        d_ij = np.sum(np.log(neighbor_dists.data), axis=1)
        mean_d = d_ij[J > 0] / J[J > 0]
        x = np.arange(len(mean_d))

        # Compute the mean of the set of parallel lines (for j = 1, 2, ..., N ),
        # each with slope roughly proportional to λ1.
        # The largest Lyapunov exponent is then
        # calculated using a least-squares fit to the average line
        X = np.vstack((x, np.ones(len(mean_d)))).T
        [alpha, _] = np.linalg.lstsq(X, mean_d, rcond=None)[0]
        return fs * alpha

    tau = 4
    n = 4
    T = 1
    fs = 250

    return np.apply_along_axis(LLE_1d, axis, epochs, tau, n, T, fs)


def lumpiness(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes the variance of the variances based on a division of the series in non-overlapping portions.
    The size of the portions if the frequency of the series.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Variance of the variances of the data.

    References
    ----------
    .. arXiv:2010.10742v2 [cs.LG] 29 Oct 2020
    """

    def lump_1d(x):
        freq = 250
        nr = len(x)
        lower = np.arange(0, nr, freq)
        upper = lower + freq
        nsegs = nr / freq
        varx = [np.nanvar(x[lower[idx]:upper[idx]], ddof=1) for idx in np.arange(int(nsegs))]
        if nr < 2 * freq:
            lump = np.array([0])
        else:
            lump = np.nanvar(varx, ddof=1)
        return lump

    return np.apply_along_axis(lump_1d, axis, epochs)


def flat_spots(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes the number of flat spots in the series, calculated by discretizing the series
    into 10 equal sized intervals and counting the maximum run length within any single interval.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Number of flat spots.

    References
    ----------
    .. arXiv:2010.10742v2 [cs.LG] 29 Oct 2020
    """

    def flatspots_1d(epochs):
        try:
            cut_x = np.histogram_bin_edges(epochs, bins=10)
            hist_x = np.histogram(epochs, bins=cut_x)[0]
            nofs = np.amax(hist_x)
            return nofs
        except:
            return np.zeros(len(epochs))

    return np.apply_along_axis(flatspots_1d, axis, epochs)


def zero_crossing(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes the number of zero crossings in the given epochs.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Number of zero crossings in the given epochs.
    """
    e = 0.01
    norm = epochs - epochs.mean()
    return np.apply_along_axis(lambda epoch: np.sum((epoch[:-5] <= e) & (epoch[5:] > e)), axis, norm)


def zero_crossing_derivative(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes the number of zero crossings in the derivative of the given epochs.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Number of zero crossings in the derivative of the given epochs.
    """
    e = 0.01
    diff = np.diff(epochs)
    norm = diff - diff.mean()
    return np.apply_along_axis(lambda epoch: np.sum(((epoch[:-5] <= e) & (epoch[5:] > e))), axis, norm)


def energy(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes the energy of the given epochs.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Energy of the given epochs.
    """
    return np.mean(epochs * epochs, axis=axis)


def non_linear_energy(epochs: np.ndarray, axis: int, **kwargs):
    """
    Computes the non-linear energy of the given epochs.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Non-linear energy of the given epochs.
    """
    return np.apply_along_axis(lambda epoch: np.mean((np.square(epoch[1:-1]) - epoch[2:] * epoch[:-2])), axis, epochs)


def skewness(epochs: np.ndarray, axis: int, ** kwargs):
    """
    Compute the skewness of the input epochs.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Skewness of the epochs.
    """
    return _skew(epochs, axis=axis, bias=False)


def kurtosis(epochs: np.ndarray, axis: int, **kwargs):
    """
    Compute the kurtosis of the input epochs.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Kurtosis of the input `epochs` along the specified `axis`.
    """
    return _kurt(epochs, axis=axis, bias=False)


def line_length(epochs: np.ndarray, axis: int, **kwargs):
    """
    Compute the line length of the input epochs.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Line length of the input `epochs` along the specified `axis`.
    """
    return np.sum(np.abs(np.diff(epochs)), axis=axis)


def maximum(epochs: np.ndarray, axis: int, **kwargs):
    """
    Compute the maximum value of the input epochs.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Maximum of the input `epochs` along the specified `axis`.
    """
    return np.max(epochs, axis=axis)


def minimum(epochs: np.ndarray, axis: int, **kwargs):
    """
    Compute the minimum value of the input epochs.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Minimum of the input `epochs` along the specified `axis`.
    """
    return np.min(epochs, axis=axis)


def mean(epochs: np.ndarray, axis: int, **kwargs):
    """
    Calculates the mean of the input ndarray along the specified axis.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Mean of the input `epochs` along the specified `axis`.
    """
    return np.mean(epochs, axis=axis)


def median(epochs: np.ndarray, axis: int, **kwargs):
    """
    Calculates the median of the input ndarray along the specified axis.

    :param epochs: 2D embedding sequence.
    :param axis: Axis along which the computation is performed.

    :return: Median of the input `epochs` along the specified `axis`.
    """
    return np.median(epochs, axis=axis)
