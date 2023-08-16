import numpy as np
import matplotlib.pyplot as plt

def test_LLE_with_example():
    from numpy import (random, pi, log, sum,
        mean, diff, sin, tile,
        transpose, linspace, sqrt,
        tri, ones, linalg, inf,
        arange, logical_and, ma)
    random.seed(2023)

    # Generate a noisy sine wave
    t = linspace(0, 10 * pi, 1000)
    x = sin(t) + 0.1 * random.randn(len(t))
    tau = 20
    n = 5
    T = 5
    fs = 1 / mean(diff(t))

    from features.time.features_time import _embed_seq
    Em = _embed_seq(x, tau, n)
    M = len(Em)
    A = tile(Em, (len(Em), 1, 1))
    B = transpose(A, [1, 0, 2])

    # square_dists[i,j,k] = (Em[i][k]-Em[j][k])^2
    square_dists = (A - B) ** 2

    # D[i,j] = ||Em[i]-Em[j]||_2
    D = sqrt(square_dists[:, :, :].sum(axis=2))

    # Exclude elements within T of the diagonal
    band = tri(D.shape[0], k=T) - tri(D.shape[0], k=-T - 1)
    band[band == 1] = inf
    neighbors = (D + band).argmin(axis=0)  # nearest neighbors more than T steps away

    # Locate nearest neighbors of each point on the trajectory
    # in_bounds[i,j] = (i+j <= M-1 and i+neighbors[j] <= M-1)
    inc = tile(arange(M), (M, 1))
    row_inds = (tile(arange(M), (M, 1)).T + inc)
    col_inds = (tile(neighbors, (M, 1)) + inc.T)
    in_bounds = logical_and(row_inds <= M - 1, col_inds <= M - 1)

    row_inds[~in_bounds] = 0
    col_inds[~in_bounds] = 0

    # Nearest neighbor, Xˆj , is found by searching for the point that minimizes
    # The distance to the particular reference point, Xj.
    # neighbor_dists[i,j] = ||Em[i+j]-Em[i+neighbors[j]]||_2
    neighbor_dists = ma.MaskedArray(D[row_inds, col_inds], ~in_bounds)
    J = (~neighbor_dists.mask).sum(axis=1)
    neighbor_dists[neighbor_dists == 0] = 1

    # Handle division by zero cases
    neighbor_dists.data[neighbor_dists.data == 0] = 1

    # Impose the additional constraint that nearest neighbors need to have a temporal separation
    # greater than the mean period of the time series.
    d_ij = sum(log(neighbor_dists.data), axis=1)
    mean_d = d_ij[J > 0] / J[J > 0]
    X = arange(len(mean_d))

    # Compute the mean of the set of parallel lines (for j = 1, 2, ..., N ),
    # each with slope roughly proportional to λ1.
    # The largest Lyapunov exponent is then
    # calculated using a least-squares fit to the average line
    XX = np.vstack((X, ones(len(mean_d)))).T
    [alpha, beta] = linalg.lstsq(XX, mean_d, rcond=None)[0]

    # largest Lyapunov exponent
    lle = fs * alpha

    # Plot the results
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False, figsize=(8, 6))
    ax1.set_title('Mean Lyapunov exponent vs. Iteration')
    ax1.plot(x, label='data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax2.plot(mean_d, 'o', label='Data')
    ax2.plot(alpha * X + beta, 'r', label=f'Fitted line with lle = {lle:.3f}')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Lyapunov exponent')
    ax2.legend()
    plt.show()




