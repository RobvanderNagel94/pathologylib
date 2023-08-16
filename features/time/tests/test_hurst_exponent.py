def test_hurst_with_example():
    from numpy import (random, arange, cumsum, zeros, std, ptp, log, column_stack, linalg)
    random.seed(2023)

    X = random.randn(1000)
    N = X.size
    T = arange(1, N + 1)
    Y = cumsum(X)
    Ave_T = Y / T

    # Compute the standard deviation "St" and the range of
    # cumulative deviate series "Rt"
    S_T = zeros(N)
    R_T = zeros(N)
    for i in range(N):
        S_T[i] = std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = ptp(X_T[:i + 1])

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
    R_S = log(R_S)
    n = log(T)[k:]
    A = column_stack((n, np.ones(n.size)))

    # Fit a straight line with y = ax + b,
    [alpha, beta] = linalg.lstsq(A, R_S, rcond=None)[0]
    hurst = alpha

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))
    ax1.set_title('Hurst exponent')
    ax1.plot(X, label='data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax2.plot(R_S, 'b.')
    ax2.plot(alpha * n, 'r', label=f'fitted line with H = {hurst:.3f}')
    ax2.legend()
    ax2.set_xlabel('log(time)')
    ax2.set_ylabel('log(R/S)')
    plt.show()

