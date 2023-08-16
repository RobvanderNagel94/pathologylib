def test_higuchi_fractal_dimension_with_example():
    from numpy import (array, random, cumsum, floor, mean, linalg, log)
    random.seed(2023)

    N = 10000
    X = cumsum(random.randn(N))
    Kmax = 10

    # Estimate Higuchi's fractal dimension
    L, x = [], []
    N = len(X)
    for k in range(1, Kmax):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            # create x_m^k reconstructed sequences
            for i in range(1, int(floor((N - m) / k))):
                # for each produced sequence x_m^k
                # calculate the average box length Lmk
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / floor((N - m) / float(k)) / k

            # add the average box length of the sequences
            Lk.append(Lmk)

        # take the log of the average box length which behaves
        # proportionally to the fractal dimension D times
        # log of the reciprocal of k
        L.append(log(mean(Lk)))
        x.append([log(1. / k), 1])

    # fit a straight line with y = ax + b,
    [alpha, beta] = linalg.lstsq(x, L, rcond=None)[0]
    hfd = alpha

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False, figsize=(8, 6))
    ax1.set_title('Higuchi fractal dimension')
    ax1.plot(X, label='data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax2.plot(range(1, Kmax), L, 'o', label='Lk')
    ax2.plot(range(1, Kmax), alpha * log(1. / array(range(1, Kmax))) + beta, label=f'Fitted line '
                                                                                         f'with hfd = {hfd:.3f}')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Ln(L(k))')
    ax2.legend()
    plt.show()

