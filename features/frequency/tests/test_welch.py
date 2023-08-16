def test_welch_with_example():
    from numpy import (random, asarray, log10)

    random.seed(2023)

    import pickle
    with open('../datasets/eeg_dummy.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    handle.close()

    print('\n>> Initiate EEGDataSet and set montage ..')
    from core.eeg_data_set import EEGDataSet
    fs, age, id, gender = 250, 51, "001", 'm'
    noverlap, nperseg, nfft = 256, 512, 512
    fmin, fmax = 1, 18
    eeg_data = EEGDataSet.from_dict(data_dict, fs=fs, age=age, id=id, gender=gender)
    eeg_data.set_montage('source')
    sig = asarray(data_dict['o2'])
    print(sig.shape)

    print('\n>> Estimate PSD and plot log power spectrum ..')
    from features.frequency.welch import estimate_welch
    f, Sxx = estimate_welch(x=sig,
                            fs=fs,
                            window='hann',
                            noverlap=noverlap,
                            nfft=nfft,
                            nperseg=nperseg,
                            fmin=fmin,
                            fmax=fmax)

    import matplotlib.pyplot as plt
    plt.plot(f, log10(Sxx))
    plt.title('PSD Welch')
    plt.xlabel('Frequency')
    plt.ylabel('log(Sxx)')
    plt.show()
