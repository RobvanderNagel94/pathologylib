def test_stft_with_example():
    from numpy import random
    random.seed(2023)

    print('\n>> Load the data ..')

    import pickle
    with open('../datasets/eeg_dummy.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    handle.close()

    print('\n>> Initiate EEGDataSet and set montage ..')
    from core.eeg_data_set import EEGDataSet
    from numpy import asarray
    fs, age, id, gender = 250, 51, "001", 'm'
    noverlap, nperseg = 256, 512
    file_name = 'spectogram.png'
    eeg_data = EEGDataSet.from_dict(data_dict, fs=fs, age=age, id=id, gender=gender)
    eeg_data.set_montage('source')
    sig = asarray(data_dict['o2'])
    print(sig.shape)

    print('\n>> Create spectogram using short-time fourier transform ..')
    from features.frequency.spectogram import STFT
    STFT(x=sig,
         fs=fs,
         noverlap=noverlap,
         nperseg=nperseg,
         file_name=file_name,
         show=True)
