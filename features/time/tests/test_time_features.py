def test_time_features_with_example():
    from numpy import random
    random.seed(2023)

    # Open EEG data sample
    import pickle
    with open('eeg_dummy.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    handle.close()

    # Initiate EEGDataSet and set montage
    from core.eeg_data_set import EEGDataSet
    fs, age, id, gender = 250, 51, "001", 'm'
    eeg_data = EEGDataSet.from_dict(data_dict, fs=fs, age=age, id=id, gender=gender)
    eeg_data.set_montage('1020')

    # Extract the signals and channels
    sigs, chans = eeg_data.as_array()

    # Segment signals and cap outliers at 800 microVolt
    from features.preprocessing import (split_into_epochs, replace_outliers)
    crops = split_into_epochs(signals=sigs, fs=fs, epoch_duration_s=5)
    crops = replace_outliers(crops=crops, outlier_threshold=800, replacement_threshold=800)

    # Extract time features
    from features.time.generator_time import generate_time_features
    import numpy as np
    agg_mode = np.median
    time_features = generate_time_features(crops=crops, agg_func=agg_mode, fs=fs)
    print('Time features: ', time_features)
    print('Features of shape: ', time_features.shape)
