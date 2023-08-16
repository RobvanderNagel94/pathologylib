def test_dwt_with_example():
    from numpy import random
    random.seed(2023)

    import pickle
    with open('eeg_dummy.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    handle.close()

    from core.eeg_data_set import EEGDataSet
    fs, age, id, gender = 250, 51, "001", 'm'
    eeg_data = EEGDataSet.from_dict(data_dict, fs=fs, age=age, id=id, gender=gender)
    eeg_data.set_montage('source')

    from features.preprocessing import assemble_overlapping_band_limits
    band_limits = [[0, 2], [2, 4], [4, 8], [8, 13], [13, 18], [18, 24], [24, 30], [30, 49.9]]
    non_overlapping_bands = band_limits
    band_limits = assemble_overlapping_band_limits(non_overlapping_bands)

    from features.preprocessing import split_into_epochs
    epoch_duration_s = 10
    sigs, chans = eeg_data.as_array()
    crops = split_into_epochs(signals=sigs, fs=fs, epoch_duration_s=epoch_duration_s)

    from features.preprocessing import replace_outliers
    outlier_threshold = replacement_threshold = 800.
    crops = replace_outliers(crops=crops, outlier_threshold=outlier_threshold,
                             replacement_threshold=replacement_threshold)

    from features.preprocessing import apply_window_function
    window_name = "blackman"
    weighted_crops = apply_window_function(epochs=crops, window_name=window_name)

    from features.frequency.generator_wavelet import generate_dwt_features
    agg_mode = None
    discrete_wavelet = "db4"
    dwt_features = generate_dwt_features(crops=weighted_crops, wavelet=discrete_wavelet,
                                         fs=fs, agg_func=agg_mode)
    print(dwt_features.shape)
    print(dwt_features)
