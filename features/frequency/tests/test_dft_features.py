def test_dft_with_example():
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
    eeg_data.set_montage('source')

    # Define band limits
    from features.preprocessing import assemble_overlapping_band_limits
    band_limits = [[0, 2], [2, 4], [4, 8], [8, 13], [13, 18], [18, 24], [24, 30], [30, 49.9]]
    non_overlapping_bands = band_limits
    band_limits = assemble_overlapping_band_limits(non_overlapping_bands)

    # Split signals in 10-s epochs
    from features.preprocessing import split_into_epochs
    epoch_duration_s = 10
    sigs, chans = eeg_data.as_array()
    crops = split_into_epochs(signals=sigs, fs=fs, epoch_duration_s=epoch_duration_s)

    # Cap outliers at +-800 microVolt
    from features.preprocessing import replace_outliers
    outlier_threshold = replacement_threshold = 800.
    crops = replace_outliers(crops=crops, outlier_threshold=outlier_threshold,
                             replacement_threshold=replacement_threshold)

    # Apply window function
    from features.preprocessing import apply_window_function
    window_name = "blackman"
    weighted_crops = apply_window_function(epochs=crops, window_name=window_name)

    # Extract dft features
    from features.frequency.generator_frequency import generate_dft_features
    agg_mode = None
    dft_features = generate_dft_features(crops=weighted_crops, fs=fs, band_limits=band_limits, agg_func=agg_mode)
    print(dft_features.shape)
    print(dft_features)