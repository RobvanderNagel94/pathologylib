def test_eeg_dataset_with_example():
    from numpy import random
    random.seed(2023)

    # Generate example data
    from core.constants import MONTAGE_1020
    age, gender, id, fs = 50, "m", "001", 250
    signal_length = 100000
    signals = random.rand(signal_length, len(MONTAGE_1020))
    annotations = random.choice(['eo', 'ec', 'rem'], size=signal_length)

    # Initiate EEGDataSet and set montage
    from core.eeg_data_set import EEGDataSet
    eeg_data = EEGDataSet(signals=signals,
                          channels=MONTAGE_1020,
                          annotations=annotations,
                          fs=fs,
                          age=age,
                          id=id,
                          gender=gender)

    eeg_data.set_montage('source')

    # Test getters
    eeg_dict = eeg_data.as_dict(annot=True)
    eeg_frame = eeg_data.as_frame(annot=True)
    sigs, chans = eeg_data.as_array()
    eeg_meta = eeg_data.get_meta()
    annots = eeg_data.get_annotations()
    sig_fp2 = eeg_data.get_channel(channel='fp2')
    eeg_dict = eeg_data.filter_annotations(keep=['ec'])

    # Test setters
    eeg_data.set_montage('reference')
    eeg_data.set_new_annotations(annot=random.choice(['eo', 'ec', 'rem'], size=signal_length))
    eeg_data.set_channel(channel="o2", signal=random.rand(signal_length).flatten())

    # Create instance from dict
    import pickle
    with open('../datasets/eeg_dummy.pickle', 'rb') as handle:
        eeg_dict = pickle.load(handle)
    eeg_data = EEGDataSet.from_dict(eeg_dict, fs=fs, age=age, id=id, gender=gender)
    eeg_dict = eeg_data.as_dict(annot=True)
