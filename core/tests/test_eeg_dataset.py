import pytest
import numpy as np
import pandas as pd
import random
from core.constants import MONTAGE_1020
from core.eeg_data_set import EEGDataSet

random.seed(2023)

@pytest.mark.parametrize("signals, channels, annotations, fs, age, gender, id, expected_exception", [
    (np.array([1, 2, 3]), MONTAGE_1020, [], 0, 0, 0, "id1", None),
    (np.array([1, 2, 3]), MONTAGE_1020, [], 0, 0, 1, "id2", None),
    (np.array([1, 2, 3]), MONTAGE_1020, [], 0, 0, 0, "id3", None),
    (None, MONTAGE_1020, [], 0, 0, 0, "id4", ValueError),
    ([], MONTAGE_1020, [], 0, 0, 0, "id5", ValueError),
    (0, MONTAGE_1020, [], 0, 0, 0, "id6", ValueError),
    ("signal", MONTAGE_1020, [], 0, 0, 0, "id7", ValueError),
])
def test_eeg_dataset_with_example(signals, channels, annotations, fs, age, gender, id, expected_exception):
    if not isinstance(signals, np.ndarray) or not isinstance(channels, np.ndarray) or not isinstance(annotations, np.ndarray):
        expected_exception = ValueError
    if not all(isinstance(val, int) for val in [fs, age, gender]):
        expected_exception = ValueError
    
    if expected_exception:
        with pytest.raises(expected_exception):
            eeg_data = EEGDataSet(signals=signals,
                                  channels=channels,
                                  annotations=annotations,
                                  fs=fs,
                                  age=age,
                                  id=id,
                                  gender=gender)
    else:
        eeg_data = EEGDataSet(signals=signals,
                              channels=channels,
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
        eeg_data.set_new_annotations(annot=random.choice(['eo', 'ec', 'rem'], size=len(signals)))
        eeg_data.set_channel(channel="o2", signal=random.rand(len(signals)).flatten())

        # Create instance from dict
        import pickle
        with open('../datasets/eeg_dummy.pickle', 'rb') as handle:
            eeg_dict = pickle.load(handle)
        eeg_data = EEGDataSet.from_dict(eeg_dict, fs=fs, age=age, id=id, gender=gender)
        eeg_dict = eeg_data.as_dict(annot=True)
