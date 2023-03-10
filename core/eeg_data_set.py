from core.constants import (ANNOT, SFREQ, AGE, ID, GENDER)
from core.montages import reference_channels
from utils.validation import validate_input

from pandas import DataFrame
from numpy import (ndarray, char, transpose, arange, where)


class EEGDataSet:
    """
    A class to represent an EEG dataset.

    :param signals: EEG signals in a 2D numpy array with shape (n_samples, n_channels).
    :param channels: Channel names in a 1D numpy array of strings with length n_channels.
    :param annotations: Annotations in a 1D numpy array of strings with length n_samples.
    :param sfreq: Sampling frequency of the EEG signals in Hz.
    :param age: Age of the subject whose EEG data is represented in the dataset.
    :param id: ID of the subject whose EEG data is represented in the dataset.
    :param gender: Gender of the subject whose EEG data is represented in the dataset.

    Note:
    -------
    The input signals and channels must satisfy the output of the 1020 system with 19 channels:
    MONTAGE_1020 = [FP2, F8, T4, T6, O2, F4, C4, P4, FZ, CZ, PZ, F3, C3, P3, FP1, F7, T3, T5, O1]

    see 'pathologylib.base.constants' for the specification

    Examples:
    --------
    >>> import numpy as np
    >>> np.random.seed(2023)
    >>> MONTAGE_1020 = np.array(['fp2', 'f8', 't4', 't6', 'o2', 'f4', 'c4', 'p4', 'fz', 'cz',
    >>>                          'pz', 'f3', 'c3', 'p3', 'fp1', 'f7', 't3', 't5', 'o1'])
    >>> # Define the dataset
    >>> eeg_data = EEGDataSet(signals=np.random.rand(5000, len(MONTAGE_1020)),
    >>>                       channels=MONTAGE_1020,
    >>>                       annotations=np.random.choice(['eo', 'ec', 'rem'], size=5000),
    >>>                       sfreq=250,
    >>>                       age=25,
    >>>                       id='001',
    >>>                       gender='m')
    >>>
    >>> # Set the montage to ('LPM', 'SLM', 'REF', 'G19', '1020')
    >>> eeg_data.set_montage('G19')
    >>> # Convert EEGDataSet to dict with meta information
    >>> eeg_dict = eeg_data.as_dict(meta=True)
    >>> # Convert EEGDataSet to a pandas DataFrame with meta information
    >>> eeg_frame = eeg_data.as_frame(meta=True)
    """

    @validate_input
    def __init__(self,
                 signals: ndarray,
                 channels: ndarray,
                 annotations: ndarray,
                 sfreq: int,
                 age: int,
                 id: str,
                 gender: str):

        self.id = id
        self.age = age
        self.sfreq = sfreq
        self.gender = gender
        self.signals = signals
        self.len = len(self.signals)

        self.channels = char.lower(channels)
        self.annotations = char.lower(annotations)
        self.data = dict(zip(self.channels, transpose(self.signals)))
        self.meta = {

            ANNOT: self.annotations,
            SFREQ: [self.sfreq] * self.len,
            AGE: [self.age] * self.len,
            ID: [self.id] * self.len,
            GENDER: [self.gender] * self.len
        }

        self.copy_data = self.data.copy()

    def as_dict(self, meta=False) -> dict:
        """ Return a dictionary of the signal data."""
        if meta:
            return {**self.data, **self.meta}
        else:
            return self.data

    def as_frame(self, meta=False) -> DataFrame:
        """ Return a pandas dataframe of the signal data."""
        df = DataFrame(self.as_dict(meta=meta))
        df.index = arange(0, self.len) / self.sfreq
        df.index.name = 'Time'
        return df

    def get_channel(self, channel) -> ndarray:
        """ Retrieve the signal data for a specific channel."""
        if channel not in self.channels:
            raise ValueError(f"Channel {channel} not found in dataset")

        idx = where(self.channels == channel)[0][0]
        return self.signals[:, idx]

    def set_channel(self, channel, data) -> None:
        """ Set new signal data for a specific channel."""
        if channel not in self.channels:
            raise ValueError(f"Channel {channel} not found in dataset")

        idx = where(self.channels == channel)[0][0]
        self.signals[:, idx] = data

    def set_montage(self, montage: str = '1020') -> None:
        """ Re-reference 1020 montage to a specific montage. ('LPM', 'SLM', 'REF', 'G19', '1020')"""
        self.data = reference_channels(dic=self.copy_data, montage_type=montage)

