from core.constants import (FS, ANNOT, GENDER, AGE, ID)
from core.montages import reference_channels
from utils.array_validation import validate_input

from numpy import (ndarray, asarray, char, transpose, arange, where, pad)
from pandas import DataFrame
from typing import Union


class EEGDataSet:
    """
    A base class to represent an EEG dataset.

    :param signals: Array with shape (n_samples, n_channels).
    :param channels: Array of strings with length n_channels.
    :param annotations: Array of strings with length n_samples.
    :param fs: Sampling frequency of the EEG signals in Hz.
    :param age: Age of the subject whose EEG data is represented in the dataset.
    :param id: ID of the subject whose EEG data is represented in the dataset.
    :param gender: Gender of the subject whose EEG data is represented in the dataset.

    Notes
    ----------
    The input signals and channels must satisfy the output of the 1020 system with 19 channels:
    MONTAGE_1020 = [FP2, F8, T4, T6, O2, F4, C4, P4, FZ, CZ, PZ, F3, C3, P3, FP1, F7, T3, T5, O1]

    see 'pathologylib.core.constants' for the specification
    """

    @validate_input
    def __init__(self,
                 signals: Union[ndarray, list],
                 channels: Union[ndarray, list],
                 annotations: Union[ndarray, list],
                 fs: int,
                 age: int,
                 id: str,
                 gender: str
                 ):

        self.id = id
        self.age = age
        self.fs = fs
        self.gender = gender
        self.signals = asarray(signals)
        self.len = len(signals)
        self.channels = asarray(char.lower(channels))
        self.annotations = asarray(char.lower(annotations))
        self.data = dict(zip(self.channels, transpose(self.signals)))
        self.data_annot = {ANNOT: self.annotations}

        self.meta = {
            FS: self.fs,
            AGE: self.age,
            ID: self.id,
            GENDER: self.gender
        }
        
        self.copy_data = self.data.copy()

    @classmethod
    def from_dict(cls, data_dict: dict, fs: int, age: int, id: str, gender: str):
        """
        Create an EEGDataSet object from a dictionary of signal data.

        :param data_dict: Dictionary with keys being the channel names and values the arrays of signal values.
        :param fs: Sampling frequency of the EEG signals in Hz.
        :param age: Age of the subject whose EEG data is represented in the dataset.
        :param id: ID of the subject whose EEG data is represented in the dataset.
        :param gender: Gender of the subject whose EEG data is represented in the dataset.

        :return: EEGDataSet object.
        """

        channels = asarray(list(data_dict.keys()))
        signals = asarray(list(data_dict.values())).T
        annotations = asarray([""] * len(signals))

        return cls(signals=signals,
                   channels=channels,
                   annotations=annotations,
                   fs=fs,
                   age=age,
                   id=id,
                   gender=gender)

    def as_dict(self, annot: bool = False) -> dict:
        """
        Return a dictionary of the signal data and channels.

        :param annot: Check if annotations should also be returned.

        :return: Dictionary of channels (keys) and arrays of signal values (values).

        """
        if annot:
            return {**self.data, **self.data_annot}
        else:
            return self.data

    def as_frame(self, annot: bool = False) -> DataFrame:
        """
        Return a pandas dataframe of the signal data and channels.

        :param annot: Check if annotations should also be returned.

        :return: Pandas dataframe of channels (cols) and signal values (rows) with time as index.

        """
        df = DataFrame(self.as_dict(annot=annot))
        df.index = arange(0, self.len) / self.fs
        df.index.name = 'Time'
        return df

    def as_array(self) -> (ndarray, ndarray):
        """ Return numpy arrays of the signal data and channels."""
        return transpose(self.signals), self.channels

    def get_annotations(self) -> ndarray:
        """ Retrieve the annotation data for each time index."""
        return self.annotations

    def get_meta(self) -> dict:
        """ Retrieve the meta data."""
        return self.meta

    def get_channel(self, channel: str) -> ndarray:
        """
        Retrieve the signal data for a specific channel.

        :param channel: Channel from which the signal data needs to be returned.

        :return: Array of signal data for a specific channel.
        """
        if channel not in self.channels:
            raise ValueError(f"{channel} not found in channels")

        idx = where(self.channels == channel)[0][0]
        return self.signals[:, idx]

    def set_new_annotations(self, annot: Union[ndarray, list]) -> None:
        """
        Set new annotations for a given dataset.

        :param annot: Array of strings presenting the annotations.
        """

        annot = asarray(annot)

        if not annot.shape == (len(annot),):
            raise ValueError(f"input must have shape (n,) but (n,m) was given.")
        if len(annot) != self.len:
            raise ValueError(f"shape of annot must be the same as signal length.")

        self.annotations = char.lower(annot)

    def set_channel(self, channel: str, signal: Union[ndarray, list]) -> None:
        """
        Set new signal data for a specific channel.

        :param channel: Channel for which the signal data needs to be replaced.
        :param signal: Array of signal data that is being replaced.
        """

        data = asarray(signal)

        if channel not in self.channels:
            raise ValueError(f"Channel {channel} not found in dataset")
        if not data.shape == (len(data),):
            raise ValueError(f"input must have shape (n,) but (n,m) was given.")
        if len(data) != self.len:
            raise ValueError(f"shape of data must be the same as signal length.")

        idx = where(self.channels == channel)[0][0]
        self.signals[:, idx] = data

    def set_montage(self, montage: str = '1020') -> None:
        """
        Re-reference channels according to a specified montage type.

        :param montage: Type of montage to apply. Should be one of:
          'longitudinal_bipolar', 'small_laplacian', 'reference', 'source', '1020'

        Notes
        ----------
        The input signals and channels must satisfy the output of the 1020 system with 19 channels:
        MONTAGE_1020 = [FP2, F8, T4, T6, O2, F4, C4, P4, FZ, CZ, PZ, F3, C3, P3, FP1, F7, T3, T5, O1]

        see 'pathologylib.core.constants' for the exact specifications.
        """

        montages = ["longitudinal_bipolar", "small_laplacian", "reference", "source", "1020"]

        if montage not in montages:
            raise ValueError(
                f"{montage} is not supported."
                f" Supported montages are "
                f" {montages} "
            )

        self.data = reference_channels(eeg_dic=self.copy_data, type=montage)

    def filter_annotations(self, keep: Union[ndarray, list]) -> dict:
        """
        Set dataset that match with 'keep'.

        :param keep: Array of strings presenting the annotations to be kept.
        :return: Dictionary with filtered annotations and corresponding signals.
        """

        keep = asarray(keep)

        if not all(k in self.annotations for k in keep):
            raise ValueError(f"One or more elements in {keep} not found "
                             f"in annotations = {self.annotations}")

        data_dict = self.as_dict(annot=True)
        annotations = data_dict[ANNOT]
        indices_to_keep = [i for i, annot in enumerate(annotations) if annot in keep]

        filtered_dict = {}
        for key, value in data_dict.items():
            if key == ANNOT:
                filtered_dict[key] = annotations[indices_to_keep]
            else:
                filtered_dict[key] = value[indices_to_keep]

        max_len = max([len(arr) for arr in filtered_dict.values()])
        for key, value in filtered_dict.items():
            filtered_dict[key] = pad(value, (0, max_len - len(value)), mode='constant', constant_values=0)

        data_annot = {ANNOT: asarray(filtered_dict[ANNOT])}

        del filtered_dict[ANNOT]
        data = filtered_dict

        return {**data, **data_annot}
