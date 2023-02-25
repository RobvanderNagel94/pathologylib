from core.constants import MONTAGE_1020

from numpy import ndarray


def validate_input(func):
    """ Decorator function to validate the input data of the signal dataset"""

    def wrapper(self, signals, channels, annotations, sfreq, age, id, gender):
        if not isinstance(signals, ndarray) or signals.ndim != 2:
            raise ValueError("Signals must be a 2D numpy array")
        if not isinstance(channels, ndarray) or not all(isinstance(c, str) for c in channels):
            raise ValueError("Channels must be a array of strings")
        if not set(channels) == set(MONTAGE_1020):
            raise ValueError("Channels must match the MONTAGE_1020")
        if not isinstance(sfreq, (int, float)) or sfreq <= 0:
            raise ValueError("Sampling frequency (sfreq) must be a positive number")
        if id is not None and not isinstance(id, str):
            raise ValueError("Subject ID must be a string")
        if gender is not None and gender not in ['m', 'f']:
            raise ValueError("Gender must be either 'm' or 'f'")
        if age is not None and not isinstance(age, int):
            raise ValueError("Age must be an int")
        if not isinstance(annotations, ndarray) or not all(isinstance(a, str) for a in annotations):
            raise ValueError("Annotations must be an array of strings")
        if len(channels) != signals.shape[1]:
            raise ValueError("Number of channels must match the second dimension of signals")
        if len(annotations) != signals.shape[0]:
            raise ValueError("Number of annotations must match the first dimension of signals")
        if signals.shape[0] < 2 * sfreq:
            raise ValueError("Length of signals must be at least twice the sfreq")

        return func(self, signals, channels, annotations, sfreq, age, id, gender)

    return wrapper
