from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt


def STFT(x: np.ndarray,
         fs: float = 250,
         noverlap: int = 256,
         nfft: int = 512,
         nperseg: int = 512,
         file_name: str = 'img.png',
         show: bool = False
         ) -> None:
    """
    Produces a spectogram image using Short-Time Fourier Transform (STFT) method.

    :param x: One-dimensional series of signal values.
    :param fs: Sampling frequency of the series, default 250 Hz.
    :param noverlap: Number of points to overlap between segments, default 256.
        If None, noverlap = nperseg // 2
    :param nfft: Length of the FFT used, default 512.
    :param nperseg: Length of each segment, default 512.
    :param file_name: Figure name of the spectogram, supports .png and .jpg
    :param show: If True, shows the spectogram plot, by default False.

    Notes:
    ------
    Saves image to current working directory.

    Raises
    ------
    Exception
        if input length is less than nperseg

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(2*np.pi*50*np.arange(0, 5, 0.01))
    >>> STFT(x, file_name='spectogram_img.png')
    """

    error_msg = "Exception: input length = {} must be greater than nperseg = {}."
    assert len(x) >= nperseg, error_msg.format(len(x), nperseg)

    f, t, Pxx = spectrogram(x,
                            fs=fs,
                            nperseg=nperseg,
                            noverlap=noverlap,
                            nfft=nfft,
                            detrend='constant',
                            return_onesided=True,
                            scaling='density',
                            axis=-1,
                            mode='psd'
                            )

    plt.pcolormesh(t, f, Pxx, shading='gouraud')

    if show:
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (sec)')
        plt.show()
        plt.savefig(file_name)
    else:
        plt.savefig(file_name)
