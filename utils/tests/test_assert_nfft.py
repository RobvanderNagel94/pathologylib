from utils.number_validation import _assert_nfft

import pytest


@pytest.mark.parametrize(
    "n, nfft, nperseg, noverlap, expected_exception",
    [
        # nperseg is None, nfft <= n, no overlap.
        (100, 100, None, 0, None),
        # nperseg is None, nfft > n, no overlap.
        (100, 200, None, 0, ValueError),
        # nperseg is None, nfft > n, overlap > 0.
        (100, 200, None, 50, ValueError),
        # nperseg < nfft, overlap < nperseg.
        (100, 200, 50, 25, None),
        # nperseg > nfft, overlap < nfft.
        (100, 50, 200, 25, None),
        # nperseg > nfft, overlap > nfft.
        (100, 50, 200, 100, ValueError),
    ],
)
def test_check_nfft(n, nfft, nperseg, noverlap, expected_exception):
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            _assert_nfft(n, nfft, nperseg, noverlap)

