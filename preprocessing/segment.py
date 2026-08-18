"""
Windowing / epoching utilities.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def segment_into_windows(
    data: np.ndarray,
    fs: float,
    window_sec: float = 2.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Split a continuous multi-channel recording into overlapping windows.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples).
    fs : float
        Sampling frequency (Hz).
    window_sec : float
        Length of each window in seconds.
    overlap : float
        Fractional overlap between consecutive windows (0 ≤ overlap < 1).

    Returns
    -------
    np.ndarray
        Array of shape (n_windows, n_channels, window_samples).
    """
    if data.ndim != 2:
        raise ValueError("Expected data of shape (n_channels, n_samples)")

    n_channels, n_samples = data.shape
    win_samples = int(window_sec * fs)
    step = int(win_samples * (1.0 - overlap))

    if win_samples > n_samples:
        raise ValueError(
            f"Window length ({win_samples}) exceeds signal length ({n_samples})"
        )

    starts = np.arange(0, n_samples - win_samples + 1, step)
    windows = np.stack(
        [data[:, s : s + win_samples] for s in starts],
        axis=0,
    )
    return windows


def windows_to_list(
    windows: np.ndarray,
) -> List[np.ndarray]:
    """
    Convert a 3-D window array into a Python list of 2-D arrays
    (useful for some sklearn pipelines that expect a list of samples).
    """
    return [windows[i] for i in range(windows.shape[0])]
