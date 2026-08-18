"""
Digital filtering utilities for EEG signals.

All functions accept arrays of shape (n_channels, n_samples) or
(n_samples,) and return arrays of the same shape.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal. Shape (n_channels, n_samples) or (n_samples,).
    fs : float
        Sampling frequency in Hz.
    low : float
        Lower cutoff frequency (Hz).
    high : float
        Upper cutoff frequency (Hz).
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered signal of the same shape as `data`.
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=-1)

    return filtered.squeeze() if squeeze else filtered


def notch_filter(
    data: np.ndarray,
    fs: float,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """
    Apply a notch (band-stop) filter to remove line noise.

    Parameters
    ----------
    data : np.ndarray
        Input signal. Shape (n_channels, n_samples) or (n_samples,).
    fs : float
        Sampling frequency in Hz.
    freq : float
        Centre frequency of the notch (Hz). Use 50 or 60 depending on region.
    quality : float
        Quality factor of the notch filter.

    Returns
    -------
    np.ndarray
        Filtered signal of the same shape as `data`.
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    b, a = signal.iirnotch(freq, quality, fs=fs)
    filtered = signal.filtfilt(b, a, data, axis=-1)

    return filtered.squeeze() if squeeze else filtered


def apply_standard_pipeline(
    data: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 45.0,
    notch_freq: float | None = 50.0,
) -> np.ndarray:
    """
    Convenience wrapper: notch (optional) + band-pass.

    Parameters
    ----------
    data : np.ndarray
        Raw EEG, shape (n_channels, n_samples).
    fs : float
        Sampling rate.
    low, high : float
        Band-pass cut-offs.
    notch_freq : float or None
        If given, a notch filter is applied first.

    Returns
    -------
    np.ndarray
        Pre-filtered signal.
    """
    out = data.copy()
    if notch_freq is not None:
        out = notch_filter(out, fs=fs, freq=notch_freq)
    out = bandpass_filter(out, fs=fs, low=low, high=high)
    return out
