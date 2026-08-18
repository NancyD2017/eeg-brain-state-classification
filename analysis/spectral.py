"""
Power spectral density and band-power computation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import signal


# Standard EEG bands used throughout the project
EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def compute_psd(
    data: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate power spectral density using Welch's method.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples) or (n_samples,).
    fs : float
        Sampling rate.
    nperseg : int, optional
        Length of each segment for Welch. Defaults to min(256, n_samples).
    noverlap : int, optional
        Number of overlapping points. Defaults to nperseg // 2.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins (Hz).
    psd : np.ndarray
        Power spectral density. Shape (n_channels, n_freqs) or (n_freqs,).
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    n_samples = data.shape[-1]
    if nperseg is None:
        nperseg = min(256, n_samples)
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = signal.welch(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
        scaling="density",
    )
    return freqs, psd.squeeze() if squeeze else psd


def band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: Tuple[float, float],
) -> np.ndarray:
    """
    Integrate PSD over a frequency band (absolute power).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency vector from `compute_psd`.
    psd : np.ndarray
        PSD values, shape (..., n_freqs).
    band : tuple of float
        (low, high) frequency limits in Hz.

    Returns
    -------
    np.ndarray
        Absolute band power, shape (...).
    """
    idx = np.logical_and(freqs >= band[0], freqs < band[1])
    # Trapezoidal integration
    return np.trapezoid(psd[..., idx], freqs[idx], axis=-1)


def relative_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Dict[str, Tuple[float, float]] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Compute relative power for a set of classical EEG bands.

    Relative power of band b = power(b) / total_power (0.5–45 Hz).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency vector.
    psd : np.ndarray
        PSD array.
    bands : dict, optional
        Custom band definition. Defaults to EEG_BANDS.

    Returns
    -------
    dict
        Mapping band_name → relative power array.
    """
    if bands is None:
        bands = EEG_BANDS

    total = band_power(freqs, psd, (0.5, 45.0))
    # Avoid division by zero
    total = np.maximum(total, 1e-20)

    rel = {}
    for name, lims in bands.items():
        abs_p = band_power(freqs, psd, lims)
        rel[name] = abs_p / total
    return rel
