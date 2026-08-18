"""
High-level feature extraction that turns raw windows into
tabular feature matrices suitable for classical ML classifiers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .spectral import compute_psd, relative_band_power, EEG_BANDS


def extract_spectral_features(
    window: np.ndarray,
    fs: float,
) -> Dict[str, float]:
    """
    Extract a compact spectral feature vector from a single
    multi-channel window.

    Features (averaged across channels):
        - relative power in delta, theta, alpha, beta, gamma
        - spectral centroid (mean frequency weighted by power)
        - spectral entropy (normalised Shannon entropy of the PSD)

    Parameters
    ----------
    window : np.ndarray
        Shape (n_channels, n_samples).
    fs : float
        Sampling rate.

    Returns
    -------
    dict
        Feature name → scalar value.
    """
    freqs, psd = compute_psd(window, fs=fs)
    # Average PSD across channels for a global spectrum
    psd_mean = np.mean(psd, axis=0)

    rel = relative_band_power(freqs, psd_mean[np.newaxis, :], EEG_BANDS)
    features = {f"rel_{b}": float(v[0]) for b, v in rel.items()}

    # Spectral centroid
    total_power = np.trapezoid(psd_mean, freqs) + 1e-20
    centroid = np.trapezoid(freqs * psd_mean, freqs) / total_power
    features["spectral_centroid"] = float(centroid)

    # Spectral entropy
    psd_norm = psd_mean / (np.sum(psd_mean) + 1e-20)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-20))
    # Normalise by log2 of number of bins
    features["spectral_entropy"] = float(entropy / np.log2(len(psd_norm)))

    return features


def extract_window_features(
    windows: np.ndarray,
    fs: float,
    labels: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Vectorise a batch of windows into a pandas DataFrame of features.

    Parameters
    ----------
    windows : np.ndarray
        Shape (n_windows, n_channels, n_samples).
    fs : float
        Sampling rate.
    labels : np.ndarray, optional
        State labels of shape (n_windows,). If provided they are
        added as a column ``label``.

    Returns
    -------
    pd.DataFrame
        One row per window, columns = feature names (+ optional label).
    """
    rows: List[Dict[str, float]] = []
    for i in range(windows.shape[0]):
        feat = extract_spectral_features(windows[i], fs=fs)
        if labels is not None:
            feat["label"] = labels[i]
        rows.append(feat)
    return pd.DataFrame(rows)
