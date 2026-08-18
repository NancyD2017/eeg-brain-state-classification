"""
Amplitude normalisation helpers for EEG.
"""

from __future__ import annotations

import numpy as np


def zscore_normalize(
    data: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Z-score normalise along the given axis (usually time).

    Parameters
    ----------
    data : np.ndarray
        Input array, typically (n_channels, n_samples).
    axis : int
        Axis along which to compute mean and std.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalised array of the same shape.
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / (std + eps)


def minmax_normalize(
    data: np.ndarray,
    axis: int = -1,
    feature_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    Scale data to a fixed range along the given axis.

    Parameters
    ----------
    data : np.ndarray
        Input array.
    axis : int
        Axis of normalisation.
    feature_range : tuple of float
        Desired (min, max) after scaling.

    Returns
    -------
    np.ndarray
        Scaled array.
    """
    data_min = np.min(data, axis=axis, keepdims=True)
    data_max = np.max(data, axis=axis, keepdims=True)
    scale = (feature_range[1] - feature_range[0]) / (data_max - data_min + 1e-12)
    return feature_range[0] + (data - data_min) * scale
