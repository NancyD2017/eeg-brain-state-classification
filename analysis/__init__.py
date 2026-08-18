"""Spectral analysis and feature extraction for EEG."""

from .spectral import compute_psd, band_power, relative_band_power
from .features import extract_spectral_features, extract_window_features

__all__ = [
    "compute_psd",
    "band_power",
    "relative_band_power",
    "extract_spectral_features",
    "extract_window_features",
]
