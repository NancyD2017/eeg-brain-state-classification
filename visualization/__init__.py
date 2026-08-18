"""Plotting helpers for EEG signals, spectra and classification results."""

from .plots import (
    plot_raw_signal,
    plot_psd_comparison,
    plot_band_power_bars,
    plot_confusion_matrix,
    plot_spectrogram,
)

__all__ = [
    "plot_raw_signal",
    "plot_psd_comparison",
    "plot_band_power_bars",
    "plot_confusion_matrix",
    "plot_spectrogram",
]
