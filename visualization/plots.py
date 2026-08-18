"""
Publication-quality visualisation utilities.
All functions accept an optional `ax` or `savefig` path so they can be
used both interactively and in batch scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal

from analysis.spectral import compute_psd, relative_band_power, EEG_BANDS


def _maybe_save(fig: plt.Figure, savepath: Optional[str | Path]) -> None:
    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Figure saved -> {savepath}")


def plot_raw_signal(
    data: np.ndarray,
    fs: float,
    channel: int = 0,
    title: str = "Raw EEG",
    duration: Optional[float] = None,
    savepath: Optional[str | Path] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a single channel time series.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples) or (n_samples,).
    fs : float
        Sampling rate.
    channel : int
        Which channel to display when data is 2-D.
    """
    if data.ndim == 2:
        sig = data[channel]
    else:
        sig = data

    if duration is not None:
        n = int(duration * fs)
        sig = sig[:n]

    t = np.arange(len(sig)) / fs
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.figure

    ax.plot(t, sig, color="#1f77b4", linewidth=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _maybe_save(fig, savepath)
    return ax


def plot_psd_comparison(
    signals: Dict[str, np.ndarray],
    fs: float,
    title: str = "Power Spectral Density by State",
    savepath: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Overlay mean PSDs of several states on a log-scale plot.

    Parameters
    ----------
    signals : dict
        Mapping state_name -> array of shape (n_channels, n_samples).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"awake": "#2ca02c", "sleep": "#1f77b4", "anesthesia": "#d62728"}

    for state, data in signals.items():
        freqs, psd = compute_psd(data, fs=fs)
        psd_mean = np.mean(psd, axis=0)
        ax.semilogy(
            freqs,
            psd_mean,
            label=state.capitalize(),
            color=colors.get(state, None),
            linewidth=1.8,
        )

    ax.set_xlim(0.5, 45)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _maybe_save(fig, savepath)
    return fig


def plot_band_power_bars(
    rel_powers: Dict[str, Dict[str, float]],
    title: str = "Relative Band Power by State",
    savepath: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart of relative power across states and bands.

    Parameters
    ----------
    rel_powers : dict
        Outer key = state, inner dict = band -> relative power.
    """
    states = list(rel_powers.keys())
    bands = list(EEG_BANDS.keys())
    x = np.arange(len(bands))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, state in enumerate(states):
        vals = [rel_powers[state].get(b, 0.0) for b in bands]
        ax.bar(x + i * width, vals, width, label=state.capitalize())

    ax.set_xticks(x + width)
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_ylabel("Relative power")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    _maybe_save(fig, savepath)
    return fig


def plot_spectrogram(
    data: np.ndarray,
    fs: float,
    channel: int = 0,
    title: str = "Spectrogram",
    savepath: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Compute and display a spectrogram for one channel.
    """
    if data.ndim == 2:
        sig = data[channel]
    else:
        sig = data

    f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=256, noverlap=192)
    fig, ax = plt.subplots(figsize=(10, 4))
    pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="viridis")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 45)
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    _maybe_save(fig, savepath)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
    savepath: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Heat-map style confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    _maybe_save(fig, savepath)
    return fig
