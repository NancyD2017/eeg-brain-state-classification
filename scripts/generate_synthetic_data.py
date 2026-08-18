#!/usr/bin/env python3
"""
Generate synthetic multi-channel EEG signals that approximate three
brain states relevant to consciousness research:

    - Awake (wakefulness): elevated beta / low-gamma power
    - Sleep (NREM-like): dominant delta / theta
    - Anesthesia (deep sedation): strong delta, suppressed higher bands

The generator produces realistic band-limited noise with state-specific
spectral profiles, mild non-stationarity, and inter-channel correlation.
Data are saved as NumPy arrays + a metadata CSV for easy downstream use.

This module is intentionally self-contained so that the rest of the
pipeline can be run without access to the original clinical recordings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_FS: float = 256.0          # sampling frequency (Hz)
DEFAULT_DURATION: float = 60.0     # seconds per recording
DEFAULT_N_CHANNELS: int = 8        # synthetic "electrodes"
DEFAULT_N_SUBJECTS: int = 12       # number of synthetic subjects
DEFAULT_SEED: int = 42

# Classical EEG frequency bands (Hz)
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Relative power profiles for each state (must sum ~1.0)
# Values are approximate and chosen to produce visually distinct spectra.
STATE_PROFILES: Dict[str, Dict[str, float]] = {
    "awake": {
        "delta": 0.15,
        "theta": 0.15,
        "alpha": 0.25,
        "beta":  0.30,
        "gamma": 0.15,
    },
    "sleep": {
        "delta": 0.55,
        "theta": 0.25,
        "alpha": 0.10,
        "beta":  0.07,
        "gamma": 0.03,
    },
    "anesthesia": {
        "delta": 0.70,
        "theta": 0.15,
        "alpha": 0.08,
        "beta":  0.05,
        "gamma": 0.02,
    },
}


def _band_limited_noise(
    n_samples: int,
    fs: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate approximately band-limited Gaussian noise via filtering."""
    white = rng.standard_normal(n_samples)
    # Design a 4th-order Butterworth band-pass
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfilt(sos, white)
    # Normalise to unit variance
    return filtered / (np.std(filtered) + 1e-12)


def generate_state_signal(
    duration: float,
    fs: float,
    n_channels: int,
    state: str,
    rng: np.random.Generator,
    amplitude: float = 50.0,
) -> np.ndarray:
    """
    Synthesise a multi-channel EEG segment for a given brain state.

    Parameters
    ----------
    duration : float
        Length of the segment in seconds.
    fs : float
        Sampling rate (Hz).
    n_channels : int
        Number of synthetic channels.
    state : str
        One of {'awake', 'sleep', 'anesthesia'}.
    rng : np.random.Generator
        Random number generator for reproducibility.
    amplitude : float
        Overall scale in microvolts (µV).

    Returns
    -------
    np.ndarray
        Array of shape (n_channels, n_samples).
    """
    if state not in STATE_PROFILES:
        raise ValueError(f"Unknown state '{state}'. Choose from {list(STATE_PROFILES)}")

    n_samples = int(duration * fs)
    profile = STATE_PROFILES[state]
    signal_out = np.zeros((n_channels, n_samples), dtype=np.float64)

    # Shared common-mode component (weak correlation across channels)
    common = np.zeros(n_samples)
    for band, (lo, hi) in BANDS.items():
        weight = profile[band]
        common += weight * _band_limited_noise(n_samples, fs, lo, hi, rng)

    for ch in range(n_channels):
        channel = np.zeros(n_samples)
        for band, (lo, hi) in BANDS.items():
            weight = profile[band]
            # Slight per-channel variation of the profile
            local_weight = weight * (1.0 + 0.15 * rng.standard_normal())
            local_weight = max(local_weight, 0.01)
            channel += local_weight * _band_limited_noise(n_samples, fs, lo, hi, rng)

        # Mix a fraction of the common signal
        channel = 0.7 * channel + 0.3 * common
        # Mild slow drift (non-stationarity)
        t = np.arange(n_samples) / fs
        drift = 5.0 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi))
        channel += drift
        signal_out[ch] = amplitude * channel / (np.std(channel) + 1e-12)

    return signal_out


def generate_dataset(
    n_subjects: int = DEFAULT_N_SUBJECTS,
    duration: float = DEFAULT_DURATION,
    fs: float = DEFAULT_FS,
    n_channels: int = DEFAULT_N_CHANNELS,
    seed: int = DEFAULT_SEED,
    out_dir: Path | None = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate a full synthetic multi-subject EEG dataset.

    Returns
    -------
    data : np.ndarray
        Shape (n_recordings, n_channels, n_samples)
    meta : pd.DataFrame
        Metadata with columns: subject_id, state, recording_id, fs, duration
    """
    rng = np.random.default_rng(seed)
    states = list(STATE_PROFILES.keys())
    recordings: List[np.ndarray] = []
    rows: List[Dict] = []

    rec_id = 0
    for subj in range(1, n_subjects + 1):
        for state in states:
            # Two short recordings per subject/state for more samples
            for trial in range(2):
                sig = generate_state_signal(
                    duration=duration,
                    fs=fs,
                    n_channels=n_channels,
                    state=state,
                    rng=rng,
                )
                recordings.append(sig)
                rows.append(
                    {
                        "recording_id": rec_id,
                        "subject_id": f"S{subj:02d}",
                        "state": state,
                        "trial": trial,
                        "fs": fs,
                        "duration_sec": duration,
                        "n_channels": n_channels,
                    }
                )
                rec_id += 1

    data = np.stack(recordings, axis=0)  # (N, C, T)
    meta = pd.DataFrame(rows)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "synthetic_eeg.npy", data)
        meta.to_csv(out_dir / "metadata.csv", index=False)
        print(f"Saved {data.shape[0]} recordings to {out_dir}")
        print(f"  data shape : {data.shape}  (recordings, channels, samples)")
        print(f"  metadata   : {out_dir / 'metadata.csv'}")

    return data, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EEG dataset")
    parser.add_argument("--out", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--subjects", type=int, default=DEFAULT_N_SUBJECTS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--channels", type=int, default=DEFAULT_N_CHANNELS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    generate_dataset(
        n_subjects=args.subjects,
        duration=args.duration,
        fs=args.fs,
        n_channels=args.channels,
        seed=args.seed,
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
