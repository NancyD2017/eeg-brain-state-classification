#!/usr/bin/env python3
"""
End-to-end demonstration pipeline:

1. Load (or generate) synthetic EEG
2. Pre-process (filter -> normalise -> window)
3. Extract spectral features
4. Train / evaluate classical classifiers
5. Produce diagnostic plots under results/figures/

Run from the repository root:

    python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis.features import extract_window_features
from analysis.spectral import compute_psd, relative_band_power
from classification.classifiers import train_and_evaluate
from preprocessing.filter import apply_standard_pipeline
from preprocessing.normalize import zscore_normalize
from preprocessing.segment import segment_into_windows
from scripts.generate_synthetic_data import generate_dataset
from visualization.plots import (
    plot_band_power_bars,
    plot_confusion_matrix,
    plot_psd_comparison,
    plot_raw_signal,
    plot_spectrogram,
)


def main() -> None:
    data_dir = ROOT / "data" / "raw"
    fig_dir = ROOT / "results" / "figures"
    metrics_dir = ROOT / "results" / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    npy_path = data_dir / "synthetic_eeg.npy"
    meta_path = data_dir / "metadata.csv"

    if npy_path.exists() and meta_path.exists():
        print("Loading existing synthetic dataset …")
        data = np.load(npy_path)
        meta = pd.read_csv(meta_path)
    else:
        print("Generating synthetic EEG dataset …")
        data, meta = generate_dataset(out_dir=data_dir)

    fs = float(meta["fs"].iloc[0])
    print(f"Dataset shape : {data.shape}  (recordings x channels x samples)")
    print(f"Sampling rate : {fs} Hz")

    # ------------------------------------------------------------------
    # 2. Pre-processing & windowing
    # ------------------------------------------------------------------
    print("\nPre-processing and windowing …")
    all_windows = []
    all_labels = []

    for i, row in meta.iterrows():
        raw = data[i]  # (C, T)
        filtered = apply_standard_pipeline(raw, fs=fs, low=0.5, high=45.0)
        normalised = zscore_normalize(filtered)
        windows = segment_into_windows(
            normalised, fs=fs, window_sec=2.0, overlap=0.5
        )
        labels = np.full(windows.shape[0], row["state"])
        all_windows.append(windows)
        all_labels.append(labels)

    X_win = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"Total windows : {X_win.shape[0]}")

    # ------------------------------------------------------------------
    # 3. Feature extraction
    # ------------------------------------------------------------------
    print("Extracting spectral features …")
    feat_df = extract_window_features(X_win, fs=fs, labels=y)
    feature_cols = [c for c in feat_df.columns if c != "label"]
    X = feat_df[feature_cols]
    y = feat_df["label"].values

    feat_df.to_csv(metrics_dir / "features.csv", index=False)
    print(f"Feature matrix : {X.shape}")

    # ------------------------------------------------------------------
    # 4. Classification
    # ------------------------------------------------------------------
    print("\nRunning cross-validated classification …")
    results = train_and_evaluate(X, y, n_splits=5)

    # Persist metrics
    summary = {
        name: {"accuracy": r["accuracy"], "f1_macro": r["f1_macro"]}
        for name, r in results.items()
    }
    pd.DataFrame(summary).T.to_csv(metrics_dir / "cv_metrics.csv")
    print(f"\nMetrics saved -> {metrics_dir / 'cv_metrics.csv'}")

    # ------------------------------------------------------------------
    # 5. Visualisations
    # ------------------------------------------------------------------
    print("\nGenerating figures …")

    # Pick one representative recording per state for qualitative plots
    example_signals = {}
    for state in ["awake", "sleep", "anesthesia"]:
        idx = meta.index[meta["state"] == state][0]
        example_signals[state] = apply_standard_pipeline(data[idx], fs=fs)

    # Raw traces
    for state, sig in example_signals.items():
        plot_raw_signal(
            sig,
            fs=fs,
            channel=0,
            title=f"Example {state.capitalize()} EEG (channel 0)",
            duration=10.0,
            savepath=fig_dir / f"raw_{state}.png",
        )

    # PSD overlay
    plot_psd_comparison(
        example_signals,
        fs=fs,
        savepath=fig_dir / "psd_comparison.png",
    )

    # Spectrogram of one anesthesia recording
    plot_spectrogram(
        example_signals["anesthesia"],
        fs=fs,
        channel=0,
        title="Spectrogram – Anesthesia",
        savepath=fig_dir / "spectrogram_anesthesia.png",
    )

    # Relative band-power bars (computed on the same examples)
    rel_powers = {}
    for state, sig in example_signals.items():
        freqs, psd = compute_psd(sig, fs=fs)
        psd_mean = np.mean(psd, axis=0)
        rel = relative_band_power(freqs, psd_mean[np.newaxis, :])
        rel_powers[state] = {b: float(v[0]) for b, v in rel.items()}

    plot_band_power_bars(
        rel_powers,
        savepath=fig_dir / "band_power_bars.png",
    )

    # Confusion matrices
    class_names = sorted(np.unique(y))
    for name, res in results.items():
        plot_confusion_matrix(
            res["confusion_matrix"],
            class_names=class_names,
            title=f"Confusion Matrix – {name}",
            savepath=fig_dir / f"cm_{name.lower()}.png",
        )

    print("\nPipeline finished successfully.")
    print(f"Figures -> {fig_dir}")
    print(f"Metrics -> {metrics_dir}")


if __name__ == "__main__":
    main()
