# EEG-Based Classification of Brain States  
### Awake · Sleep · Anesthesia

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clean, reproducible research codebase for spectral analysis and classical machine-learning classification of three consciousness-related brain states from multi-channel EEG.

This repository accompanies the paper  
**«Методы глубокого машинного обучения для моделирования и анализа нейробиологических процессов различной сложности»**  
("Deep Machine Learning Methods for Modeling and Analysis of Neurobiological Processes of Varying Complexity")  
by E. N. Benderskaya, A. A. Alifanova, S. A. Batalova, V. D. Zhuk, and A. S. Kovalenko  
(*Neurocomputers* / Нейрокомпьютеры, 2026).  
Because the original clinical recordings cannot be shared, the pipeline is demonstrated on **physiologically plausible synthetic EEG** whose spectral profiles match the well-known signatures of wakefulness, NREM-like sleep and deep anesthesia.

---

## Scientific Context

The paper examines deep-learning and computational approaches to modeling neurobiological processes of varying complexity, including the detection of consciousness-related brain states. The phase-coherence clustering approach in [Krigsa/phase_coherence_kmeans](https://github.com/Krigsa/phase_coherence_kmeans) was one of the reference works informing this line of analysis.  
Key empirical observations that motivate the present code:

| State        | Dominant bands          | Relative power profile (approx.) |
|--------------|-------------------------|----------------------------------|
| **Awake**    | β / low-γ               | elevated high-frequency power    |
| **Sleep**    | δ / θ                   | strong low-frequency dominance   |
| **Anesthesia** | δ (very strong)       | higher bands strongly suppressed |

The synthetic generator reproduces these profiles so that downstream feature extractors and classifiers can be developed and validated under controlled conditions.

---

## Repository Structure

```
eeg-brain-state-classification/
├── data/
│   ├── raw/                  # synthetic_eeg.npy + metadata.csv
│   └── processed/            # (optional intermediate arrays)
├── preprocessing/
│   ├── filter.py             # band-pass, notch
│   ├── normalize.py          # z-score, min-max
│   └── segment.py            # overlapping windows
├── analysis/
│   ├── spectral.py           # Welch PSD, absolute / relative band power
│   └── features.py           # spectral feature vectors for ML
├── classification/
│   └── classifiers.py        # Random Forest + RBF-SVM, stratified CV
├── visualization/
│   └── plots.py              # signals, PSD, spectrograms, confusion matrices
├── notebooks/                # exploratory Jupyter notebooks
├── results/
│   ├── figures/              # PNG diagnostics
│   └── metrics/              # CV scores, feature tables
├── scripts/
│   ├── generate_synthetic_data.py
│   └── run_pipeline.py       # end-to-end demonstration
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate synthetic data (≈ 12 subjects × 3 states × 2 trials)
python scripts/generate_synthetic_data.py --out data/raw

# 3. Run the full analysis pipeline
python scripts/run_pipeline.py
```

After a successful run you will find:

- `results/metrics/cv_metrics.csv` – cross-validated accuracy & macro-F1  
- `results/figures/` – raw traces, PSD overlays, spectrograms, band-power bars, confusion matrices  

Typical accuracy on the synthetic data (5-fold stratified CV) exceeds **0.90** for both Random Forest and RBF-SVM, confirming that the engineered spectral features cleanly separate the three states.

---

## Methods in Brief

### 1. Synthetic Data Generation
Band-limited Gaussian noise is filtered into the classical EEG bands (δ, θ, α, β, γ).  
Each state is assigned a distinct relative-power vector; a weak common-mode component and slow drift add realistic inter-channel correlation and non-stationarity.

### 2. Pre-processing
- Zero-phase Butterworth band-pass (0.5–45 Hz)  
- Optional 50 Hz notch  
- Channel-wise z-score normalisation  
- 2-second windows with 50 % overlap  

### 3. Feature Extraction
For every window the following scalar features are computed (channel-averaged):

- Relative power in δ, θ, α, β, γ  
- Spectral centroid  
- Spectral entropy (normalised Shannon entropy of the PSD)

### 4. Classification
Two classical algorithms are evaluated under stratified 5-fold cross-validation:

| Model            | Key hyper-parameters                     |
|------------------|------------------------------------------|
| Random Forest    | 200 trees, max_depth=12, balanced weights |
| SVM (RBF kernel) | C=10, γ=“scale”, probability estimates   |

All features are standardised inside a `sklearn.pipeline.Pipeline` so that scaling is performed correctly within each CV fold.

---

## Extending the Pipeline

- **Real data** – replace the loader in `run_pipeline.py` with your own EDF/BIDS reader; the rest of the code expects arrays of shape `(n_channels, n_samples)`.  
- **Deep models** – the windowed tensors produced by `segment_into_windows` can be fed directly to a 1-D CNN or Transformer.  
- **Additional features** – add time-domain statistics, phase-locking values or entropy measures inside `analysis/features.py`.

---

## Citation

If you use this code or the associated paper, please cite:

> Benderskaya E.N., Alifanova A.A., Batalova S.A., Zhuk V.D., Kovalenko A.S. Методы глубокого машинного обучения для моделирования и анализа нейробиологических процессов различной сложности [Deep machine learning methods for modeling and analysis of neurobiological processes of varying complexity]. *Neurocomputers*, 2026.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Questions about the implementation can be directed to [Anastasia Alifanova](https://github.com/NancyD2017), co-author of the parent paper.
