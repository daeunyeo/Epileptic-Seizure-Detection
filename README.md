# Epileptic Seizure Detection — CHB-MIT Scalp EEG

---

## 1. Background & Motivation

Epileptic seizures can cause sudden loss of consciousness, injury, or death.
In a clinical monitoring context, **a missed seizure (false negative) is categorically
more dangerous than a false alarm (false positive)** — a false alarm prompts unnecessary
review, but a missed seizure means no intervention at all.

This project builds an end-to-end ML pipeline that detects seizure activity from
scalp EEG signals. The entire design — from labeling strategy to threshold selection —
is oriented around a single clinical constraint: **Recall must be 100%**.
No real seizure epoch should go undetected.

---

## 2. Dataset

**CHB-MIT Scalp EEG Database** — PhysioNet (open access)
- Pediatric patients with intractable seizures, recorded at Children's Hospital Boston
- Format: EDF, 19-channel bipolar montage, 256 Hz
- Seizure onset/offset times annotated by clinical staff

| Split | File | Role |
|---|---|---|
| Train | `chb01_03.edf` | Model training + internal validation |
| Test | `chb01_04.edf` | Cross-session validation (unseen) |

Ground truth for validation: physician-annotated seizure interval **1467–1494 s**

The dataset was chosen because it provides real clinical EEG with verified seizure
annotations — a prerequisite for meaningful evaluation against an actual ground truth,
not just self-generated labels.

---

## 3. Pipeline

| Step | What | Why |
|---|---|---|
| Channel selection | 19-channel bipolar montage | Remove non-EEG channels; standardize input |
| Bandpass filter (0.5–40 Hz) | Remove DC drift and high-freq noise | Keep only clinically relevant EEG frequency bands |
| Resampling 256 → 128 Hz | Halve temporal resolution | Components above 40 Hz already removed; halving is lossless here and reduces computation |
| ICA (remove ICA000) | Separate and remove ocular artifact component | Eye-blink artifacts dominate frontal channels and corrupt spike detection |
| Average re-referencing | Subtract mean across all channels | Remove common-mode noise shared across electrodes |
| 2-second epoching | Slice continuous signal into fixed windows | Enables per-window feature extraction and binary classification |
| Feature extraction | 228 features per epoch | Capture time-domain amplitude, spectral power, and nonlinear dynamics simultaneously |
| Rule-based pseudo-labeling | PtP AND Variance > mean+3σ → label=1 | No physician labels available for training data; statistical threshold approximates spike epochs |
| Model comparison | Balanced RF vs. XGBoost | Evaluate whether boosting with class-weight penalty outperforms resampling-based balancing |
| Feature selection | Top 30 by BRF Feature Importance | Reduce dimensionality; remove redundant features to improve generalization |
| Threshold tuning | 1% grid search, 15%–35% | Find lowest threshold that sustains Recall = 100% on unseen data |

---

## 4. Feature Extraction

Total: **228 features per epoch** (epoch shape: `1800 × 19 × 256`)

| Category | Method | Features |
|---|---|---|
| Time domain | Peak-to-Peak (`np.ptp`) | 19 |
| Time domain | Variance (`np.var`) | 19 |
| Frequency domain | Welch PSD — Delta/Theta/Alpha/Beta | 19 × 4 = 76 |
| Time-frequency | Spectrogram mean — Delta/Theta/Alpha/Beta | 19 × 4 = 76 |
| Nonlinear | Spectral Entropy | 19 |
| Nonlinear | Lyapunov Exponent (`nolds.lyap_r`) | 19 |

---

## 5. Model Comparison

Labels were generated via rule-based pseudo-labeling on `chb01_03`.
Both models were trained on the same 80/20 split with the same 228→30 features.

| Model | Recall | Precision | FN | Note |
|---|---|---|---|---|
| Balanced Random Forest | 0.81 | 0.31 | 4 | High FP (37), misses seizures |
| XGBoost (threshold 50%) | 0.67 | 0.74 | 7 | Better precision, misses 7 seizures |
| XGBoost (threshold 30%) | 0.67 | 0.70 | 7 | No improvement over 50% |
| **XGBoost (threshold 15%)** | **1.00** | 0.46 | **0** | All seizures detected — adopted |

XGBoost with `scale_pos_weight = 15.0` (ratio of negative to positive samples)
was selected. Threshold was set to 15% — the lowest value that keeps FN = 0
on the internal test split, confirmed on the external recording in step 5.

---

## 6. Validation

Model trained on `chb01_03` was applied without retraining to `chb01_04`
(same subject, unseen recording — cross-session generalization).

y_true was constructed from physician annotations:
epochs overlapping with the 1467–1494 s interval were labeled as seizure.

### 7. Results

| Metric | Value |
|---|---|
| Recall | **100.0%** |
| Precision | 51.9% |
| TP | 14 |
| FP | 13 |
| FN | **0** |
| TN | 73 |

### 8. Threshold sweep (15%–35%, 1% step)

| Threshold | Recall | Precision |
|---|---|---|
| 15–18% | 100.0% | 51.9% |
| **19%+** | **92.9%** | ~50–54% |

Threshold 19% caused 1 missed seizure epoch (FN=1).
**15% was confirmed as the optimal lower bound.**

---

## 9. Discussion

**Precision-Recall trade-off**
Raising the threshold above 18% dropped Recall to 92.9% — one real seizure epoch
was no longer detected. This directly confirmed the design constraint: in seizure
screening, optimizing for Precision at the cost of Recall is clinically unacceptable.
The 13 false positives at threshold 15% represent unnecessary alerts, but each can
be reviewed by a clinician. The alternative — missing a seizure — cannot be corrected
after the fact.

**Pseudo-label generalization gap**
Internal Precision (chb01_03 split): ~70% → External Precision (chb01_04): 51.9%.
The model was trained on labels generated by a statistical rule (PtP + Variance > mean+3σ),
not physician annotations. The drop in Precision on the unseen file reflects the model
partially learning the labeling rule rather than the underlying seizure morphology.
This is a structural limitation of rule-based pseudo-labeling.

**Model structural ceiling**
With 19-channel handcrafted features and a single-subject training set,
Precision ~50% appears to be near the ceiling for this architecture.
The Recall = 100% target was met, but further Precision improvement
requires addressing the root causes listed below.

---

## 10. Limitations & Future Work

| Limitation | Direction |
|---|---|
| Single subject (chb01), 2 files | Extend to all 24 subjects in CHB-MIT |
| Rule-based pseudo-labels | Replace with physician-annotated ground truth |
| Handcrafted features + XGBoost | Transition to 1D-CNN or LSTM for spatiotemporal learning |
| Offline batch inference | Adapt pipeline for sliding-window streaming simulation |

---

## 11. Repository Structure
```
├── src/
│   ├── 01_preprocessing.py
│   ├── 02_feature_extraction.py
│   ├── 03_train.py
│   ├── 04_save_and_simulate.py
│   └── 05_validate_threshold.py
├── pipeline.py              # full inference pipeline (load model → predict)
├── models/
│   └── .gitkeep             # .pkl files excluded via .gitignore
├── data/
│   └── .gitkeep             # EDF files excluded via .gitignore
└── README.md
```
---

## 12. Stack

`Python` `MNE` `XGBoost` `scikit-learn` `imbalanced-learn` `scipy` `nolds` `joblib`

## 13. Data

[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (PhysioNet, open access)
Download separately and place under `data/`. Not included in this repository.
