# Epileptic Seizure Detection — CHB-MIT Scalp EEG

---

## 1. Background & Motivation

In clinical seizure monitoring, **missing a real seizure (false negative) is far more 
dangerous than a false alarm** — an undetected seizure means no intervention, 
which can result in serious harm to the patient.

The goal of this project is to build an ML pipeline that automatically detects 
epileptic seizure activity from scalp EEG signals in 2-second windows, 
prioritizing Recall above all other metrics so that no real seizure goes undetected.

To ensure the pipeline was tested against clinically meaningful ground truth, 
the CHB-MIT Scalp EEG Database — real EEG recordings from epileptic patients 
annotated by physicians — was used as the data source.
The model trained on one recording (`chb01_03`) was validated on a separate 
unseen recording (`chb01_04`), achieving **Recall 100% (FN = 0)** on 
physician-annotated seizure intervals.

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
| Rule-based pseudo-labeling | PtP AND Variance > mean+3σ → label=1 | No physician labels available **for training data**; statistical threshold approximates spike epochs |
| Model comparison | Balanced RF vs. XGBoost | Evaluate whether boosting with class-weight penalty outperforms resampling-based balancing |
| Feature selection | Top 30 by BRF Feature Importance | Reduce dimensionality; remove redundant features to improve generalization |
| Threshold tuning | 1% grid search, 15%–35% | 15% showed lowest FN on internal split; Recall = 100% confirmed on unseen data (chb01_04) |

<img width="651" height="491" alt="9 ica000제거 overlay" src="https://github.com/user-attachments/assets/66ca30de-c8e3-4f95-ade5-d943875a17cb" />
*Before/after ICA removal of ICA000 (ocular artifact component) — red: original, black: cleaned*



<img width="772" height="793" alt="9 ptp,v threshold_스파이크의심구간" src="https://github.com/user-attachments/assets/adb268fb-26ef-452b-a861-8b2592783f5b" />
*First epoch flagged by dual-threshold pseudo-labeling (PtP AND Variance > mean+3σ) — epoch 17, 34–36 s*

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

<img width="988" height="790" alt="epi feature importance" src="https://github.com/user-attachments/assets/05396b0d-7691-45c9-a1bf-e4e0d2dfc926" />

*Top 20 feature importances from Balanced Random Forest — used to select final 30 features for XGBoost*

---

## 5. Model Comparison

Labels were generated via rule-based pseudo-labeling on `chb01_03`.
Both models were trained on the same 80/20 split with the same 228→30 features.

| Model | Recall | Precision | FN | Note |
|---|---|---|---|---|
| Balanced Random Forest | 0.81 | 0.31 | 4 | FP=38 — alarm fatigue risk; XGBoost confirmed FN=0 on external validation |
| XGBoost (threshold 50%)   | 0.62 | 0.76 | 8  | Misses 8 seizures |
| XGBoost (threshold 30%)   | 0.67 | 0.74 | 7  | Misses 7 seizures |
| **XGBoost (threshold 15%)**   | **0.71** | 0.68 | **6**  | Lowest FN on internal split — selected for external validation |

XGBoost with scale_pos_weight = 15.0 was selected.
Threshold 15% produced the lowest FN (6) and highest Recall (0.71)
on the internal test split — selected as candidate and confirmed
on unseen data (chb01_04) in the Validation section.

---

## 6. Validation

Model trained on `chb01_03` was applied without retraining to `chb01_04`
(same subject, unseen recording — cross-session generalization).
Evaluation window: 1360–1560 s (200 s, 100 epochs — 14 seizure, 86 normal).

y_true was constructed from physician annotations:
epochs overlapping with the 1467–1494 s interval were labeled as seizure.
Evaluation window: 1360–1560 s (200 s, 100 epochs total — 14 seizure, 86 normal).

## 7. Results

| Metric | Value |
|---|---|
| Recall | **100.0%** |
| Precision | 51.9% |
| TP | 14 |
| FP | 13 |
| FN | **0** |
| TN | 73 |

## 8. Threshold sweep (15%–35%, 1% step)

| Threshold | Recall | Precision |
|---|---|---|
| 15–27% | 100.0% | 51.9% |
| 28%+   | 92.9%  | ~50–54% |

Threshold 28% caused 1 missed seizure epoch (FN=1).
**15% was confirmed as the optimal lower bound.**

---

## 9. Discussion

**Precision-Recall trade-off**
Raising the threshold above 27% dropped Recall to 92.9% — one real seizure epoch
was no longer detected. This directly confirmed the design constraint: in seizure
screening, optimizing for Precision at the cost of Recall is clinically unacceptable.
The 13 false positives at threshold 15% represent unnecessary alerts, but each can
be reviewed by a clinician. The alternative — missing a seizure — cannot be corrected
after the fact.

**Pseudo-label generalization gap**
Internal Precision (chb01_03 split): ~74% (XGBoost, default threshold 50%) → External Precision (chb01_04): 51.9%.
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
├── train.ipynb      # preprocessing → feature extraction → model training → save .pkl
├── validate.ipynb   # load .pkl → chb01_04 validation + threshold sweep
├── pipeline.ipynb   # load .pkl → inference on any EDF window
└── README.md
```
---

## 12. Stack

`Python` `MNE` `XGBoost` `scikit-learn` `imbalanced-learn` `scipy` `nolds` `joblib`

## 13. Data

[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (PhysioNet, open access)
Download separately and place in a local directory.
Set DATA_DIR in each notebook to the folder path before running.
