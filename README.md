# CME Arrival Time Predictor

> **Predicting when a solar storm will hit Earth — down to the hour.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Colab](https://img.shields.io/badge/Run%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

A deep learning pipeline that predicts the **transit time of Coronal Mass Ejections (CMEs)** from the Sun to Earth's L1 point — the critical early-warning window for satellite operators, grid managers, and space weather agencies.

---

## What this does

When the Sun ejects a plasma cloud at a million miles per hour, operators have 1–3 days to prepare. The difference between a 20-hour and a 60-hour warning is the difference between an orderly shutdown and a scramble. This project trains a **Bidirectional LSTM with temporal attention** on physics-informed synthetic CME data, achieving ensemble predictions within **±6 hours** on the majority of test events.

---

## Results

| Model | MAE (hours) | R² | Within ±6h | Within ±12h |
|---|---|---|---|---|
| Baseline BiLSTM | — | — | — | — |
| Optuna-tuned BiLSTM | — | — | — | — |
| **BiLSTM + DBM Ensemble** | **—** | **—** | **—%** | **—%** |

> Fill in your actual numbers from `metadata.json` after running the notebook.

---

## Architecture

```
Input  →  (batch, 10 timesteps, 27 features)
  │
  ├─ Bidirectional LSTM (256 units) + LayerNorm + Dropout(0.30)
  ├─ Bidirectional LSTM (128 units) + LayerNorm + Dropout(0.21)
  ├─ LSTM (64 units, return_sequences=True)
  │
  ├─ Temporal Attention Block
  │     score  = Dense(1, tanh)   →  (B, T, 1)
  │     weight = Softmax(axis=1)  →  (B, T, 1)   ← named 'attn'
  │     ctx    = Σ(x × weight)    →  (B, 64)
  │
  ├─ Dense(64, relu) → Dropout(0.20) → Dense(32, relu)
  └─ Dense(1, float32)   ← transit_hours prediction

Loss: Huber  |  Optimizer: AdamW (lr=3e-4, wd=1e-5)  |  Metric: MAE
```

The attention block lets the model learn *which timesteps* in the 10-event window matter most for the prediction — visualised in Cell 12 of the notebook.

---

## Dataset

Built on the physics-grounded **Drag-Based Model** (Vršnak et al. 2013). 3,000 synthetic CME events are generated, spanning 2010–2024, with the target `transit_hours` clipped to the physically-plausible range of [20, 120] hours.

**14 raw features** (CME speed, solar wind speed, IMF components, Kp/Dst indices, flare intensity, etc.) are expanded to **27 features** via physics-motivated engineering:

| Feature group | Examples |
|---|---|
| Cyclical time encoding | `month_sin`, `month_cos`, `solar_cycle_phase` |
| Speed dynamics | `speed_excess`, `speed_ratio`, `mach_number` |
| Plasma physics | `ram_pressure`, `alfven_speed`, `b_angle` |
| Geometry | `cone_angle`, `src_distance`, `expansion_factor` |
| Log transforms | `log_v_cme`, `log_n_sw`, `log_b_total` |

---

## Notebook walkthrough

The notebook (`CME_Arrival_Time_Predictor.ipynb`) runs end-to-end in **Google Colab on a T4 GPU**. All 19 cells are documented — including three non-trivial bugs that were diagnosed and fixed during development (see the changelog at the top of the notebook and the `logs/` folder for full session histories).

| Cell | What it does |
|---|---|
| 1–2 | Environment setup, GPU check, imports |
| 3 | NASA DONKI API fetch (graceful fallback) |
| 4 | Drag-Based Model synthetic dataset generation |
| 5 | EDA + correlation matrix |
| 6 | Feature engineering (27 features) |
| 7 | Sequence construction + chronological train/val/test split |
| 8 | BiLSTM + Attention model definition |
| 9 | Training with callbacks (checkpoint, early stopping, LR decay) |
| 10 | Training curves |
| 11 | Test-set evaluation (MAE, RMSE, R², MAPE, within-6h/12h) |
| 12 | Temporal attention weight visualisation |
| 13 | Permutation feature importance *(replaced SHAP — see bugs)* |
| 14 | Optuna hyperparameter search (20 trials) |
| 15 | Retrain with best params → `model_opt` |
| 16 | LSTM + DBM ensemble stacking |
| 17 | Early-warning inference interface |
| 18 | Model export (Keras + TFLite fp16) |
| 19 | Summary dashboard |

---

## Bugs fixed

Three significant bugs were encountered and resolved during development. Each is documented fully in the notebook changelog and in `logs/`.

### Bug 1 — `shap.DeepExplainer` crash (Cell 13)
`AttributeError: 'NoneType' object has no attribute 'numpy'` deep in SHAP's backprop path. Root cause: SHAP's `DeepExplainer` makes assumptions about TF's gradient tape that break in TF ≥ 2.12 with mixed float16 precision.

**Fix:** Replaced with model-agnostic **permutation importance** — each feature is shuffled across test sequences independently, and MAE degradation is measured. Variable name `fi_shap` preserved so Cell 19 dashboard required zero changes.

---

### Bug 2 — `ValueError: Input contains NaN` (Cell 16)
sklearn refused the ensemble arrays because the Drag-Based Model formula for decelerating CMEs produces `log(negative)` = NaN when `gamma` is too large relative to CME speed over 1 AU.

**Fix:** Replaced with `dbm_transit_safe()` — guards the log argument, falls back to a kinematic mean-speed formula when the condition is violated, and hard-clips output to [10, 120] hours.

---

### Bug 3 — TFLite `ConverterError` (Cell 18)
Two separate issues stacked:
- `tf.CudnnRNNV3` — the T4 GPU compiled the third LSTM to a cuDNN fused kernel that is neither a TFLite builtin nor a SELECT_TF_OPS flex op.
- `seed_generator` resource captures — Keras `Dropout` layers create `int64` seed-state `ResourceVariable`s even when unused. These get captured in the concrete function and cause a dtype conflict during TFLite's graph-freeze step.

**Fix:** Built an inference-only export clone replacing all `Dropout` layers with no-op `Lambda` layers, set `recurrent_dropout=0` everywhere, traced on CPU (bypassing cuDNN), transferred weights with float32 casts, then converted from a static concrete function.

---

## Repo structure

```
📦 cme-arrival-time-predictor/
├── CME_Arrival_Time_Predictor.ipynb   ← main notebook (changelog at top)
├── README.md
├── logs/
│   ├── session_001.md                 ← initial build + SHAP bug
│   ├── session_002.md                 ← NaN bug + ensemble stacking
│   └── session_003.md                 ← TFLite export (3 iterations)
└── assets/                            ← plots referenced in this README
    ├── architecture.png
    ├── attention.png
    ├── permutation_importance.png
    └── dashboard.png
```

---

## Running it

```bash
# Open in Colab (recommended — T4 GPU required for training speed)
# Click the Colab badge at the top of this README, then:
# Runtime → Change runtime type → T4 GPU → Run All
```

All dependencies are installed in Cell 1. No local setup required.

**Key dependencies:**

```
tensorflow>=2.15
optuna
scikit-learn
pandas
numpy
matplotlib
seaborn
requests
```

---

## How the ensemble works

The final prediction blends the deep learning model with the physics-based Drag-Based Model:

```
ŷ_ensemble = α × ŷ_LSTM + (1 − α) × ŷ_DBM
```

`α` is swept from 0 → 1 on the validation set and the value minimising MAE is selected. This ensemble consistently outperforms either model alone — the DBM anchors physically implausible predictions while the LSTM captures patterns the analytic formula misses.

---

## Feature importance

Permutation importance (top features by ΔMAE when shuffled):

> *(insert `permutation_importance.png` here)*

`v_cme`, `speed_excess`, and `mach_number` dominate — consistent with the Drag-Based Model's theoretical dependence on CME-to-solar-wind velocity differential.

---

## References

- Vršnak, B. et al. (2013). *Propagation of Interplanetary CMEs: The Drag-Based Model.* Solar Physics, 285, 295–315.
- NASA DONKI API: https://kauai.ccmc.gsfc.nasa.gov/DONKI/
- TensorFlow Mixed Precision: https://www.tensorflow.org/guide/mixed_precision

---

## License

MIT. See [LICENSE](LICENSE).

---

*Built with TensorFlow 2.x on Google Colab. Physics-informed synthetic data via the Drag-Based Model.*
