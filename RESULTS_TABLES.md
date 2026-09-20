# All results tables

> **⚠ SUPERSEDED — do not cite (2026-09-14).** Built before the unconstrained classical baselines were run at every training size and before test-set scores existed at every size. Replaced by `COMPLETE_RESULTS.md` (`23_build_tables.py`).


Generated directly from `baseline_results.csv`, `sweep_results.csv`, `final_results.csv` and `qsvm_results.csv` by script — no hand-copied numbers.

## Table 1 — Master results (every model, headline numbers)

Test AUC and PR-AUC at full training data (N = 68,296). Validation AUC at the two ends of the scarcity sweep. **Bold** = best in column.

| Model | Family | Trainable params | Test AUC | Test AUC, drift excluded | Test PR-AUC | Val AUC, N=571 | Val AUC, N=68,296 |
|---|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting (histogram) | Classical | — | **0.8316** | 0.9494 | 0.4312 | — | — |
| Random Forest | Classical | — | 0.8225 | **0.9497** | **0.4338** | 0.8260 | **0.8950** |
| Hybrid Quantum Neural Network — data re-uploading | Quantum variational | 161 | 0.8074 | 0.9019 | 0.3740 | 0.6237 | 0.8674 |
| Hybrid Quantum Neural Network — classical pre-layer | Quantum variational | 154 | 0.8074 | 0.9025 | 0.3841 | 0.5577 | 0.8573 |
| Hybrid Quantum Neural Network — frozen circuit (control) | Quantum variational | 106 | 0.8067 | 0.9007 | 0.3819 | 0.5390 | 0.8608 |
| Multi-Layer Perceptron (unconstrained) | Classical | 71 | 0.8041 | 0.8965 | 0.3636 | — | — |
| Matched Multi-Layer Perceptron — twin of re-uploading HQNN | Classical, matched | 165 | 0.7987 | 0.8867 | 0.3478 | 0.6412 | 0.8503 |
| Logistic Regression | Classical | 13 | 0.8016 | 0.8846 | 0.3021 | — | — |
| Matched Multi-Layer Perceptron — twin of pre-layer HQNN | Classical, matched | 155 | 0.7876 | 0.8773 | 0.3448 | 0.6632 | 0.8426 |
| Matched Multi-Layer Perceptron — twin of frozen HQNN | Classical, matched | 111 | 0.7673 | 0.8540 | 0.2957 | 0.6811 | 0.7973 |
| Isolation Forest | Classical, no labels | — | 0.7568 | 0.8645 | 0.2219 | **0.8349** | 0.8423 |
| Single-Threshold Rule | Rule, no labels | — | 0.7113 | 0.7718 | 0.2510 | — | — |
| Quantum Support Vector Machine (quantum kernel) | Quantum kernel | — | — | — | — | 0.7522 | — |

Quantum Support Vector Machine has no full-data or test numbers: its kernel costs O(N²) circuit evaluations, which is ~72 h at N = 68,296.

**Read the bold marks with care — two columns are ties, not wins.** Random Forest
and Gradient Boosting are 0.0003 apart on drift-excluded AUC and 0.0026 apart on
PR-AUC, both inside Random Forest's own seed spread (0.0028). Report them as tied.

Gradient Boosting, the unconstrained Multi-Layer Perceptron, Logistic Regression
and the Single-Threshold Rule show "—" for validation AUC because they were not
part of the scarcity sweep; they were run once at full data as baselines.


## Table 2 — Scarcity sweep: validation AUC by training-set size

Anomaly rate held at 3.5% throughout. 5 seeds per cell; Quantum Support Vector Machine 3 seeds. **Bold** = best at that size.

| Model | N = 571 | N = 1,429 | N = 2,857 | N = 14,286 | N = 68,296 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8260 | **0.8472** | **0.8621** | **0.8891** | **0.8950** |
| Isolation Forest | **0.8349** | 0.8431 | 0.8425 | 0.8461 | 0.8423 |
| Hybrid Quantum Neural Network — data re-uploading | 0.6237 | 0.7402 | 0.7751 | 0.8641 | 0.8674 |
| Hybrid Quantum Neural Network — classical pre-layer | 0.5577 | 0.6796 | 0.7544 | 0.8542 | 0.8573 |
| Hybrid Quantum Neural Network — frozen circuit (control) | 0.5390 | 0.6667 | 0.7447 | 0.8542 | 0.8608 |
| Matched Multi-Layer Perceptron — twin of re-uploading HQNN | 0.6412 | 0.7834 | 0.8111 | 0.8461 | 0.8503 |
| Matched Multi-Layer Perceptron — twin of pre-layer HQNN | 0.6632 | 0.7557 | 0.7759 | 0.8480 | 0.8426 |
| Matched Multi-Layer Perceptron — twin of frozen HQNN | 0.6811 | 0.7126 | 0.7432 | 0.8380 | 0.7973 |
| Quantum Support Vector Machine (quantum kernel) | 0.7522 | 0.7613 | — | — | — |

## Table 3 — Detection by attack type (test AUC, full data)

**Bold** = best for that attack.

| Model | Speed falsification | Position jump | Reporting gap | Transponder switch-off | Gradual drift |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting (histogram) | 0.994 | 0.919 | **0.936** | 0.961 | 0.637 |
| Random Forest | 0.991 | **0.922** | 0.936 | **0.961** | 0.612 |
| Hybrid Quantum Neural Network — data re-uploading | **0.998** | 0.908 | 0.914 | 0.812 | 0.651 |
| Hybrid Quantum Neural Network — classical pre-layer | 0.997 | 0.907 | 0.909 | 0.819 | 0.650 |
| Hybrid Quantum Neural Network — frozen circuit (control) | 0.997 | 0.908 | 0.908 | 0.814 | 0.651 |
| Multi-Layer Perceptron (unconstrained) | 0.994 | 0.908 | 0.890 | 0.818 | 0.651 |
| Matched Multi-Layer Perceptron — twin of re-uploading HQNN | 0.995 | 0.910 | 0.900 | 0.769 | 0.653 |
| Logistic Regression | 0.988 | 0.908 | 0.887 | 0.780 | **0.664** |
| Matched Multi-Layer Perceptron — twin of pre-layer HQNN | 0.995 | 0.908 | 0.888 | 0.746 | 0.639 |
| Matched Multi-Layer Perceptron — twin of frozen HQNN | 0.943 | 0.870 | 0.883 | 0.741 | 0.624 |
| Isolation Forest | 0.925 | 0.911 | 0.870 | 0.765 | 0.579 |
| Single-Threshold Rule | 0.971 | 0.909 | 0.714 | 0.539 | 0.611 |

Test anomalies by type: Speed falsification 128, Position jump 175, Reporting gap 172, Transponder switch-off 170, Gradual drift 390.


## Table 4 — Hybrid Quantum Neural Network vs its matched classical twin

Quantum AUC minus twin AUC, test set, 10 paired seeds, Wilcoxon signed-rank. ✔ = p < 0.05.

| Comparison | Δ at N=571 | p | Δ at N=68,296 | p |
|---|---:|---:|---:|---:|
| Classical pre-layer HQNN vs matched MLP | -0.0582 | 0.027 ✔ | +0.0198 | 0.010 ✔ |
| Frozen circuit (control) HQNN vs matched MLP | -0.0813 | 0.020 ✔ | +0.0395 | 0.002 ✔ |
| Data re-uploading HQNN vs matched MLP | +0.0164 | 0.557 | +0.0087 | 0.160 |

## Table 5 — Data hunger: validation AUC gained from N=571 to N=1,429

Same seeds (0, 1, 2), identical training windows. Larger gain = more dependent on data.

| Model | N = 571 | N = 1,429 | Gain |
|---|---:|---:|---:|
| Matched Multi-Layer Perceptron — twin of frozen HQNN | 0.6770 | 0.6678 | -0.0092 |
| Isolation Forest | 0.8354 | 0.8422 | +0.0068 |
| Quantum Support Vector Machine (quantum kernel) | 0.7522 | 0.7613 | +0.0092 |
| Random Forest | 0.8328 | 0.8541 | +0.0212 |
| Matched Multi-Layer Perceptron — twin of pre-layer HQNN | 0.6559 | 0.7392 | +0.0833 |
| Matched Multi-Layer Perceptron — twin of re-uploading HQNN | 0.6415 | 0.7731 | +0.1316 |
| Hybrid Quantum Neural Network — data re-uploading | 0.6179 | 0.7666 | +0.1487 |
| Hybrid Quantum Neural Network — frozen circuit (control) | 0.5474 | 0.7044 | +0.1570 |
| Hybrid Quantum Neural Network — classical pre-layer | 0.5455 | 0.7489 | +0.2034 |

The negative gain for the twin of the frozen HQNN is instability, not a real
decline: its seed standard deviation is 0.174 at N = 1,429, the largest of any
model. Every Hybrid Quantum Neural Network gains more than its own matched twin.
