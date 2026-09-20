# Where quantum is better — the complete grid

> **⚠ SUPERSEDED — do not cite (2026-09-14).** This grid contains *extrapolated* Quantum Support Vector Machine values (marked ᵉ) and mixes validation-only results. It is being replaced by `COMPLETE_RESULTS.md`, built by `23_build_tables.py` from `full_sweep_results.csv` and `qsvm_results.csv` — every number measured, validation and test, 5 seeds, no extrapolation.


Generated from `sweep_results.csv`, `baseline_sweep_results.csv` and `qsvm_results.csv`. Every model at every data size; extrapolated cells marked ᵉ.

## A. Every model at every training-set size (validation AUC)

ᵉ = extrapolated, not measured. Everything else is measured, 5 seeds (Quantum Support Vector Machine: 3). **Bold** = best quantum and best classical at that size.

| Model | Family | N=571 | N=1,429 | N=2,857 | N=14,286 | N=68,296 |
|---|---|---:|---:|---:|---:|---:|
| HQNN — data re-uploading | Quantum | 0.624 | 0.740 | **0.775** | **0.864** | **0.867** |
| HQNN — classical pre-layer | Quantum | 0.558 | 0.680 | 0.754 | 0.854 | 0.857 |
| HQNN — frozen circuit | Quantum | 0.539 | 0.667 | 0.745 | 0.854 | 0.861 |
| Quantum Support Vector Machine | Quantum | **0.752** | **0.761** | 0.766ᵉ | 0.772ᵉ | 0.775ᵉ |
| Gradient Boosting | Classical | 0.765 | 0.810 | 0.834 | 0.866 | 0.872 |
| Random Forest | Classical | 0.826 | 0.847 | **0.862** | **0.889** | **0.895** |
| Logistic Regression | Classical | 0.835 | **0.855** | 0.848 | 0.861 | 0.849 |
| Multi-Layer Perceptron | Classical | 0.635 | 0.803 | 0.802 | 0.851 | 0.863 |
| Matched MLP — re-uploading twin | Classical | 0.641 | 0.783 | 0.811 | 0.846 | 0.850 |
| Matched MLP — pre-layer twin | Classical | 0.663 | 0.756 | 0.776 | 0.848 | 0.843 |
| Matched MLP — frozen twin | Classical | 0.681 | 0.713 | 0.743 | 0.838 | 0.797 |
| Isolation Forest (no labels) | Classical | **0.835** | 0.843 | 0.842 | 0.846 | 0.842 |
| Single-Threshold Rule (no labels) | Classical | 0.749 | 0.749 | 0.749 | 0.749 | 0.749 |

## B. Best quantum vs best classical at each size

| N | Best quantum | AUC | Best classical | AUC | Gap |
|---:|---|---:|---|---:|---:|
| 571 | Quantum Support Vector Machine | 0.752 | Isolation Forest (no labels) | 0.835 | -0.083 |
| 1,429 | Quantum Support Vector Machine | 0.761 | Logistic Regression | 0.855 | -0.094 |
| 2,857 | HQNN — data re-uploading | 0.775 | Random Forest | 0.862 | -0.087 |
| 14,286 | HQNN — data re-uploading | 0.864 | Random Forest | 0.889 | -0.025 |
| 68,296 | HQNN — data re-uploading | 0.867 | Random Forest | 0.895 | -0.028 |

## C. Which classical models each quantum model beats

Count out of 9 classical models, and the ones beaten. This is where quantum is better.

| Quantum model | N=571 | N=1,429 | N=2,857 | N=14,286 | N=68,296 |
|---|---:|---:|---:|---:|---:|
| HQNN — data re-uploading | 0/9 | 1/9 | 2/9 | 7/9 | 7/9 |
| HQNN — classical pre-layer | 0/9 | 0/9 | 2/9 | 6/9 | 6/9 |
| HQNN — frozen circuit | 0/9 | 0/9 | 1/9 | 6/9 | 6/9 |
| Quantum Support Vector Machine | 5/9 | 3/9 | 2/9ᵉ | 1/9ᵉ | 1/9ᵉ |

## D. Where each HQNN overtakes its own matched classical twin

Crossover estimated by linear interpolation in log N between the two measured sizes where the gap changes sign.

| Pair | Δ N=571 | Δ N=1,429 | Δ N=2,857 | Δ N=14,286 | Δ N=68,296 | Crossover |
|---|---:|---:|---:|---:|---:|---:|
| HQNN — classical pre-layer vs twin | -0.105 | -0.076 | -0.021 | +0.006 | +0.015 | N ≈ 9,965 (~353 labelled anomalies) |
| HQNN — data re-uploading vs twin | -0.018 | -0.043 | -0.036 | +0.018 | +0.017 | N ≈ 8,360 (~296 labelled anomalies) |
| HQNN — frozen circuit vs twin | -0.142 | -0.046 | +0.002 | +0.016 | +0.064 | N ≈ 2,794 (~99 labelled anomalies) |

## E. Quantum Support Vector Machine: measured vs extrapolated

Power law AUC = a − b/√N fitted to the two measured points. Error band from backtesting the same two-point fit on the three flat-curve models with complete data (Isolation Forest, Random Forest, Logistic Regression): typical ±0.015, worst ±0.036.

| N | AUC | Status | Plausible range |
|---:|---:|---|---|
| 571 | 0.752 | measured | ± seed spread only |
| 1,429 | 0.761 | measured | ± seed spread only |
| 2,857 | 0.766 | extrapolated | 0.730 – 0.802 |
| 14,286 | 0.772 | extrapolated | 0.736 – 0.808 |
| 68,296 | 0.775 | extrapolated | 0.739 – 0.811 |
