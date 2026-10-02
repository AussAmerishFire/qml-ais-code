# Follow-up experiments — quantum features and the geometric difference

Run 2026-09-18 23:18 → 2026-09-19 02:07, both exit 0. Every number below is
computed from `results/qfeat_results.csv` and `results/geometric_difference.csv`.

---

## 1. Do quantum features improve Random Forest and Gradient Boosting?

`scripts/26_quantum_features_rf.py` · 500 fits · 5 sizes × 10 seeds × 5 arms × 2 models

The two industry-standard models, with their tuned settings unchanged, trained on
the 12 engineered features alone (**base**) or with 16 extra features from the
QSVM's quantum circuit (**+Q**). Three classical controls add features computed
from the same 8 inputs: random Fourier (**+RFF**), random tanh (**+TANH**), and
the 8 inputs themselves (**+PCA**).

### Answer: no — quantum features make both models worse on test

Test AUC, mean of 10 seeds:

| Model | N | base | +Q | +RFF | +TANH | +PCA |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 571 | **0.7640** | 0.7404 | 0.7458 | 0.7523 | 0.7608 |
| | 1,429 | **0.7920** | 0.7727 | 0.7750 | 0.7776 | 0.7887 |
| | 2,857 | **0.8039** | 0.7904 | 0.7926 | 0.7893 | 0.8001 |
| | 14,286 | **0.8177** | 0.8079 | 0.8078 | 0.8039 | 0.8109 |
| | 68,296 | **0.8234** | 0.8138 | 0.8151 | 0.8110 | 0.8163 |
| Gradient Boosting | 571 | **0.7363** | 0.7222 | 0.7114 | 0.7233 | 0.7288 |
| | 1,429 | **0.7633** | 0.7573 | 0.7567 | 0.7596 | 0.7598 |
| | 2,857 | **0.7892** | 0.7858 | 0.7835 | 0.7827 | 0.7890 |
| | 14,286 | **0.8148** | 0.8083 | 0.8074 | 0.8063 | 0.8129 |
| | 68,296 | **0.8315** | 0.8264 | 0.8272 | 0.8267 | 0.8308 |

- **+Q is below base in all 10 cells**: significantly for Random Forest at all five
  sizes (−0.010 to −0.024, every p ≤ 0.004) and for Gradient Boosting at three
  (N = 571, 14,286, 68,296).
- **+Q is never better than the best classical control.** Against +RFF, the matched
  classical random-feature control, no Random Forest difference is significant at
  any size. Adding *any* 16 random nonlinear features dilutes a tree ensemble,
  quantum or not.
- **Base wins every cell.** Nothing added helps on test.

### The validation/test reversal — why this cannot be claimed as a win

For Random Forest at N ≥ 14,286, **validation** showed exactly what a quantum
advantage would look like: +Q above base *and* significantly above every classical
control (+0.006 to +0.010, p ≤ 0.049). On the held-out **test** set the same
comparison reversed: +Q significantly *below* base (−0.0096, p = 0.002).

Nothing was selected on either split, so both are fair estimates — of different
vessels (364 in validation, 609 in test). An effect that flips sign between two
groups of ships is not a property of the model. **Had only validation been
examined, this would have been reported as a quantum advantage.** It is the
strongest argument in the study for keeping a separate test set.

---

## 2. Could any quantum kernel beat classical kernels on this data?

`scripts/25_geometric_difference.py` · Huang et al. (2021), *Power of data in quantum
machine learning*, Nature Communications, DOI 10.1038/s41467-021-22539-9

The QSVM's exact quantum kernel, rebuilt at N = 571 and 1,429 (5 seeds each),
compared against 8 classical kernels on the same 8 inputs. A classical kernel
counts only if it can actually fit (training-error term g_tra ≤ 0.1).

### Answer: the room for a quantum advantage is small

| N | Geometric difference g (min over classical kernels) | Largest possible g | g as share of max | Complexity factor g² |
|---:|---:|---:|---:|---:|
| 571 | **2.91** (2.79–3.16), 5/5 seeds | 23.9 | 12% | ≈ 8× |
| 1,429 | **3.83** (3.78–4.15), 5/5 seeds | 37.8 | 10% | ≈ 15× |

λ = 0.001; λ = 0.01 gives 2.88 and 3.99. At λ ≥ 0.1 no classical kernel fits,
so those values carry no information.

- **What g means.** A classical Gaussian kernel can learn *any* function the
  quantum kernel can, needing at most g² times the model complexity. The
  maximum possible gap is g = √N. Measured g is about a tenth of that.
- **It grows more slowly than an advantage would need.** From 571 to 1,429
  windows, g grew by N^0.32. The ceiling grows by N^0.5. The share falls from
  12% to 10%.
- **Only the sharpest classical kernel matches it.** The quantum kernel treats
  most pairs of windows as nearly unrelated (mean off-diagonal similarity
  0.05–0.09). Only the narrowest Gaussian kernel tested (30× the median
  heuristic) can both fit and match it. Every smoother kernel either failed to
  fit or sat far away.

### On the real anomaly labels

This is the second step of Huang et al.'s test. The question is which kernel
describes *these* labels more simply; lower complexity means better expected
generalisation.

| N | λ | Quantum complexity s_Q | Best classical s_C | Quantum simpler? |
|---:|---:|---:|---:|---|
| 571 | 0.01 | 144–273 | 272–376 | **yes, 5 of 5 seeds (~35% lower)** |
| 571 | 0.001 | 169–695 | 289–671 | 3 of 5 seeds |
| 1,429 | 0.01 | 621–718 (fitting seeds) | 693–732 | yes, 2 of 2 fitting seeds, marginally |
| 1,429 | 0.001 | 2,212–3,835 | 1,487–1,926 | **no, 0 of 5 — quantum kernel memorises (s_Q > N)** |

**This is the one theoretical point in quantum's favour.** At the smallest
training size, with moderate regularisation, the quantum kernel describes the
real anomaly labels more simply than any classical kernel, on every seed. That
matches the measured QSVM result: it was the only quantum model that did not
collapse at N = 571. By N = 1,429 the edge is gone under light regularisation.

**Limits.** Every complexity here is a sizeable fraction of N (0.25N to 2.7N).
So neither kernel's generalisation bound is tight, and the theory cannot certify
an advantage for either. These results cover one feature map: 8 qubits, IQP,
one repetition, PCA inputs. Huang et al. show g generally grows with qubit
count, so a 12-qubit map could leave more room.

---

## What this adds to the paper

1. **Quantum features do not help the models industry uses.** They lower test
   AUC and are never better than the best classical control built from the same
   inputs. There is no hybrid "quantum
   feature + Random Forest" advantage on AIS data at 8 qubits.
2. **The geometric difference puts a number on the ceiling.** A classical kernel
   is within a factor of about 3–4 of the quantum kernel on every possible task,
   out of a maximum possible gap of 24–38. Any quantum-kernel advantage on this
   data is bounded and small.
3. **A narrow theoretical edge where the data is scarcest.** At 571 windows the
   quantum kernel fits the real labels more simply than classical kernels,
   consistent with the QSVM's relative strength there. It does not persist.
4. **A methodological result worth stating.** A quantum-specific Random Forest
   gain, significant against every control on validation, reversed on test.
