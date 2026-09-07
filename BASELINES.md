# Classical baselines — what we compare against, and why each one is there

Every model below earns its place by answering a *different* question a reviewer
will ask. None is decorative. Nothing here is preference; each choice is tied
either to a published result or to a measured property of our dataset.

The governing constraint comes from **Bowles, Ahmed & Schuld (2024)**, who
benchmarked QML models against properly-tuned classical baselines and found the
quantum advantage reported in much of the literature disappears once the
classical side is tuned honestly. That paper is the reason this file exists: if
we hand-wave the baselines, our result is worthless regardless of which way it
points.

---

## 1. The five baselines

| # | Model | The question it answers | Why it must be in the paper |
|---|-------|------------------------|-----------------------------|
| 1 | **Percentile rule** on `spd_mismatch_max` | Do you need machine learning at all? | This is what maritime operations centres actually run today — a threshold on one physical inconsistency. `spd_mismatch_max` is our strongest single feature (\|AUC−0.5\| = 0.235). If the QNN cannot beat one threshold, there is no paper. Uses **zero labels**. |
| 2 | **Isolation Forest** | Do you need *labels* at all? | Unsupervised. Its score does not change when labelled anomalies get scarce, so on the headline scarcity plot it is a flat horizontal line. At the far-left end (20 anomalies) it is the model to beat, and it is the honest competitor there. Liu, Ting & Zhou (2008), 6,002 citations — the standard method for this problem shape. |
| 3 | **Logistic regression** | How much of this problem is linear? | The floor for supervised learning. One free hyperparameter (regularisation strength `C`), deterministic, no seed variance. If LR reaches AUC 0.95 the task is easy and there is no room for a quantum result to be interesting. |
| 4 | **Gradient-boosted trees** (and Random Forest) | What does the *best* classical model do? | The hardest baseline to beat and the one reviewers demand. Grinsztajn, Oyallon & Varoquaux (2022) showed tree ensembles still outperform deep learning on typical tabular data — 8 features and 68k rows is exactly "typical tabular". Omitting this is the single most common way QML papers manufacture an advantage. |
| 5 | **Parameter-matched MLP** | Is any difference actually *quantum*? | **The scientifically load-bearing comparison.** Same input, same optimiser, same loss, same number of trainable parameters as the QNN, same training budget. Every difference between #5 and the QNN is attributable to the circuit and nothing else. Without it, a QNN win could just be capacity or optimiser luck. |

Baselines 1 and 2 are cheap — minutes. Baselines 3–5 are the experimental core.

---

## 2. Parameter matching (measured, not estimated)

The QNN is `AngleEmbedding` (0 trainable parameters — it only loads data) followed
by `StronglyEntanglingLayers` on 8 qubits, read out as `expval(PauliZ(0))` through
a trainable scale and bias.

Counts confirmed by running `qml.StronglyEntanglingLayers.shape()`:

| Circuit depth L | Weight shape | Circuit params | + readout | **Total T** |
|---|---|---:|---:|---:|
| 1 | (1, 8, 3) | 24 | 2 | **26** |
| 2 | (2, 8, 3) | 48 | 2 | **50** |
| 3 | (3, 8, 3) | 72 | 2 | **74** |
| 4 | (4, 8, 3) | 96 | 2 | **98** |

An MLP with 8 inputs, one hidden layer of width h, and one output has
`8h + h + h + 1 = 10h + 1` trainable weights:

| h | Params |
|---|---:|
| 2 | 21 |
| 3 | 31 |
| **5** | **51** |
| 7 | 71 |
| 10 | 101 |

**Pairing rule: choose L first, then set h so the two counts are within ~5%.**
At the depth we expect to use, L = 2 gives T = 50, and h = 5 gives 51 parameters.
That is a 2% mismatch in the classical model's *favour*, which is the safe
direction — we are not handicapping the baseline.

T is also the quantity in the Caro et al. bound, sqrt(T/N). Recording it exactly
is not bookkeeping; it is the independent variable of the hypothesis.

---

## 3. Rules that make the comparison fair

These are not optional. Each one closes a specific way the result could be
attacked.

1. **All hyperparameters tuned on `dataset_val.csv` only** (17,381 windows, 537
   anomalies). The test split is touched **once**, at the end, for the final
   numbers. Every model gets the same tuning budget — the same number of
   configurations tried. (Bowles et al. 2024)
2. **Same features, same scaling.** All models read the identical 8 columns
   scaled to [0, pi]. The scaling exists for angle encoding, but the classical
   models get it too, so no model gets a preprocessing edge.
3. **Same splits, same seeds.** Vessel-wise disjoint train/val/test, already
   verified — zero MMSI overlap. 5 seeds per configuration, mean +/- std reported.
4. **Metrics: ROC-AUC and average precision (PR-AUC). Never accuracy.** At a
   3.5% anomaly rate, a model that predicts "normal" for everything scores
   96.5% accuracy while detecting nothing. Reporting accuracy here would be
   actively misleading.
5. **Per-attack breakdown reported for every model.** We already know the five
   attacks differ enormously in difficulty (speed_lie 0.982 down to drift 0.583).
   An aggregate AUC hides which model wins where, and drift is the interesting
   case.
6. **Difference claims need a paired test across seeds**, not a comparison of
   two means. Wilcoxon signed-rank over the 5 paired seed results.

---

## 4. What we deliberately exclude, and why

- **Deep networks / transformers / LSTMs.** 8 features, and the headline regime
  has as few as 20 labelled anomalies. These would overfit catastrophically
  exactly where the experiment lives. Park et al. (2026) is cited as the
  classical state of the art instead of reproduced.
- **RBF-kernel SVM.** This is the natural classical counterpart to a *quantum
  kernel* method. We are not using a quantum kernel — quantum kernels have no
  trainable gates, so the Caro bound does not apply to them, which is why the
  study uses a variational model (see `CITATIONS.md`, Havlicek et al. 2019).
  Optional add-on if a reviewer asks; not load-bearing.
- **Quantum kernel methods as our quantum model.** Same reason, stated
  positively: the hypothesis is specifically about trainable gate count T.

---

## 5. Open design decision, flagged before it becomes a problem

For the label-scarcity experiment we sweep N = 500 / 200 / 100 / 50 / 20. There
are two ways to shrink the training set and **they test different things**:

- **(A) Cut anomalies only, keep all 65,876 normal windows.** Realistic — normal
  traffic is free, confirmed anomalies cost investigator time. But total N stays
  around 66,000, and Caro's bound is stated in terms of *total* training
  examples, so this does not actually probe sqrt(T/N). It also changes the class
  ratio at every step, confounding comparisons across scarcity levels.
- **(B) Shrink the whole training set, holding the 3.5% anomaly rate fixed.**
  20 anomalies gives 571 windows total; 500 anomalies gives 14,286. This is the
  regime Caro's bound describes, and holding the ratio constant means the only
  variable changing across the sweep is N — which is the point.

**Recommendation: (B) is the primary experiment**, because it is the one that
tests the stated hypothesis and because it isolates N as the single variable.
(A) is worth running as a secondary robustness check if time allows; it is the
more operationally realistic framing and a reviewer may prefer it.

---

## New citations introduced by this file

Metadata verified against OpenAlex on 2026-08-31.

**Grinsztajn, Oyallon & Varoquaux (2022)** — *Why Do Tree-Based Models Still
Outperform Deep Learning on Typical Tabular Data?* NeurIPS 2022, Datasets and
Benchmarks Track. 515 citations. arXiv:2207.08815

> Cite when justifying that the strong baseline is a tree ensemble rather than a
> deep network. This is the paper that makes omitting trees indefensible.

**Liu, Ting & Zhou (2008)** — *Isolation Forest.* ICDM 2008, 413–422.
DOI: [10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17) · 6,002 citations

> Cite for the unsupervised zero-label baseline.

```bibtex
@inproceedings{grinsztajn2022tree,
  title     = {Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?},
  author    = {Grinsztajn, Leo and Oyallon, Edouard and Varoquaux, Gael},
  booktitle = {Advances in Neural Information Processing Systems 35,
               Datasets and Benchmarks Track},
  year      = {2022}
}

@inproceedings{liu2008isolation,
  title     = {Isolation Forest},
  author    = {Liu, Fei Tony and Ting, Kai Ming and Zhou, Zhi-Hua},
  booktitle = {2008 Eighth IEEE International Conference on Data Mining (ICDM)},
  pages     = {413--422},
  year      = {2008},
  doi       = {10.1109/ICDM.2008.17}
}
```

---

# 6. RESULTS (run 2026-08-31)

Produced by `10_classical_baselines.py`. Selection on validation only; test read
once. Files: `baseline_results.csv`, `baseline_per_attack.csv`,
`baseline_selection.json`.

## 6.1 Selected configurations

| Model | Configs tried | Val AUC | Chosen |
|---|---:|---:|---|
| rf | 9 | 0.8659 | 300 trees, max_depth 6, no class weight |
| mlp | 12 | 0.8441 | alpha 0.01, lr 0.01, oversample to 35% positives |
| logreg | 10 | 0.8306 | C = 0.01, no class weight |
| iforest | 12 | 0.8269 | 200 trees, max_samples 1024, contamination 0.02 |
| gbt | 12 | 0.8216 | 100 iters, max_depth 3, lr 0.05 |
| rule | 3 | 0.7494 | spd_mismatch_max |

Tuning budgets are 3-12 configurations, comparable across models as required.

## 6.2 Test set (29,292 windows, 1,035 anomalies, 609 unseen vessels)

| Model | Test AUC | PR-AUC | Prec@3.5% | Recall@3.5% | Labels used |
|---|---:|---:|---:|---:|---:|
| rf | 0.7857 +/- 0.0012 | 0.3252 +/- 0.0053 | 0.338 | 0.335 | 2,420 |
| gbt | 0.7854 +/- 0.0013 | 0.3099 +/- 0.0048 | 0.310 | 0.307 | 2,420 |
| **mlp (51 params)** | **0.7711 +/- 0.0107** | **0.3067 +/- 0.0179** | 0.335 | 0.332 | 2,420 |
| logreg | 0.7636 +/- 0.0000 | 0.2826 +/- 0.0000 | 0.298 | 0.295 | 2,420 |
| iforest | 0.7387 +/- 0.0027 | 0.1369 +/- 0.0037 | 0.213 | 0.211 | **0** |
| rule | 0.7113 +/- 0.0000 | 0.2510 +/- 0.0000 | 0.233 | 0.231 | **0** |

`+/-` is the standard deviation over 5 seeds. logreg and rule are deterministic.

**The QNN target is the mlp row: 0.7711 +/- 0.0107.** With 5 seeds the standard
error on that mean is 0.0107/sqrt(5) = 0.0048, so a QNN would need to beat it by
roughly 0.01 AUC before the difference is worth claiming, and the claim must be
backed by a paired test across seeds, not a comparison of means.

## 6.3 Per-attack AUC (normals vs one attack at a time)

| Model | speed_lie | drift | jump | gap | dark |
|---|---:|---:|---:|---:|---:|
| rule | 0.971 | **0.611** | 0.909 | 0.714 | 0.539 |
| gbt | 0.964 | 0.602 | 0.913 | 0.935 | 0.789 |
| mlp | **0.981** | 0.600 | 0.913 | 0.920 | 0.710 |
| rf | 0.970 | 0.596 | **0.921** | **0.934** | **0.793** |
| logreg | 0.976 | 0.589 | 0.906 | 0.898 | 0.722 |
| iforest | 0.875 | 0.574 | 0.891 | 0.894 | 0.699 |

Two findings here are load-bearing for the paper.

**Drift is not solved by anything.** Every model lands between 0.574 and 0.611,
and the *best* score belongs to the single-threshold rule with zero trainable
parameters. No classical learner extracts drift signal that a threshold on one
feature does not already have. This is the clearest opening for the quantum
comparison, and it should be the focus of the per-attack analysis.

**Learning only pays on `dark` and `gap`.** The rule gets 0.539 on dark (near
chance) while learned models reach 0.71-0.79; on gap the rule gets 0.714 against
0.89-0.94. Those two attacks are where supervised learning earns its labels.

## 6.4 Metric disagreement: ROC-AUC vs PR-AUC

Isolation Forest ranks 4th on ROC-AUC (0.7387) but **last by a wide margin on
PR-AUC** (0.1369 against logreg's 0.2826 -- less than half). The single-threshold
rule, ranked last on ROC-AUC, beats Isolation Forest on PR-AUC (0.2510).

This is the reason BASELINES.md rule 4 demands both metrics. ROC-AUC averages
over all thresholds including operationally useless ones; PR-AUC weights the
high-confidence region an analyst actually works in. On validation ROC-AUC alone
it looked as though 2,420 hand-labelled anomalies bought only ~0.4 percentage
points over an unsupervised method. On test PR-AUC the same labels **more than
double** precision-recall performance. **Labels buy a great deal; ROC-AUC just
hides it.** Report both.

## 6.5 Validation-to-test drop, and why it is not overfitting

Every model scores 4-9 points lower on test than on validation. That pattern
would normally suggest overfitting to the validation set during selection --
but the `rule` baseline, which has **zero trainable parameters** and only three
candidate configurations, drops 3.8 points as well. A model that cannot overfit
still drops, so the cause is a genuine difference between the splits.

Measured cause -- attack composition (share of each split's anomalies):

| Split | speed_lie | drift | jump | gap | dark |
|---|---:|---:|---:|---:|---:|
| train | 16.0% | 32.8% | 17.8% | 16.7% | 16.7% |
| val | 13.6% | 27.7% | 20.1% | 18.6% | 19.9% |
| test | 12.4% | **37.7%** | 16.9% | 16.6% | 16.4% |

Test carries **+9.9 percentage points more drift** than validation, and drift is
the hardest attack by a wide margin.

Counterfactual check: resample test anomalies to match the validation attack mix
(200 draws, all normals retained, so only the mix changes):

| Model | Val AUC | Test AUC | Test re-mixed to val proportions | Gap explained |
|---|---:|---:|---:|---:|
| logreg | 0.8306 | 0.7636 | 0.7898 +/- 0.0035 | 39% |
| rf | 0.8675 | 0.7865 | 0.8150 +/- 0.0033 | 35% |

So roughly a third of the drop is attack mix; the remainder is that the 609 test
vessels are simply harder than the 364 validation vessels.

**Decision: keep the splits as they are.** The vessel-wise split is verified
(42 property checks, 20 independent-recomputation checks), every model is scored
on the identical test set so no comparison is affected, and a harder test set is
the *conservative* direction -- it makes a claimed quantum advantage harder to
obtain, not easier. Declared as a limitation: attack composition is balanced
per-vessel but not per-window across splits, so absolute AUCs are depressed
relative to validation by roughly 4 points beyond the mix effect.

## 6.6 Other observations for the write-up

- **rf and gbt are tied**, not ranked: 0.7857 +/- 0.0012 vs 0.7854 +/- 0.0013,
  a difference of 0.0003. Do not report one as beating the other.
- **The validation ranking did not survive contact with test.** gbt was 5th of 6
  on validation and joint-1st on test. This is a concrete illustration of why
  selection and evaluation must use different splits, and is worth a sentence in
  the methods section.
- **Logistic regression is competitive** (0.7636 against rf's 0.7857), and the
  grid chose the most heavily regularised setting available (C = 0.01). Most of
  the accessible signal in these 8 features is linear.
- **The mlp has by far the largest seed variance** (0.0107, roughly 8x rf's).
  Small networks trained by gradient descent are unstable here -- which is
  precisely why the QNN comparison needs 5 seeds and a paired test.
