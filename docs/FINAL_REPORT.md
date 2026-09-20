# Final report — quantum machine learning for AIS maritime anomaly detection

Compiled 2026-09-09. Sources: `results/final_results.csv` (endpoint runs, test set),
`results/sweep_results.csv` (230-run scarcity sweep, validation),
`results/baseline_results.csv` (classical baselines, test set).

**Completeness: 120/120 runs, both endpoints at the full 10 seeds.** The run
was interrupted once (the machine slept at 22:01 on 08-09 despite process-scoped
sleep prevention — a lid close or hibernate overrides it) and resumed from its
checkpoint without losing work.

---

## 1. Bottom line

Three findings, in descending order of how much they should shape the paper.

**(a) The quantum-versus-classical gap inverts with training-set size, in the
direction opposite to the generalisation bound's prediction.** Quantum models
are significantly *worse* than depth- and parameter-matched classical twins when
data is scarce, and significantly *better* when data is plentiful.

**(b) With little training data, simple models win and every neural network
collapses — quantum or classical.** At N = 571 the three best detectors are
Isolation Forest, Logistic Regression and Random Forest (validation AUC
0.826–0.835); every neural network, all six quantum and matched classical models
plus the small 71-parameter Multi-Layer Perceptron, scores 0.539–0.681. The failure
is about model complexity, not about quantum computing.

> **Correction (2026-09-14).** An earlier version of this finding read *"below
> ~100 labelled anomalies, an unsupervised method using no labels at all beats
> every supervised model tested."* That was false. Logistic Regression was not in
> the original sweep; once run, it ties Isolation Forest at N = 571
> (0.8348 vs 0.8349) and beats it at N = 1,429 (0.8552 vs 0.8431). The claim held
> only against the models that happened to be tested.

**(c) Quantum models beat their matched twins at full data, but no quantum model
beats the best unconstrained classical model.** Gradient boosting reaches test
AUC 0.8316 against the best quantum 0.8074. The quantum advantage is real only
under a parameter-and-depth constraint that nobody deploying a detector would
actually impose.

---

## 2. The inversion, with paired tests

Quantum minus its matched classical twin, on the **test set**, Wilcoxon
signed-rank over paired seeds:

| Pair | N=571 | | N=68,296 | |
|---|---:|---|---:|---|
| | Δ | p | Δ | p |
| **Q_pre** − MLP_pre | **−0.0582** | **0.027** ✔ | **+0.0198** | **0.010** ✔ |
| **Q_frozen** − MLP_frozen | **−0.0813** | **0.020** ✔ | **+0.0395** | **0.002** ✔ |
| Q_reup − MLP_reup | +0.0164 | 0.557 | +0.0087 | 0.160 |

All cells are 10 paired seeds, test set, Wilcoxon signed-rank.

Two of three architectures flip sign **and reach significance at both ends**.
That is a replicated inversion, not a single noisy comparison.

Caro et al. (2022) bound the generalisation gap of a model with `T` trainable
gates as `sqrt(T/N)`. The practical inference everyone draws — smaller quantum
models should therefore win on small data — is what these data contradict.
**Nothing here contradicts the theorem itself**, which is an *upper* bound. What
the result shows is that at these sizes the binding constraint is optimisation,
not generalisation: quantum circuits are simply harder to fit from few examples,
and that dominates whatever room the bound leaves.

The within-architecture test gives no support to the bound's inference either.
`Q_frozen` is `Q_pre` with its 48 rotation angles held at random initialisation —
`T = 106` against `154`, everything else identical. The inference predicts the
lower-`T` model should do *better* when data is scarce. It does not: across the
sweep `frozen − trained` is −0.019, −0.013, −0.010, −0.000, +0.004 as N rises, and
**no value is significant** (5 seeds, every p ≥ 0.625; at 10 seeds on the test
set, N = 571 gives +0.004, p = 1.000, and N = 68,296 gives −0.001, p = 0.922).
The defensible statement is that **freezing the 48 trainable quantum angles made
no measurable difference at any training-set size** — so the lower-`T` model
got none of the small-data benefit the inference predicts.

> **Correction (2026-09-14).** This paragraph previously read *"Monotone, and
> backwards"*, presenting the point estimates as a trend against the bound. None
> of the differences is significant at 5 or 10 seeds, so no direction can be
> claimed.

### The exception that sharpens the claim

`Q_reup` — the architecture with **data re-uploading**, re-encoding the input
before each circuit block — shows **no significant difference from its twin at
either endpoint**. It is the best quantum model at N=571 and ties its twin at
full data.

So the honest claim is not "quantum is worse when data is scarce." It is
**"quantum is worse when data is scarce, unless the encoding is repeated."**
That is consistent with Schuld, Sweke & Meyer (2021) on encoding repetition
widening the representable frequency spectrum, and it is a more specific and
more useful finding than a blanket negative.

---

## 3. Test-set results

### At full data (N = 68,296, 10 seeds)

| Model | T | Test AUC | PR-AUC | drift | dark |
|---|---:|---:|---:|---:|---:|
| **Gradient boosting** (unconstrained) | — | **0.8316** | **0.4312** | 0.637 | 0.961 |
| Random forest (unconstrained) | — | 0.8225 | 0.4338 | 0.612 | 0.961 |
| Q_reup | 161 | 0.8074 ± 0.014 | 0.3740 | 0.643 | 0.819 |
| Q_pre | 154 | 0.8074 ± 0.011 | 0.3841 | 0.646 | 0.802 |
| Q_frozen | 106 | 0.8067 ± 0.010 | 0.3819 | 0.656 | 0.800 |
| MLP_reup | 165 | 0.7987 ± 0.006 | 0.3478 | 0.656 | 0.759 |
| MLP_pre | 155 | 0.7876 ± 0.011 | 0.3448 | 0.631 | 0.733 |
| MLP_frozen | 111 | 0.7673 ± 0.059 | 0.2957 | 0.625 | 0.733 |
| Isolation Forest (0 labels) | — | 0.7568 | 0.2219 | 0.579 | 0.765 |
| Threshold rule (0 labels) | — | 0.7113 | 0.2510 | 0.611 | 0.539 |

All three quantum models beat all three matched twins. All three are beaten by
gradient boosting, which is under no parameter constraint.

### At N = 571 (20 labelled anomalies, 10 seeds)

| Model | T | Test AUC |
|---|---:|---:|
| MLP_frozen | 111 | **0.6763 ± 0.070** |
| Q_reup | 161 | 0.6599 ± 0.042 |
| MLP_pre | 155 | 0.6494 ± 0.046 |
| MLP_reup | 165 | 0.6435 ± 0.053 |
| Q_frozen | 106 | 0.5950 ± 0.046 |
| Q_pre | 154 | 0.5912 ± 0.071 |

Every one of these is far below Isolation Forest's 0.8349 validation AUC at the
same N, which brings us to the second finding.

---

## 4. The small-data result: simple beats neural

Validation AUC, 5 seeds, identical training windows. "Simple" means Isolation
Forest, Logistic Regression and Random Forest; "neural" means every model with
learned hidden layers — the three Hybrid Quantum Neural Networks, their three
matched Multi-Layer Perceptrons, and the 71-parameter Multi-Layer Perceptron.

| N | Best simple model | AUC | Best neural model | AUC | Simple ahead by |
|---:|---|---:|---|---:|---:|
| 571 | Isolation Forest | 0.8349 | Matched MLP, frozen-HQNN twin | 0.6811 | **0.154** |
| 1,429 | Logistic Regression | 0.8552 | Multi-Layer Perceptron | 0.8034 | 0.052 |
| 2,857 | Random Forest | 0.8621 | Matched MLP, re-uploading twin | 0.8111 | 0.051 |

At N = 571 the gap between simple and neural models is 0.15 AUC, and it is the
same gap for quantum and classical neural networks alike. **Scarce-data failure
is a property of learning hidden representations from few examples, not of the
quantum circuit.**

On labels specifically: Isolation Forest, which uses none, ties Logistic
Regression trained on 20 labelled anomalies at N = 571 (0.8349 vs 0.8348). By
N = 1,429 (~51 labelled anomalies) Logistic Regression is ahead (0.8552 vs
0.8431). So twenty labels buy a linear model essentially nothing over no labels,
and supervision starts to pay somewhere between 20 and 50 labelled anomalies.
That is a narrower claim than the one this section originally made, and the
original is recorded as corrected in §1(b).

Random Forest degrades most gracefully: 0.8260 at N = 571 rising to 0.8950 at
full data, the best model at every size from N = 2,857 upward.

---

## 5. The detector itself

The dataset is 114,969 windows from 8,156,509 raw AIS reports, 3.47% anomalous,
split vessel-wise with zero MMSI overlap. Twelve features. Five injected attack
types with severities derived from measured properties of the clean data.

Adding four trend/direction features mid-study moved the best classical test AUC
from 0.7854 to **0.8316**, and PR-AUC from 0.3099 to **0.4312** — a 39% gain on
the metric that matters at 3.5% imbalance. The largest single gain was on the
`dark` attack, 0.79 → 0.96.

Per-attack, at the 3.5% alert budget, the detector is strong on four of five
attack types and **weak on drift** — 0.61–0.66 for every model including a
zero-parameter threshold rule. Drift is 37.7% of test anomalies. Diagnosis is in
`scripts/PIPELINE.py`: `a_drift` slides position along a fixed bearing while leaving SOG,
COG and Heading untouched, so it produces no spike for any max-based feature.
`course_resid_mean` was added to target it and is the best single drift feature
measured (0.6554 univariate), but drift remains unsolved.

---

## 6. Limitations to declare

- **Tuning budget of 2 configurations per model**, cut from 8 against a compute
  window. Equal on both sides, but small; the paper must say so.
- **`MLP_frozen` is unstable** — seed standard deviation 0.059 at full data and
  up to 0.149 in the sweep. Its gap is significant (p=0.002) but that instability
  should be stated alongside it.
- **Every quantum model carries a classical Dense(12→8) pre-layer**, forced by
  the 8-qubit budget. The quantum side compresses 12 features into 8 while its
  twin reads all 12 — a handicap, in the conservative direction.
- **The twin wins may partly be a training-budget effect.** The separate
  Multi-Layer Perceptron baseline is a 12-5-1 network with **71 parameters** —
  fewer than any HQNN (106–161) — trained on its own schedule, and it scores
  0.8041 at full data, level with the HQNNs (0.8067–0.8074). The matched twins
  (111–165 parameters, two hidden layers, the same fixed 640 steps as the
  quantum models) score lower (0.7673–0.7987). So the quantum-vs-twin gap shows
  that, at equal size, depth and step budget, the quantum layer trains better; it
  does **not** show that quantum does more per parameter. Earlier drafts called
  this 71-parameter baseline "unconstrained"; that label was wrong and has been
  corrected (2026-09-17).
- **8-qubit noiseless simulation**, not hardware. No noise model.
- **Test read twice** — once by the classical baselines, once by the endpoint
  runs. No selection ever occurred on test, which is what the hold-out rule
  protects, but the paper should state both reads.
- **Injected anomalies, not observed ones.** Real spoofing may differ.
- **One day of US coastal AIS.** No seasonal or geographic generalisation claim.

---

## 7. Verification performed

- 42/42 property assertions (`scripts/07_validate_dataset.py`)
- 22/22 independent-recomputation checks on a separate code path — different
  distance formula, explicit loops, forward azimuth written longhand
  (`scripts/09_independent_check.py`)
- 9 full pipeline rebuilds sweeping the three main thresholds
  (`scripts/08_sensitivity_sweep.py`)
- Depth-matched and parameter-matched classical twins for every quantum model,
  with the classical side always holding at least as many parameters
- Frozen-gate control isolating whether trainable quantum parameters contribute
- Paired significance tests at 10 seeds where the claim depends on them

Bugs found and fixed during the work are recorded in `QNN_RESULTS.md` §3–4 and
`RESULTS_SWEEP.md` §4, including a two-seed false positive, a saturated control
that returned exactly chance, five scripts silently running on a stale
8-feature list, and a verification script importing a superseded module.

---

## 8. What the paper is now

The study set out to test whether quantum models generalise better from few
labels. **They do not — they generalise worse**, significantly, in two of three
architectures, with the effect replicating at both ends of a 120× range of
training-set size, against depth- and parameter-matched controls.

That is a publishable negative result of the kind Bowles, Ahmed & Schuld (2024)
argue the field lacks, and it carries a control their paper does not: a
frozen-gate ablation showing that training the quantum parameters made no
measurable difference at any training-set size.

The research question needs rewriting to match. Suggested direction:
*"Quantum models need more data, not less: a controlled test of
generalisation-bound predictions in maritime AIS anomaly detection."*
Primary finding the inversion; secondary that simple models beat every neural
network, quantum or classical, when data is scarce; the
re-uploading exception as the mechanism worth following up.
