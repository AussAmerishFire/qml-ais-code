# The whole study, explained

Written 2026-10-05 for the person writing the paper. It covers everything from
the research question to where the work stands now: what was done, in what
order, why, how every number was calculated, and which file it comes from.

Read it top to bottom once. After that, use it as a reference while you write.
Section 20 lists the verbs to use when you describe each step in the paper.

Every number here was checked against a result file, a log, or the code. The
source is given in brackets, e.g. **[scripts/PIPELINE.py]**.

---

## 1. The study in one paragraph

We took one day of real ship-tracking data (AIS), cut it into 114,969 short
windows of 10 reports each, and computed 12 features per window. We injected
five kinds of fake behaviour into 3.47% of the windows so we knew exactly which
ones were anomalous. We then trained 13 models to tell anomalous windows from
normal ones: three hybrid quantum neural networks, a classical "twin" for each
of them with the same size and depth, a quantum kernel model, and six standard
classical models. We repeated every training run at five training-set sizes,
from 571 to 68,296 windows, with five to ten random seeds each, 1,222 model fits
in total, and compared the models statistically on data from ships none of them
had ever seen. Then we ran two follow-up experiments to test whether quantum
could win anywhere else on this data.

---

## 2. Research question, aim and rationale

### 2.1 Where the question came from

Caro et al. (2022, *Nature Communications*) proved that the generalisation error
of a quantum model with `T` trainable gates, trained on `N` examples, is bounded
by roughly `√(T/N)`. People commonly read this as: *a small quantum model should
learn well from few examples, so quantum models should hold up better than
classical ones when training data is scarce.*

Labelled maritime anomalies **are** scarce. Confirmed cases of spoofing or
transponder switch-off are rare and expensive to verify. So if that reading is
right, maritime surveillance is exactly where quantum machine learning should
help. That is the study's rationale.

### 2.2 The original question and hypothesis

**Original question:** *Do quantum models generalise better than classical
models from few labelled anomalies?*

**Hypothesis (H1):** as the training set shrinks, a quantum model's performance
falls more slowly than that of a classical model of the same size and depth.

### 2.3 What happened to it

The data contradicted H1. The quantum models fell **faster** than their matched
classical twins as data shrank. The question was rewritten on 2026-09-30 to
match the evidence, not the expectation **[docs/RESEARCH_QUESTION.md]**.

### 2.4 The final research question

> **Under what conditions do quantum machine learning models outperform
> classical models in detecting anomalous vessel behaviour in AIS data?**

It has four sub-questions:

| | Sub-question | Short answer |
|---|---|---|
| SQ1 | Against a classical model of the same size, does the advantage depend on training-set size? | Yes. Quantum loses with little data and wins with a lot, the opposite of H1. |
| SQ2 | Do quantum models beat the best classical models? | No, at any size. |
| SQ3 | Do quantum kernels behave differently from variational quantum circuits? | Yes. The kernel does not collapse with scarce data; the circuits do. |
| SQ4 | Is there a measurable ceiling on quantum advantage for this data? | Yes, and it is low. |

### 2.5 Aim

To measure, under controlled conditions, when (if ever) quantum machine-learning
models detect anomalous ship behaviour better than classical models, and to
explain why.

### 2.6 Why "under what conditions" and not "is quantum better"

A yes/no question would have to be answered "no", and would throw away the main
finding: the answer changes sign depending on how much data there is and what
quantum is compared against. Schuld & Killoran (2022, *PRX Quantum*) argue that
"under what conditions" is the more useful question for quantum machine learning
in general.

---

## 3. Background you need

**AIS (Automatic Identification System).** Ships broadcast their position,
speed over ground (SOG), course over ground (COG), heading and identity (MMSI,
a 9-digit number) every few seconds to minutes. Shore stations and satellites
collect it. Ships can falsify it or switch it off, which is how sanctions
evasion and illegal fishing are hidden.

**Anomaly detection.** Each window gets a score; a higher score means "more
likely anomalous". The model is judged on how well its scores rank anomalous
windows above normal ones.

**Hybrid quantum neural network (HQNN).** A classical layer feeds numbers into a
small quantum circuit with trainable rotation angles, whose measured outputs are
combined into a score. Training adjusts the angles and the classical weights by
gradient descent. This is a *variational* quantum model.

**Quantum kernel / QSVM.** A fixed (untrainable) quantum circuit maps each window
to a quantum state. The similarity between two windows is the overlap of their
states. A classical support vector machine (SVM) then learns from those
similarities. Nothing inside the circuit is trained.

**Simulation.** All quantum models ran on a classical computer that simulates an
8-qubit quantum computer exactly, with no noise (PennyLane `default.qubit`).
No real quantum hardware was used.

---

## 4. Data collection

### 4.1 Source

| | |
|---|---|
| Provider | NOAA Office for Coastal Management, via MarineCadastre.gov |
| File | `AIS_2023_01_01.csv`, all US coastal AIS for 1 January 2023 |
| Size | 836 MB, **8,156,509** position reports from **15,932** vessels |
| Licence | US federal data, public domain |
| Columns used | MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselType, Length, TransceiverClass |

**[logs/pipeline_v2.log, docs/INJECTION_AND_TIMELINE.md]**

The raw file is not in the repository: it exceeds GitHub's 100 MB file limit.
Download it again from MarineCadastre to rebuild everything.

### 4.2 Filtering: 8,156,509 → 1,256,434 reports

Applied in this order **[scripts/PIPELINE.py, `load()`]**:

| Step | Rule | Reports left | Vessels | Why |
|---|---|---:|---:|---|
| 0 | Raw | 8,156,509 | 15,932 | |
| 1 | Class A transceivers only | 6,111,077 | 8,428 | Class A is the mandatory, higher-rate system on large commercial ships. Scopes the study to them. |
| 2 | SOG < 40 knots | 6,094,912 | 8,415 | AIS encodes "speed unavailable" as 102.3; 40 kn is above any real merchant ship. |
| 3 | Latitude in [−90, 90], longitude in [−180, 180] | 6,094,912 | 8,415 | Removes impossible coordinates (none found). |
| 4 | SOG ≥ 0.5 knots (under way) | **1,463,926** | 5,150 | Removes moored ships. This step removes 4.6 million reports, 76% of what reached it: most AIS traffic is ships sitting at berth. Median speed before: exactly 0.00 kn; after: 7.50 kn. |
| 5 | Vessels with ≥ 200 reports | **1,256,434** | **2,441** | Ensures each vessel yields about 20 windows, enough to split by vessel. |

### 4.3 Voyages ("tracks")

Each vessel's reports were sorted by time and split into separate voyages
wherever the gap between consecutive reports exceeded **1,800 s (30 min)**
(`GAP_BREAK_S`). Measured: 0.000% of normal Class A reporting intervals reach
1,800 s, so a gap that long means the ship stopped transmitting. This produced
**7,935 voyage tracks** **[logs/pipeline_v2.log]**.

Order matters: the pipeline splits voyages **before** computing differences
between consecutive reports. Done the other way round, it once compared the last
report of one voyage with the first of the next and produced implied speeds of
258,249 knots **[scripts/PIPELINE.py, `per_step()`]**.

### 4.4 Windows

Each track was cut into **non-overlapping windows of 10 consecutive reports**
(`WINDOW = 10`, `STRIDE = 10`). At the median reporting interval of 70 s, 10
reports span about 12 minutes. Non-overlapping because overlapping windows share
reports, and two windows sharing reports but landing on opposite sides of a
train/test split would leak information.

---

## 5. The 12 features

Computed per window, from differences **within one voyage** only
**[scripts/PIPELINE.py, `per_step()` and `windows()`; docs/METHODS.md Part 6]**.

First, per consecutive pair of reports, the pipeline computes:
- `dt` = seconds between the two reports
- `dist` = great-circle (haversine) distance between the two positions, in metres
- `implied speed` = `dist / dt × 1.94384` (metres per second → knots)
- `speed mismatch` = |implied speed − reported SOG|
- `bearing` = the compass direction actually travelled between the two positions

Then, per window:

| # | Feature | Calculation |
|---:|---|---|
| 1 | `sog_mean` | mean reported SOG |
| 2 | `sog_max_jump` | largest change in reported SOG between consecutive reports |
| 3 | `cog_turn_max` | largest change in reported COG between consecutive reports (angles wrapped to 0–180°) |
| 4 | `dt_max` | longest interval between consecutive reports |
| 5 | `spd_mismatch_max` | largest speed mismatch |
| 6 | `crab_mean` | mean angle between reported COG and reported heading |
| 7 | `prev_silence_s` | the silence before this voyage began (non-zero only if a > 30 min gap started it) |
| 8 | `course_resid_mean` | mean angle between reported COG and the bearing actually travelled |
| 9 | `mismatch_slope` | least-squares slope of the speed mismatch across the 10 reports |
| 10 | `straightness` | straight-line distance from first to last position ÷ total path length (1 = perfectly straight) |
| 11 | `mismatch_mean` | mean speed mismatch |
| 12 | `crab_valid` | 1 if heading was reported in the window, 0 if not |

**Design principle.** Four of the five attacks are caught as **internal
contradictions**, not deviations from normal: a ship reports its position *and*
its speed, and the two must agree. Measured in unmodified data, the median speed
mismatch is 0.11 kn and the 99th percentile 3.20 kn. Real ships agree with
themselves.

**History of the feature set.** Thirteen features were first engineered. Pairs
correlated above |0.8| (Spearman, 122,287 windows) were cut to one each, leaving
the first 7 above plus `crab_valid`. On 2026-09-07 four more (rows 8–11) were
added because a probe (`scripts/17_feature_probe.py`) showed every original
feature was a mean or a maximum, and a maximum cannot see a gradual trend. Adding
them moved random-forest AUC from 0.790 to 0.833 and transponder-switch-off AUC
from 0.823 to 0.965 **[scripts/PIPELINE.py, comment above `CORE`]**.

### 5.1 Missing values, transforms and scaling

**[scripts/PIPELINE.py, `split_and_scale()`]**

1. **Missing values.** `crab_mean` is missing when heading was not reported;
   it is filled with the median and `crab_valid` records that it was missing.
   Features 8–11 are median-filled when undefined. *Note for the paper:* these
   medians were computed over all windows before the train/test split. This uses
   no labels, but it is a small leak of statistics across splits; declare it.
2. **Log transform.** Seven features span orders of magnitude, so
   `x → log10(max(x, 0) + 0.01)` is applied to `sog_max_jump`, `cog_turn_max`,
   `dt_max`, `spd_mismatch_max`, `crab_mean`, `prev_silence_s`, `mismatch_mean`.
   `mismatch_slope` can be negative, so it gets a signed log:
   `sign(x) × log10(1 + |x|)`.
3. **Scaling to [0, π].** Each feature is min-max scaled using the **training
   split's** minimum and maximum only, clipped to [0, 1], and multiplied by π.
   π because the quantum circuit encodes each value as a rotation angle, and
   0 to π is the useful range of a single-qubit rotation.

---

## 6. Injecting the anomalies

### 6.1 Why inject instead of using real anomalies

No public dataset labels real AIS spoofing at this scale. Injecting attacks
means every label is known exactly, and their size can be controlled. The cost
is that real attacks may differ; this is a declared limitation.

### 6.2 The five attacks

**[scripts/PIPELINE.py, `a_drift` … `a_dark`; docs/INJECTION_AND_TIMELINE.md]**

| Attack | What the code does | Severity range | Median applied | Basis for the range |
|---|---|---|---:|---|
| **Speed falsification** (`speed_lie`) | multiplies or divides reported SOG from a random point onward; positions untouched | ×1.8 to ×5.0 | ×2.8 | capped so reported speed stays within 0.5–39 kn |
| **Position jump** (`jump`) | moves every position after a random point by a fixed distance on a random bearing | 500 – 20,000 m | 9,838 m | at the median 79 s interval, 500 m already implies an impossible ~12 kn jump |
| **Reporting gap** (`gap`) | deletes consecutive reports until the silence reaches the target | 500 – 1,700 s | 1,093 s | lower bound just above the 99th percentile of natural gaps (491 s); upper bound below the 1,800 s voyage-split threshold, or the gap would split the track and vanish |
| **Transponder switch-off** (`dark`) | deletes a long stretch so the voyage splits in two; the second half starts after a long silence | 3,600 – 21,600 s | 13,108 s | must exceed 1,800 s so it registers as a voyage break |
| **Gradual drift** (`drift`) | slides positions away along a random bearing, growing as `(i/n)^1.5`; speed, course and heading untouched | 100 – 3,000 m | 1,300 m | sub-km to a few km keeps the track plausible, which is the point of sophisticated spoofing |

Each attacked track's severity is drawn **uniformly** from its range. The
"median applied" column is measured from the final dataset **[data/dataset_*.csv,
column `severity`]**. Figure 17 shows each attack on one real ship's track.

### 6.3 How many tracks were attacked

**[scripts/PIPELINE.py, `inject()`; logs/pipeline_v2.log]**

1. 5,900 tracks with at least 10 reports were eligible (`N_TRACKS = 5,900`).
2. 45% were chosen for attack (`ANOMALY_FRAC = 0.45`): 5,900 × 0.45 = 2,655.
   The 45% was **derived from what the experiment needs**: the largest scarce-data
   training set uses 500 anomalies across 5 seeds, so about 2,500 anomalous
   training windows were required.
3. The 2,655 were split across attacks **inversely to how many windows each
   attack corrupts**. Speed falsification corrupts hundreds of reports per track;
   a position jump corrupts three. An even split gave a 128:1 imbalance between
   attack types, so each attack got a weight: speed_lie 1, drift 2.5, jump 12,
   gap 16, dark 22 (sum 53.5). Quota = floor(2,655 × weight / 53.5):

| Attack | Weight | Tracks attacked | Anomalous windows produced |
|---|---:|---:|---:|
| speed_lie | 1.0 | 49 | 588 |
| drift | 2.5 | 124 | 1,332 |
| jump | 12.0 | 595 | 714 |
| gap | 16.0 | 794 | 677 |
| dark | 22.0 | 1,091 | 681 |
| **total** | | **2,653** | **3,992** |

The window counts end within 2.3:1 of each other, against 128:1 before weighting.

### 6.4 Labelling

Each attack returns a **mask** marking exactly which reports it changed. A window
is labelled anomalous (1) only if it **contains at least one changed report**.
An earlier version labelled every window of an attacked track as anomalous, and
85–90% of those "anomalous" windows contained no anomaly at all.

### 6.5 Final clean-up

77 normal windows whose `spd_mismatch_max` exceeded 60 kn (`MAX_IMPLIED_KN`,
faster than any vessel) were removed as corrupt GPS data. Anomalous windows were
kept even if extreme, because extreme is what an attack looks like.

**Result: 114,969 windows, 3,992 anomalous (3.47%).**

---

## 7. Splitting by vessel

**[scripts/PIPELINE.py, `split_and_scale()`; logs/pipeline_v2.log]**

Every window from a given ship goes into the same split, so a model is always
tested on ships it has never seen. Without this, a model can memorise individual
ships and every score is inflated.

Vessels were grouped by their dominant attack type and how many anomalous
windows they have (0, 1–2, 3–8, 9+), then each group was shuffled and split 25%
test, 15% validation, 60% train. This keeps the attack mix similar across splits.

| Split | Windows | Vessels | Anomalous | Rate | Used for |
|---|---:|---:|---:|---:|---|
| Train | 68,296 | 1,462 | 2,420 | 3.54% | fitting models |
| Validation | 17,381 | 364 | 537 | 3.09% | choosing settings |
| Test | 29,292 | 609 | 1,035 | 3.53% | the final, reported scores |
| **Total** | **114,969** | **2,435** | **3,992** | **3.47%** | |

Vessel overlap between every pair of splits: **0** (checked by the pipeline).
2,435 rather than 2,441 vessels because 6 vessels produced no complete window.

---

## 8. Checking the dataset

| Check | Result | Source |
|---|---|---|
| 42 property assertions (ranges, label consistency, no NaNs, split disjointness) | 42/42 pass | `scripts/07_validate_dataset.py` |
| 22 features recomputed on a separate code path (different distance formula, explicit loops, bearing written out by hand) | 22/22 agree | `scripts/09_independent_check.py` |
| 9 full pipeline rebuilds, each of three judgement thresholds at three values | under-way threshold (0.3/0.5/1.0 kn): immaterial, max ΔAUC 0.048. Window length (5/10/20): a real trade-off, max spread 0.078; short windows favour drift, long ones point events. Voyage split: **must be ≥ 1,800 s** (below it the gap attack collapses, spread 0.230); above it, insensitive (ΔAUC 0.004 from 1,800 to 3,600 s). Run on the original 7 features. | `scripts/08_sensitivity_sweep.py`, `docs/SENSITIVITY.md` |
| Determinism: 60 runs repeated through two independent scripts | identical to 0.00e+00 | `docs/COMPLETE_RESULTS.md` §0 |

---

## 9. The 13 models

### 9.1 Three hybrid quantum neural networks (HQNNs)

**[scripts/16_sweep.py, `_circuit()`, `q_train()`, `q_params()`; figure 16]**

All three share this structure:

1. **Classical pre-layer, Dense 12 → 8:** `angles = π × sigmoid(x · W + b)`.
   It compresses the 12 features into 8 angles between 0 and π.
   Parameters: 12 × 8 weights + 8 biases = **104**.
   This layer is **forced**: one feature goes on one qubit, 12 features would
   need 12 qubits, and a 12-qubit simulation costs about 16× more per evaluation
   (4,096 amplitudes against 256), which would turn a 9-hour sweep into a 6-day
   one. It is a handicap for the quantum side: its classical twin reads all 12
   features directly.
2. **Angle embedding:** each of the 8 angles becomes an RX rotation on its own
   qubit.
3. **Strongly entangling layers:** each layer applies a general rotation
   `Rot(φ, θ, ω)` (3 trainable angles) to each of the 8 qubits, then a ring of
   CNOT gates that entangles them. 2 layers × 8 qubits × 3 angles = **48 angles**.
4. **Measurement:** the expectation value ⟨Z⟩ of each qubit, giving 8 numbers.
5. **Readout** to one score, then a sigmoid gives a probability.

The three differ only in steps 3 and 5:

| Model | Circuit | Readout | Trainable parameters T |
|---|---|---|---:|
| **Classical pre-layer** (`Q_pre`) | embed once → 2 entangling layers | `a × Σ⟨Z⟩ + b` (2 params) | 104 + 48 + 2 = **154** |
| **Data re-uploading** (`Q_reup`) | embed → layer 1 → embed again → layer 2 | linear, 8 weights + 1 bias (9 params) | 104 + 48 + 9 = **161** |
| **Frozen circuit** (`Q_frozen`) | same as pre-layer, but the 48 angles are fixed at their random starting values | `a × Σ⟨Z⟩ + b` (2 params) | 104 + 0 + 2 = **106** |

The **frozen circuit is a control**. It is identical to the pre-layer model
except that the quantum angles never train, so comparing the two isolates what
training the quantum part contributes.

Data re-uploading follows Pérez-Salinas et al. (2020): re-encoding the input
between layers lets even small circuits represent more complex functions.

### 9.2 Three matched classical twins

**[scripts/16_sweep.py, `CARCH`, `_mlp_params()`]**

Each HQNN has a classical neural network (multi-layer perceptron, MLP) twin with
the **same depth** (a two-stage model, so two hidden layers) and **at least as
many parameters**:

| Twin of | Hidden layers | Parameter count | vs quantum |
|---|---|---|---:|
| Pre-layer | 8, 5 | (12×8+8) + (8×5+5) + (5×1+1) = 104 + 45 + 6 = **155** | 155 ≥ 154 |
| Re-uploading | 8, 6 | 104 + (8×6+6) + (6×1+1) = 104 + 54 + 7 = **165** | 165 ≥ 161 |
| Frozen | 6, 4 | (12×6+6) + (6×4+4) + (4×1+1) = 78 + 28 + 5 = **111** | 111 ≥ 106 |

**Why depth matters, not just parameter count:** an early two-seed screen matched
parameters only and showed a quantum "win" that disappeared once depth was
matched too **[docs/QNN_RESULTS.md §3]**.

### 9.3 The quantum kernel (QSVM)

**[scripts/20_qsvm.py]**

- **Input:** the 12 features compressed to 8 with PCA, fitted on each training
  subsample only, rescaled to [0, π]. (PCA because a kernel has no trainable
  layer to learn its compression.)
- **Feature map:** `IQPEmbedding` (Havlíček et al. 2019, *Nature*), one repetition.
- **Kernel value** between windows x and x′: `|⟨φ(x)|φ(x′)⟩|²`, computed by
  encoding x, then un-encoding x′, and reading the probability of returning to
  the all-zero state.
- **Classifier:** a classical SVM on the precomputed kernel. Its regularisation
  constant C ∈ {1, 10} was chosen on a 20% holdout carved from the training
  subsample.
- **Cost:** one circuit evaluation per *pair* of windows, so cost grows with N².
  Measured at about 0.5–0.9 ms per pair. A 68,296-window kernel needs
  68,296² × 8 bytes = **37.3 GB** of memory, against 17 GB on the laptop.

### 9.4 Six classical baselines

**[scripts/10_classical_baselines.py; results/baseline_selection.json]**

| Model | Final settings | Configurations tried |
|---|---|---:|
| Gradient boosting (`HistGradientBoostingClassifier`) | 100 iterations, depth 3, learning rate 0.05 | 12 |
| Random forest | 300 trees, depth 6 | 9 |
| Logistic regression | C = 0.01, balanced class weights; 13 parameters | 10 |
| MLP, 12-5-1 | 71 parameters, L2 1e-4, learning rate 0.01, positives oversampled to 20% | 12 |
| Isolation forest (no labels) | 500 trees, 1,024 samples per tree, contamination 0.02 | 12 |
| Single-threshold rule (no labels) | score = `spd_mismatch_max`; nothing is trained | 12 (one per feature) |

Each was tuned by **validation AUC**. *Declare this asymmetry in the paper:* the
baselines had 9–12 configurations each, while the HQNNs and their twins had 2.
That favours the classical side, which is the safe direction for the finding that
quantum never beats gradient boosting, but it must be stated.

The 12-5-1 MLP was mislabelled "unconstrained" in early drafts. It has 71
parameters, fewer than any HQNN.

---

## 10. How the models were trained

### 10.1 Training-set sizes

**[scripts/16_sweep.py, `SWEEP_N`, `subsample()`]**

The five sizes were set by the number of **labelled anomalies** they contain, at
a 3.5% rate: N = anomalies ÷ 0.035.

| N (windows) | Calculation | Labelled anomalies (at the train rate 3.543%) |
|---:|---|---:|
| 571 | 20 ÷ 0.035 | 20 |
| 1,429 | 50 ÷ 0.035 | 51 |
| 2,857 | 100 ÷ 0.035 | 101 |
| 14,286 | 500 ÷ 0.035 | 506 |
| 68,296 | the full training split | 2,420 |

68,296 ÷ 571 = 120, so the sweep spans a 120-fold range.

**Subsampling.** For seed `s`, a random generator seeded with `1000 + s` draws
`round(N × 0.03543)` anomalous and the rest normal windows from the training
split. The anomaly rate is held fixed, and **every model sees exactly the same
windows** at the same N and seed. That is what makes the comparisons paired.

### 10.2 Phase 1: choosing settings

On one 14,286-window subsample, each neural model tried **2 configurations ×
3 seeds**, chosen by mean validation AUC **[results/sweep_best_cfg.json]**:

| Models | Configurations | Chosen |
|---|---|---|
| HQNNs | Adam learning rate 0.02 or 0.05; weight decay 1e-4 | re-uploading 0.05; pre-layer 0.02 |
| Classical twins | learning rate 0.003 or 0.01; L2 1e-4 | 0.01 for all three |
| Frozen circuit | not tuned separately; inherits the pre-layer's settings | 0.02 |

5 models × 2 configs × 3 seeds = **30 runs**.

### 10.3 Phase 2: the fixed gradient-step budget

Every neural model, quantum and classical, gets **exactly 640 gradient steps** at
every training-set size.

**Why steps and not epochs.** An epoch is one pass through the data. With a fixed
number of epochs, a model trained on 571 windows makes about 120× fewer weight
updates than one trained on 68,296, and the comparison would measure
under-training, not data scarcity.

**Where 640 comes from.** The batch size is min(4,096, N). At full data,
68,296 ÷ 4,096 = 16 full batches per epoch, and 40 epochs × 16 = **640 steps**.
At N = 571 the batch is all 571 windows, so 640 steps is 640 epochs.

**The rest of the recipe:**
- loss: binary cross-entropy + weight decay 1e-4 × (sum of squared weights)
- optimiser: Adam
- starting values: rotation angles ~ Normal(0, 0.1); pre-layer weights ~ Normal(0, 0.3)
- no class weighting and no oversampling for the HQNNs or twins
- the model after step 640 is the one evaluated (no early stopping)

Classical twins use scikit-learn's `MLPClassifier` with the same batch size and
`max_iter` set so the step count matches exactly.

---

## 11. How performance was measured

**[scripts/19_final.py, `record()`; scripts/22_full_sweep.py]**

| Metric | What it measures | How it is computed |
|---|---|---|
| **ROC-AUC** | Probability that a random anomalous window scores above a random normal one. 0.5 = guessing, 1.0 = perfect. | `roc_auc_score` |
| **PR-AUC** | Area under the precision–recall curve; weights the top of the ranking, which is what an analyst actually reviews. Baseline = anomaly rate (~0.035). | `average_precision_score` |
| **Precision at 3.5% alert budget** | Of the windows flagged, the share that are truly anomalous. | flag the top k = round(29,292 × 0.035) = **1,025** test windows; precision = true positives ÷ 1,025 |
| **Recall at 3.5% alert budget** | Of all anomalies, the share caught. | true positives ÷ 1,035 |
| **Per-attack AUC** | How well one attack type is detected. | ROC-AUC on the normal windows plus only that attack's windows |

**Accuracy is never reported.** At 3.47% anomalies, a model that flags nothing
scores 96.5% accuracy.

**Which split for what.** Settings were chosen on validation. The test split was
used only for reported scores, never for any choice. Several experiments scored
it (baselines, endpoint runs, full sweep, quantum-features experiment); none of
them chose anything based on it.

---

## 12. Statistics

### 12.1 Seeds and pairing

Every comparison is **paired**: quantum and twin trained on identical windows
with the same seed. Differences are computed seed by seed.

### 12.2 Wilcoxon signed-rank test

`scipy.stats.wilcoxon`, two-sided, on the paired differences. It asks whether the
differences are consistently positive or negative, without assuming they are
normally distributed.

**Why 10 seeds at the endpoints.** With n pairs, the smallest p-value the test
can ever return is 2 ÷ 2ⁿ:
- n = 5: 2 ÷ 32 = **0.0625**, so p < 0.05 is impossible however large the effect.
- n = 10: 2 ÷ 1,024 = **0.002**.

So the headline comparisons were re-run at 10 seeds **[scripts/19_final.py,
results/final_results.csv]**. The five-size sweep uses 5 seeds and is reported
as effect sizes and consistency, not p-values.

### 12.3 Crossover point

Where the quantum − twin difference Δ changes sign between two measured sizes,
the crossover is found by linear interpolation in log N:

```
t  = −Δᵢ / (Δᵢ₊₁ − Δᵢ)
N* = exp( ln Nᵢ + t × (ln Nᵢ₊₁ − ln Nᵢ) )
```

**Worked example, frozen circuit:** Δ = −0.0295 at 1,429 and +0.0086 at 2,857.
t = 0.0295 ÷ 0.0381 = 0.774. ln 1,429 = 7.265, ln 2,857 = 7.958, difference 0.693.
N* = exp(7.265 + 0.774 × 0.693) = exp(7.801) = **2,443 windows**, which is
2,443 × 0.03543 ≈ **87 labelled anomalies**.

This is interpolation **between** measured points, never extrapolation beyond
them.

### 12.4 Standard error bands on figures

Standard deviation across seeds ÷ √(number of seeds).

---

## 13. Every experiment, in order

| When | Experiment | Model fits | Script → result file |
|---|---|---:|---|
| Sep 2026, after the rebuild | Classical baselines at full data, tuned on validation, scored on test | 22 | `10_classical_baselines.py` → `baseline_results.csv` |
| Aug–Sep 2026 | Architecture screen and controls (8-feature stage; superseded) | — | `13_architectures.py`, `14_controls.py` |
| 2026-09-07 | Dataset rebuilt with 12 features | — | `PIPELINE.py` |
| finished 2026-09-08 | **Scarcity sweep**: Phase 1 settings + Phase 2, 8 models × 5 sizes × 5 seeds, validation (669 min) | 230 | `16_sweep.py` → `sweep_results.csv` |
| Sep 2026 | **Endpoint runs**: 6 neural models × 2 sizes × 10 seeds, validation + test | 120 | `19_final.py` → `final_results.csv` |
| Sep 2026 | Baselines at every size, 5 seeds, validation | 100 | `21_baseline_sweep.py` → `baseline_sweep_results.csv` |
| to 2026-09-15 | **Quantum kernel** at 571 and 1,429, 5 seeds each | 10 | `20_qsvm.py` → `qsvm_results.csv` |
| finished 2026-09-15 | **Full sweep**: 12 models × 5 sizes × 5 seeds, validation **and** test (15.2 h). 60 cells reused from the endpoint runs, identical to 0.00e+00. | 240 new | `22_full_sweep.py` → `full_sweep_results.csv` |
| 2026-09-18 | **Follow-up 1**: quantum features in tree models (43 min) | 500 | `26_quantum_features_rf.py` → `qfeat_results.csv` |
| 2026-09-19 | **Follow-up 2**: geometric difference (125 min) | — | `25_geometric_difference.py` → `geometric_difference.csv` |
| | **Total** | **1,222** | |

**Not run.** The quantum kernel at 2,857 windows: about 30 hours for 5 seeds,
with the trend already flat (0.758 → 0.763). At 14,286 and 68,296 it is not
computable on this laptop (≈10 days; 37 GB of memory).

**Hardware.** One Windows laptop: 22 CPU threads, 17 GB RAM. PennyLane 0.45.1,
JAX 0.10.2, scikit-learn 1.9.0, Python 3.13.

---

## 14. Results: SQ1, the size-matched comparison

### 14.1 At the endpoints (test set, 10 paired seeds)

**[results/final_results.csv; docs/COMPLETE_RESULTS.md §6]**

| Pair | Δ at N = 571 | p | Δ at N = 68,296 | p |
|---|---:|---:|---:|---:|
| Pre-layer − twin | −0.0582 | **0.027** | +0.0198 | **0.010** |
| Frozen − twin | −0.0813 | **0.020** | +0.0395 | **0.002** |
| Re-uploading − twin | +0.0164 | 0.557 | +0.0087 | 0.160 |

Δ = quantum test AUC − twin test AUC. Two of three pairs flip sign **and** are
significant at both ends.

### 14.2 Across all 54 comparisons

3 pairs × 2 sizes × 9 metrics (ROC-AUC, PR-AUC, precision, recall, 5 per-attack
AUCs) = 54 paired tests. Significant results: **19 quantum wins and 0 losses at
68,296; 0 wins and 6 losses at 571** (figure 12).

### 14.3 Across all five sizes (test, 5 seeds)

**[results/full_sweep_results.csv; docs/COMPLETE_RESULTS.md §5]**

| Pair | 571 | 1,429 | 2,857 | 14,286 | 68,296 | Crossover |
|---|---:|---:|---:|---:|---:|---|
| Frozen | −0.112 | −0.030 | +0.009 | +0.020 | +0.054 | 2,443 (~87 anomalies) |
| Pre-layer | −0.061 | −0.034 | −0.013 | +0.010 | +0.013 | 7,229 (~256) |
| Re-uploading | +0.001 | −0.034 | −0.016 | +0.006 | +0.001 | 9,432 (~334), weak |

### 14.4 The frozen-circuit control

Frozen minus trained (pre-layer): −0.019, −0.013, −0.010, −0.000, +0.004 as N
rises. **None is significant** (5 seeds: every p ≥ 0.625; 10 seeds on test:
+0.004 at 571, p = 1.000). Training the 48 quantum angles made no measurable
difference at any size. The lower-T model got none of the small-data benefit the
bound's popular reading predicts.

### 14.5 What it means

Caro et al.'s result is an **upper bound**, so nothing here contradicts the
theorem. What fails is the inference drawn from it. With little data, the
binding constraint is **optimisation**, not generalisation: quantum circuits are
harder to train from few examples.

---

## 15. Results: SQ2, against the best classical models

**[docs/COMPLETE_RESULTS.md §1, §3]**

| N | Best quantum (test AUC) | Best classical (test AUC) | Gap |
|---:|---|---|---:|
| 571 | Re-uploading 0.6525 | Random forest 0.7715 | −0.119 |
| 1,429 | Re-uploading 0.7156 | Random forest 0.7876 | −0.072 |
| 2,857 | Re-uploading 0.7517 | Random forest 0.8011 | −0.049 |
| 14,286 | Frozen 0.8007 | Gradient boosting 0.8202 | −0.020 |
| 68,296 | Frozen 0.8056 | Gradient boosting 0.8316 | −0.026 |

At full data with 10 seeds: gradient boosting 0.8316, random forest 0.8225, best
quantum 0.8074. Against logistic regression (0.8016) and the 71-parameter MLP
(0.8041) the quantum models are **level**: the margins (0.003–0.006) are smaller
than their own seed-to-seed spread (0.010–0.014). **So no parameter-efficiency
claim is possible.**

**Simple beats neural with little data** (validation, 5 seeds): at 571 windows the
best non-neural model (isolation forest, 0.8349) leads the best neural model
(frozen twin, 0.6811) by **0.154**, and that gap is the same for quantum and
classical networks. Isolation forest, using no labels at all, ties logistic
regression trained on 20 labelled anomalies (0.8349 vs 0.8348).

---

## 16. Results: SQ3, quantum kernel vs quantum circuits

**[results/qsvm_results.csv; validation, 5 seeds]**

| N | Quantum kernel | Best quantum circuit | Best neural model | Best non-neural |
|---:|---:|---:|---:|---:|
| 571 | **0.758** | 0.624 | 0.681 | 0.835 |
| 1,429 | **0.763** | 0.740 | 0.803 | 0.855 |

At 571 the kernel beats **all seven** neural networks. It barely improves with
more data (+0.005), against +0.203 for the pre-layer circuit; by 1,429, two
classical networks (the 71-parameter MLP and the re-uploading twin) overtake it.

**Interpretation.** The kernel solves a convex problem (an SVM); the circuits
are trained by gradient descent over a non-convex landscape. Since the kernel
does not collapse, the scarce-data failure belongs to **training variational
circuits**, not to quantum feature maps as such.

---

## 17. Results: SQ4, the ceiling, and quantum features

### 17.1 Geometric difference (follow-up 2)

**[scripts/25_geometric_difference.py; results/geometric_difference.csv]**

Huang et al. (2021, *Nature Communications*) define a number, the geometric
difference g, that says how far a quantum kernel `K_Q` can sit from a classical
kernel `K_C` on a given dataset. A classical kernel can learn any function the
quantum kernel can, needing at most g² times the model complexity. g ranges from
1 to √N; small g means no quantum kernel advantage is possible.

**How it was computed.**
1. Rebuild the QSVM's exact quantum kernel on the same subsamples as before
   (5 seeds × 2 sizes).
2. Build 8 classical kernels on the same 8 inputs: Gaussian (RBF) at 7 widths
   spanning 1,000× around the median-distance heuristic, plus linear.
3. Normalise each kernel so its trace equals N.
4. Compute the regularised form (with ridge parameter λ):
   `g = √( largest eigenvalue of  √K_Q · √K_C · (K_C + λI)⁻² · √K_C · √K_Q )`.
5. Also compute a training-error term; a classical kernel only counts if it can
   actually fit the data (`g_tra ≤ 0.1`). Without this check, a kernel too smooth
   to fit anything would wrongly look close to the quantum one.
6. Report the smallest g over the classical kernels that fit.

| N | g (median of 5 seeds) | Largest possible (√N) | Share |
|---:|---:|---:|---:|
| 571 | **2.9** | 23.9 | 12% |
| 1,429 | **3.8** | 37.8 | 10% |

So the room for a quantum-kernel advantage is about a tenth of the maximum, and
shrinks as a share as data grows.

**On the real labels.** Model complexity `s = yᵀ(K+λI)⁻¹K(K+λI)⁻¹y` measures how
simply a kernel describes the actual anomaly labels. At 571 windows (λ = 0.01)
the quantum kernel is simpler than every fitting classical kernel on **5 of 5
seeds**. At 1,429 (λ = 0.001) on **0 of 5**. The one theoretical point in
quantum's favour sits at the smallest size, consistent with the QSVM result.

### 17.2 Quantum features inside tree models (follow-up 1)

**[scripts/26_quantum_features_rf.py; results/qfeat_results.csv]**

**Question.** Do features computed by a quantum circuit improve the models
industry actually uses?

**Quantum features.** Each window's 8 PCA inputs were IQP-embedded (the QSVM's
circuit), and the single-qubit expectation values ⟨X⟩, ⟨Y⟩, ⟨Z⟩ of each qubit
were read out: 24 numbers. All 8 ⟨Z⟩ values are exactly 0 for one IQP repetition
(its entangling part is diagonal), so **16 non-constant features** remained.
Verified against PennyLane's own calculation to 5.6×10⁻¹⁶.

**Five versions ("arms") of each model**, identical settings and subsamples:

| Arm | Inputs |
|---|---|
| base | the 12 features |
| +Q | 12 + 16 quantum features |
| +RFF | 12 + 16 random Fourier features `cos(Wz + b)` (classical control) |
| +TANH | 12 + 16 random `tanh(Wz + b)` features (classical control) |
| +PCA | 12 + the 8 circuit inputs themselves (rules out "the rotation alone helped") |

5 sizes × 10 seeds × 5 arms × 2 models (random forest, gradient boosting) =
**500 fits**.

**Result.** Quantum features **lowered** test AUC in all 10 model-and-size
combinations: significantly for random forest at every size (−0.010 to −0.024,
p ≤ 0.004). They were **never better than the best classical control** (they beat
the random-Fourier control once: gradient boosting at 571, p = 0.037).

**The validation/test reversal.** On validation, random forest with quantum
features beat the plain model and every classical control at large sizes
(+0.006 to +0.010, p ≤ 0.049). On test, the same comparison reversed (−0.0096,
p = 0.002). Validation and test contain different ships. **Had only validation
been checked, this would have been reported as a quantum advantage.** This is
the strongest argument in the study for keeping a separate test set.

---

## 18. Where quantum did win (state these precisely)

All against the size-matched twin, full data, test set, 10 paired seeds:

| Where | Quantum | Twin | Gain | p |
|---|---:|---:|---:|---:|
| Transponder switch-off, pre-layer | 0.819 | 0.746 | **+0.073** | 0.002 |
| PR-AUC, frozen circuit | 0.382 | 0.296 | **+0.086** | 0.004 |
| Precision and recall at the alert budget, frozen | 0.362 / 0.358 | 0.312 / 0.309 | +0.050 each | 0.002 |
| Speed falsification, re-uploading | **0.998** | 0.995 | +0.003 | 0.014 |

Speed falsification at 0.998 is the **highest single score of any model in the
study**, above gradient boosting's 0.994. But every model is near-perfect on that
attack, so the margin is small. On transponder switch-off, the tree models still
lead at 0.961.

**Gradual drift is unsolved by everyone**: 0.61–0.66 test AUC for every model,
including a zero-parameter threshold rule, and it is 37.7% of test anomalies.

---

## 19. What went wrong and was corrected

Corrections are recorded in place, with the original wording, not silently
deleted. You can include this list as an appendix; it shows the rest is
trustworthy.

| Date | What was claimed or done | What was wrong | Fix |
|---|---|---|---|
| early | Every window of an attacked track labelled anomalous | 85–90% of "anomalous" windows had no anomaly | per-report mask; window anomalous only if it contains one |
| early | Speed lie added a fixed number of knots | pinned 30% of windows at a bound; taught "slow = anomalous" | multiplier chosen from available headroom |
| early | Differences computed before splitting voyages | implied speeds of 258,249 kn | split first |
| early | Twins matched on parameters only | two-seed quantum "win" that vanished | matched on depth too |
| Sep | Fixed number of epochs | small-N models got 120× fewer updates | fixed 640-step budget |
| Sep | Five scripts silently used the old 8-feature list | caught by identical 4-decimal results | fixed |
| Sep | Significance claimed at 5 seeds | test cannot reach p < 0.05 at 5 pairs | re-run at 10 |
| 09-14 | "Below ~100 labelled anomalies an unsupervised method beats every supervised model" | logistic regression had not been run at small N; it ties | withdrawn |
| 09-14 | Extrapolated values in a results table, one predicting AUC > 1.0 | not measured | all extrapolation removed |
| 09-14 | Freezing the quantum angles "hurts" at small N, "monotone and backwards" | none of it significant | "no measurable difference" |
| 09-16 | HQNNs beat the MLP and logistic regression, ranked 3rd–5th | margins smaller than seed noise | "level, not ahead" |
| 09-17 | The 12-5-1 MLP called "unconstrained" | it has 71 parameters, fewer than any HQNN | relabelled; no parameter-efficiency claim |
| 10-02 | Figure 17 labelled the speed attack ×3.4 | ×2.6 applied (capped at 39 kn) | labels measured from the data |
| 10-02 | Figure 19 said all three pairs invert | re-uploading is +0.001 at both ends | title fixed |
| 10-02 | "Three neural networks overtake the kernel at 1,429" | true on 3 seeds; two on all 5 | fixed |
| 10-02 | README: 1,050 training runs | result files record 1,222 fits | fixed |
| 10-02 | Gap attack range printed as 300–1,700 s | code uses 500–1,700 s | fixed |
| 10-05 | Quantum features "behave like classical random features" | they beat one control once | "never better than the best classical control" |

---

## 20. Limitations to declare

1. **8 qubits, one feature map, noiseless simulation.** No hardware, no noise.
2. **The forced 12 → 8 compression.** The twin reads all 12 features.
3. **No parameter-efficiency or speed-up claim.** Neither was measured; a smaller
   classical network matches the quantum models.
4. **Injected, not observed, anomalies.** Real attacks may differ.
5. **One day of US coastal AIS.** No seasonal or geographic claim.
6. **Tuning asymmetry.** 2 configurations for the neural models, 9–12 for the
   classical baselines (favours classical).
7. **Twin of the frozen circuit is unstable** (seed standard deviation 0.059).
8. **Test set scored by several experiments**; no choice was ever made on it.
9. **Median imputation** computed over all windows before splitting (no labels).
10. **QSVM not run at 2,857 windows**; not computable above that.

---

## 21. Where the work stands

**Done:** every experiment; every result file complete
(`docs/COMPLETE_RESULTS.md`: "All cells complete"); 21 figures; the paper in
LaTeX (IEEE two-column) with 15 figures, 7 auto-generated tables and 36
references, compiling to PDF with 0 errors; everything on GitHub.

**Left:** writing the prose. Suggested order: Method → Results → Discussion →
Introduction → Conclusion → Abstract.

**Where to find things:**

| You need | Look in |
|---|---|
| A specific number | `docs/COMPLETE_RESULTS.md` |
| What the paper may and may not claim | `docs/RESEARCH_QUESTION.md` |
| What goes in each section | `docs/PAPER_OUTLINE.md`, and the `% WRITE` notes in `paper/main.tex` |
| Feature definitions (Appendix A) | `docs/METHODS.md` Part 6 |
| Attack parameters (Appendix B) | `docs/INJECTION_AND_TIMELINE.md` |
| Why a decision was made | `docs/DECISIONS.md` |
| Verified references | `docs/CITATIONS*.md`, `paper/refs.bib` |

---

## 22. The right verbs for each step

Use past tense, active or passive consistently, and the precise verb:

| Step | Write | Avoid |
|---|---|---|
| Getting the data | *obtained / retrieved* AIS data from MarineCadastre | "got", "downloaded" |
| Removing reports | *filtered / excluded* reports with SOG < 0.5 kn | "deleted", "cleaned out" |
| Making voyages | *segmented* tracks at gaps exceeding 1,800 s | "broke up" |
| Making windows | *partitioned* each track into non-overlapping 10-report windows | "chopped" |
| Making features | *computed / derived / engineered* 12 features per window | "made", "created" |
| Fake anomalies | *injected / synthesised* five attack types | "added fake data" |
| Labels | *labelled* a window anomalous if it *contained* a modified report | |
| Train/val/test | *partitioned the data by vessel* into…; *held out* a test set | "split up" |
| Scaling | *min-max scaled* to [0, π] using training statistics | |
| Model training | *trained / fitted / optimised* with Adam for 640 gradient steps | "ran the model" |
| Choosing settings | *selected* hyperparameters by validation AUC | "picked" |
| Matching | *matched* each HQNN to a classical twin of equal depth | |
| Encoding | *encoded* each feature as a qubit rotation | "put in" |
| Scoring | *evaluated* on the held-out test set | "tested how good" |
| Comparing | *compared* paired differences with a two-sided Wilcoxon signed-rank test | |
| Results | quantum models *outperformed / underperformed / matched* their twins | "beat", "lost to" (fine in speech, too casual in a paper) |
| Uncertain results | the difference *was not significant* (p = 0.557) | "quantum was slightly better" |
| Causes | the deficit *is attributable to* / *is consistent with* optimisation difficulty | "is because of" (too strong unless proven) |
| Limits | the study *did not measure / does not establish* | |
