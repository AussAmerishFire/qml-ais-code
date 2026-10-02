# THE DOCUMENT — everything about this study

> **Status (2026-10-02).** Narrative written while the study was running; parts describe the **8-feature** stage and results that were later corrected. For current numbers use `COMPLETE_RESULTS.md`; for claims, `RESEARCH_QUESTION.md`.

The single reference. Every term defined, every number derived, every figure
explained. Written for the person writing the paper.

**Verified 2026-08-27**, after the float64 rebuild. All numbers current.

---

# CONTENTS

1. The study in plain words
2. **Glossary — every term used anywhere**
3. The data
4. Preprocessing, and why each threshold has the value it has
5. The features
6. Anomaly injection
7. Splitting the data
8. **Every figure — what it shows and which ones go in the paper**
9. How the data was verified
10. Current state
11. Every bug found, and how
12. What remains
13. File index

---

# 1. THE STUDY IN PLAIN WORDS

**The problem.** Ships broadcast their position, speed and course by radio.
Criminals falsify those broadcasts — to hide sanctions-busting cargo transfers,
illegal fishing, smuggling. We want to detect falsified broadcasts.

**The research question.** Quantum machine learning models have far fewer
adjustable settings than ordinary neural networks. Theory proves they can
therefore learn from very few examples. Maritime anomaly detection has very few
examples — confirmed cases number in the handful. **Does that theoretical
advantage appear in practice?**

**The experiment.** Take real vessel movements. Deliberately corrupt some of
them in five documented ways. Train a quantum model and a classical model to
tell them apart. Then shrink the number of labelled examples they get — 500,
200, 100, 50, 20 — and see whether the quantum model degrades more slowly.

**Why deliberate corruption.** Real anomalies cannot be verified without port
records or satellite imagery. Without knowing the right answer you cannot
measure accuracy, and without accuracy you cannot compare two models.

---

# 2. GLOSSARY

## 2.1 Statistics

**Median** — the middle value. Sort everything; the median is the one in the
centre. Half the data is above, half below. Unlike the average, it isn't
dragged around by a few extreme values.

**Percentile (p90, p99, p99.9)** — sort every value from smallest to largest.
The **99th percentile (p99)** is the value below which 99% of the data falls.

> Example: our reporting gaps have p99 = 491 seconds. That means 99 out of every
> 100 gaps between AIS messages are shorter than 491 seconds, and only 1 in 100
> is longer.

Used constantly here to answer *"is this value unusual?"* — anything above p99
is in the rarest 1% of normal behaviour.

**IQR (interquartile range)** — the range covering the middle 50% of the data,
from the 25th to the 75th percentile. A measure of spread that ignores outliers.

**Correlation (ρ, rho)** — how strongly two things move together, from −1 to +1.
+1 means when one rises the other always rises. 0 means no relationship. −1
means they move oppositely.

> We used it to find features measuring the same thing twice — `Length` and
> `Width` correlated at 0.886, so encoding both wastes a qubit.

**Spearman correlation** — a version that measures whether two things move
together *in rank order*, rather than in a straight line. Robust to skewed data,
which ours is.

**Skew** — when most values are small and a few are enormous. Our features are
heavily right-skewed: thousands of tiny values, a long tail of large ones.

**Log transform** — taking the logarithm to compress large values and spread out
small ones. Turns a skewed distribution into a usable one.

**AUC (Area Under the ROC Curve)** — the headline metric here.

> **0.5 = useless**, the feature tells you nothing (a coin flip).
> **1.0 = perfect**, the feature separates the two classes completely.
> **Below 0.5** means it separates *in reverse*.

AUC is used instead of accuracy because our classes are imbalanced: only 3.5% of
windows are anomalous, so a model that says "normal" every time scores 96.5%
accuracy while being completely useless. AUC is immune to that.

## 2.2 Ships and AIS

**AIS (Automatic Identification System)** — the radio system ships use to
broadcast identity, position, speed and course. Roughly every 2–10 seconds for
large vessels.

**MMSI** — a vessel's 9-digit radio ID number. The first three digits encode the
country of registration.

**Knot** — one nautical mile per hour, about 1.85 km/h.

**SOG (Speed Over Ground)** — how fast the ship is actually moving across the
seabed, in knots. Includes being pushed by currents.

**COG (Course Over Ground)** — the compass direction the ship is *actually
travelling*.

**Heading** — the compass direction the ship's *bow points*.

**Crab / crab angle** — the difference between COG and Heading.

> A ship in a strong side-current points its bow slightly upstream while
> actually travelling diagonally — like walking across a moving walkway, where
> you face forward but travel sideways. Sailors call this **crabbing**.
>
> It happens legitimately all the time. But fabricated data often forgets to
> keep heading consistent with course, so a large or erratic crab angle can
> indicate falsification. Our feature `crab_mean` is the average crab angle
> across a window.

**Class A / Class B transponder** — Class A is mandatory for large commercial
vessels and reports every 2–10 seconds. Class B is for smaller craft and reports
every 30 seconds to 3 minutes. **We use only Class A**, because mixing reporting
rates would make our time-gap feature meaningless.

**Sentinel value** — a number that means "no data". AIS uses:
- **102.3** for unknown speed
- **511** for unknown heading
- **360** for unknown course

These are not measurements. Treating them as numbers corrupts everything — and
it did, before we caught it.

**Draft (draught)** — how deep the hull sits below the waterline. Changes with
cargo load.

**Going dark** — a vessel switching off its transponder so it cannot be tracked.
The classic signature of sanctions evasion and illegal fishing.

**Spoofing** — broadcasting false position data while actually being elsewhere.

**STS (ship-to-ship) transfer** — two vessels meeting at sea to transfer cargo,
often to disguise its origin.

## 2.3 Machine learning

**Feature** — one number describing one example. We have 8.

**Label** — the right answer. 0 = normal, 1 = contains an injected anomaly.

**Window** — our unit of analysis. Ten consecutive AIS reports from one vessel,
summarised into a single row of 8 features. **One window = one training example.**

**Track** — one continuous voyage segment. A vessel that sails, moors for two
hours, then sails again produces two tracks.

**Train / validation / test** — three separate slices of data.
- **Train** teaches the model
- **Validation** tunes its settings
- **Test** measures final performance and is touched exactly once

**Leakage** — when information from the test data reaches the model during
training. Produces excellent scores that mean nothing. The classic version here
would be putting the same ship in both train and test, letting the model
memorise that vessel instead of learning behaviour.

**Overfitting** — when a model memorises the training data instead of learning
the pattern. Performs beautifully on data it has seen and badly on anything new.

**Class imbalance** — when one class is far rarer than the other. Ours is 3.5%
anomalous. This is why accuracy is the wrong metric.

**Baseline** — the ordinary (classical) model you must beat for a result to mean
anything. It must be **properly tuned**, or the comparison is rigged.

**Ablation** — removing a component to see whether it mattered. We will ablate
`crab_mean` and `sog_mean` to test whether they earn their qubits.

## 2.4 Dimensionality reduction — how we look at 8-dimensional data

We have 8 features, so every window is a point in **8-dimensional space**. You
cannot draw that. These methods squash it down to 2 dimensions so it can be
plotted.

**PCA (Principal Component Analysis)** — finds the directions along which the
data spreads out most, and lets you plot the top two.

> The analogy: photographing a 3D object. Some camera angles show its shape
> clearly, others hide it. PCA finds the most informative angle.

**PC1, PC2, PC3 (Principal Component 1, 2, 3)** — those directions, in order.
**PC1** is the direction of greatest variation in the data, PC2 the next, and so
on. They are new axes built from combinations of your original features.

> In our data, **PC1 turned out to mean "how erratically is this vessel
> behaving"** — every irregularity measure (turning, accelerating,
> inconsistency) contributes positively to it, while steady speed contributes
> negatively. So the biggest source of variation in normal shipping is harbour
> tugs manoeuvring versus container ships crossing open water.

**Loading** — how much each original feature contributes to a principal
component. `cog_turn_max` has a loading of +0.519 on PC1, meaning turning is the
single biggest contributor to that axis.

**Explained variance** — what percentage of the total variation a component
captures. If PC1 and PC2 together explain 70%, then two numbers describe most of
what makes your windows differ.

**Scree plot** — the bar chart showing explained variance per component, with a
running total line. It answers *"how many dimensions does my data really have?"*
The name comes from geology — a scree slope is the pile of loose rock at the
base of a cliff, and the plot has the same shape: a steep drop then a flat tail.

**t-SNE (t-distributed Stochastic Neighbour Embedding)** — another way to squash
many dimensions into two, but **nonlinear**.

> PCA can only rotate and flatten, like casting a shadow. t-SNE looks at which
> points are near each other in the original 8D space, then arranges them on a
> flat page so those neighbours stay neighbours. It can unfold curved structure
> that PCA would flatten into mush.
>
> **One rule:** distances *between* clusters in t-SNE are meaningless. Only
> "these points are neighbours" is real. That is why our t-SNE plots have no
> axis numbers — they would imply a precision that does not exist.

**Hexbin** — a density plot. Instead of drawing millions of overlapping dots, it
divides the plot into hexagons and colours each by how many points fall inside.
Used in our PCA figures because 115,000 dots would be a solid blob.

## 2.5 Quantum

**Qubit** — the quantum version of a bit. Our circuit uses one per feature.

**Angle encoding** — turning a number into a rotation angle so a quantum circuit
can accept it. Each feature is scaled to somewhere between 0 and π (0° to 180°)
and becomes a rotation.

> This is why the log transform matters. If 99% of a feature's values crowd into
> a narrow band, they all become nearly the same angle, and that qubit carries
> almost no information.

**Variational quantum circuit / VQC** — a quantum circuit with adjustable
rotation angles that get tuned by an optimiser, exactly like weights in a neural
network. The trainable part.

**Hybrid model** — classical neural network layers wrapped around a quantum
circuit. Necessary because quantum operations are reversible rotations and
cannot do ReLU-style nonlinearity; the classical layers supply it.

**NISQ (Noisy Intermediate-Scale Quantum)** — today's quantum computers.
Hundreds to low thousands of qubits, too error-prone for error correction, so
circuits must stay small and shallow. **We run on a simulator, not real
hardware** — so our results model the algorithm, not the machine.

---

# 3. THE DATA

**NOAA MarineCadastre**, file `data/raw/AIS_2023_01_01.csv` — one day of US coastal
vessel traffic. **8,156,509 reports, 15,932 vessels, 17 columns.**

**Why 2023 and not newer — MEASURED.** Every year was probed on NOAA's server:
2018–2023 download; 2024 is listed in the index but every file returns 404; 2025
exists only as aggregated density grids, not tracks; 2026 has no index. **2023
is the most recent raw AIS that publicly exists.**

**Why this source.** US federal data is public domain, so the processed subset
can be published with the paper. Many commercial AIS sources forbid that.

**The 17 columns:** MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading,
VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft, Cargo,
TransceiverClass.

---

# 4. PREPROCESSING — every threshold, derived

## Filter 1 — Class A only
Removes 2,045,432 reports.
**Derived:** Class A reports every 2–10 s, Class B every 30 s–3 min. One of our
features is the largest gap between reports. A three-minute silence is alarming
for Class A and routine for Class B — mixing them makes the feature
uninterpretable.
*This is scoping, not cleaning. State it as a scope limitation.*

## Filter 2 — SOG < 40 knots
Removes 16,165 reports.
**Derived from the AIS standard:** 102.3 is the "speed unavailable" sentinel.
40 knots also exceeds any commercial vessel's service speed.

## Filter 3 — valid coordinates
Removes **0** rows. NOAA's data is clean; the filter is defensive.

## Filter 4 — under way (SOG ≥ 0.5 kn)
Removes **4,630,986 reports — 76% of what reaches it.**

**Derived from measurement.** What speed do vessels report when they *declare*
themselves stationary?

| Declared status | Reports | median | p90 | p95 |
|---|---:|---:|---:|---:|
| moored | 801,384 | 0.00 | 0.10 | 2.60 |
| at anchor | 272,197 | 0.00 | **0.20** | **0.40** |

**0.5 kn sits above the 95th percentile of both.** The 0.1–0.4 readings are GPS
jitter at a berth, not motion.

Effect: median speed before the filter **0.00 kn**, after **7.50 kn**.

## Filter 5 — vessels with ≥200 reports
Leaves **1,256,434 reports from 2,441 vessels.**
**Reasoned:** at 10 reports per window, 200 gives ~20 windows per vessel — enough
that no single ship's quirks dominate.

## Track splitting — 30 minutes
**Derived from measurement.** Observed gaps between consecutive reports:

| median | p90 | p99 | p99.9 | above 1800 s |
|---:|---:|---:|---:|---:|
| 70 s | 85 s | 491 s | 1,409 s | **0.000%** |

**Not one legitimate gap in 1.25 million reports reaches 1800 seconds.** The
threshold sits entirely outside normal behaviour.

## Windows — 10 reports, non-overlapping
**Why windows:** a quantum circuit takes a fixed number of inputs, so every
example must be the same size.
**Why 10:** at the median 70-second interval, ten reports span ~12 minutes — long
enough to contain a manoeuvre, short enough that an anomaly is not diluted.
**Why non-overlapping:** overlapping windows share reports, and two overlapping
windows on opposite sides of a split would be leakage.

---

# 5. THE FEATURES

| # | Feature | Plain meaning | Normal median | Best at catching |
|---|---|---|---|---|
| 1 | `sog_mean` | average speed | 7.60 kn | context |
| 2 | `sog_max_jump` | biggest speed change between two reports | 0.20 kn | impossible acceleration |
| 3 | `cog_turn_max` | sharpest course change | 6.80° | impossible manoeuvres |
| 4 | `dt_max` | longest gap inside the window | 79 s | brief silences |
| 5 | **`spd_mismatch_max`** | **reported speed vs speed implied by positions** | **0.32 kn** | **spoofing** |
| 6 | `crab_mean` | bow direction vs travel direction | 2.76° | inconsistent fabrication |
| 7 | `prev_silence_s` | silence before this voyage began | 0 s | going dark |
| 8 | `crab_valid` | was heading real or imputed? | 1 | missing-data pattern |

## The design principle

**Four of the five anomalies are caught as internal contradictions, not as
deviations from a learned norm.**

A vessel broadcasts position *and* speed *and* course. Those must agree: the
distance between two consecutive positions, divided by the time elapsed, must
match the reported speed. A falsifier has to keep all of them consistent and
usually doesn't.

That is why `spd_mismatch_max` is strongest — it needs no idea what normal looks
like, only that the vessel contradicts itself.

**Measured baseline:** in unmodified data the median mismatch is 0.11 knots, p99
is 3.20, p99.9 is 11.27. Real vessels agree with themselves.

## How 13 became 8
Spearman correlation across 122,371 windows found seven pairs above |0.8| —
pairs measuring the same thing twice. One survivor kept from each cluster.

Notable casualty: `brg_mismatch_mean` was *designed* as a spoofing detector but
correlated 0.95 with turning, because bearing between two points naturally lags
reported course *during a turn*. It measured manoeuvring, not deception.

## Current discriminative power

| feature | distance from chance | verdict |
|---|---:|---|
| `spd_mismatch_max` | 0.235 | strong |
| `sog_max_jump` | 0.111 | useful |
| `dt_max` | 0.102 | useful |
| `cog_turn_max` | 0.097 | useful |
| `prev_silence_s` | 0.080 | useful — and the only detector of `dark` |
| `crab_mean` | 0.034 | weak, ablate |
| `sog_mean` | 0.011 | near chance, ablate |

---

# 6. ANOMALY INJECTION

## Which attacks — sourced
Four families from the industry taxonomy (Windward, MarineTraffic, SkyFi):
position falsification, implausible route, transponder switch-off, false data
broadcast. A fifth (`dark`) was added because it is the anomaly maritime
security ranks most critical and the pipeline could not otherwise see it.

## How big — derived from measured natural variation

Reference measurements from 1,256,434 unmodified reports:

| Quantity | median | p90 | p99 | p99.9 |
|---|---:|---:|---:|---:|
| distance between reports | 272 m | 551 m | 1,285 m | — |
| gap between reports | 70 s | 85 s | 491 s | 1,409 s |
| **window-max gap** | 79 s | 279 s | **1,318 s** | 1,740 s |
| speed/position mismatch | 0.11 kn | 0.49 kn | 3.20 kn | 11.27 kn |

| Attack | Range | Derivation |
|---|---|---|
| **drift** | 100–3,000 m | Lower bound is *below* the median single-step movement (272 m) — the vessel's own normal motion exceeds the entire injected drift. Undetectable by construction. |
| **jump** | 500–20,000 m | At the median 79 s interval: 500 m implies 12.3 kn (plausible, and near the p90 of normal movement); 20,000 m implies **492 kn** (impossible). Spans undetectable to obvious. |
| **gap** | 500–1,700 s | Must exceed natural variation and stay **below** the 1800 s split or the anomaly becomes a track boundary and ceases to exist. 31.8% land above the window-max p99; the rest are deliberately subtle. |
| **speed_lie** | ×1.8–5.0 | On a 7.7 kn vessel, ×1.8 creates a 6.2 kn mismatch (above natural p99 of 3.20); ×5.0 creates 30.8 kn (far above p99.9). Multiplier drawn from available headroom so no clipping. |
| **dark** | 3,600–21,600 s | Must **exceed** the 1800 s split to register as a track-level blackout. Upper bound 6 h is where a vessel could plausibly have left receiver coverage legitimately. |

## Allocation
Attacks produce wildly different window counts because their masks differ in
width — `speed_lie` corrupts hundreds of rows, `jump` corrupts three. Even
allocation gave **128:1** imbalance. Tracks are now allocated inversely to
measured yield → **1.8:1**.

---

# 7. SPLITTING

**Three-way, by vessel.** No ship appears in more than one side.

| Split | Windows | Vessels | Anomalous |
|---|---:|---:|---|
| train | 69,928 | 1,462 | 2,494 (3.6%) |
| validation | 17,104 | 365 | 446 (2.6%) |
| test | 27,940 | 608 | 1,111 (4.0%) |
| **total** | **114,972** | **2,435** | **4,051 (3.5%)** |

Overlap between any pair: **0**.

**Why by vessel:** ten consecutive reports from one ship, then the next ten
minutes later, are near-identical. Split randomly by row and those near-twins
land on opposite sides — the model recognises *the ship* rather than learning
behaviour.

**Why a validation set:** tuning against test contaminates the thing you are
measuring on.

**Scaling fitted on train only:** computing min/max across all data leaks test
information into training.

---

# 8. EVERY FIGURE

## Which ones go in the paper

**Essential — these are your results:**
- **fig12** — which feature detects which attack (your RQ3 table)
- **fig7** — dataset description (every empirical paper needs one)
- **fig9** — normal vs anomalous, showing the injection worked

**Useful — supporting or appendix:**
- **fig2** — justifies cutting 13 features to 8
- **fig10 / fig11** — visual evidence the task is hard
- **fig8** — explains what an anomaly looks like to a non-quantum reader

**Working figures — for you, not the paper:**
- **fig1, fig3, fig4, fig5, fig6**

## What each one shows

### fig1 — distributions of all 13 candidate features
Histograms: how often each value occurs. Some axes are log10 because the data is
so skewed a normal scale shows one spike and nothing else.
**Told us:** all 13 have real spread, and the heavy skew means we must
log-transform before angle encoding.

### fig2 — correlation matrix ⭐ decision figure
Every feature against every other, coloured −1 (blue) to +1 (red).
**Told us:** which features duplicate each other. This cut 13 features to 8.

### fig3 — box plots of the 6 core features (clean data)
Box = middle 50%, red line = median, whiskers = normal range, dots = outliers.
**Told us:** normal vessels are internally consistent to within 0.32 knots. That
tightness is what makes anomalies detectable at all.

### fig4 — PCA of clean data
Left and middle: density maps of PC1 vs PC2 and PC1 vs PC3. Right: scree plot.
**Told us:** normal behaviour forms one connected blob, and **PC1 means
"manoeuvring"** — the biggest variation in normal shipping is tugs versus
container ships, not anomalies.

### fig5 — t-SNE of clean data
Same 4,000 windows three times, coloured differently.
**Told us:** no natural clusters in normal data — a clean canvas.

### fig6 — pairwise matrix
Every feature against every other, 6×6. Diagonal shows each feature's histogram.
**Why it exists:** fig2 compressed each relationship to one number, and single
numbers hide L-shapes, curves and separated clumps. Needs zooming.

### fig7 — dataset context ⭐ paper figure
Three panels: geography, vessel types, speed profile.
**Two jobs.** It is your "describe the data" figure. And it validates the
preprocessing — the speed panel peaks at 7–8 knots instead of at zero, visual
proof the stationary filter worked.

### fig8 — eight real vessel tracks
Actual voyages plotted from window centroids, green dot at the start.
**Now:** sanity check that track reconstruction produced coherent voyages.
**Later:** the figure that shows a non-quantum reader what an anomaly looks like.

### fig9 — normal vs anomalous, all 7 features ⭐ paper figure
Side-by-side box plots with AUC in each title.
**Shows:** the injection worked, and by how much per feature.

### fig10 — PCA with anomalies
Three panels: normal only, anomalies in red, coloured by attack type.
**Shows:** only 13.3% of anomalies sit beyond the normal 99th percentile. **Most
anomalies live inside normal behaviour** — the task is genuinely hard, which is
good, because an easy task would not distinguish the two model classes.

### fig11 — t-SNE with anomalies
Same structure.
**Shows:** whether anomalies form their own cluster. Cheap preview of whether
the study can work, before training anything.

### fig12 — feature × attack AUC heatmap ⭐ your RQ3 result
**The most important figure.** Shows which feature detects which attack:

| detector | attack | AUC |
|---|---|---|
| `spd_mismatch_max` | speed_lie | **0.982** |
| `spd_mismatch_max` | jump | **0.932** |
| `dt_max` | gap | **0.915** |
| `prev_silence_s` | dark | **0.778** |
| *nothing* | **drift** | 0.583 |

**The structure is almost diagonal** — each attack has its own detector, and no
single feature catches everything. Drift is undetectable by any single feature,
which is exactly where a model that learns *combinations* has to earn its keep.

---

# 9. HOW THE DATA WAS VERIFIED

Two independent methods, because one is not enough.

**Method 1 — property assertions** (`scripts/07_validate_dataset.py`, 42 checks).
Asserts properties of the output: no NaN, no leakage, labels consistent with
attack types, values physically possible, every attack detectable by something.
**Weakness:** if the pipeline computed something wrongly but consistently, these
would still pass.

**Method 2 — independent recomputation** (`scripts/09_independent_check.py`, 20 checks).
Recomputes features **from the raw file using different code** — explicit Python
loops instead of pandas, a different distance formula — and compares. Then
verifies the injections did what they claim, and checks every realised severity
sits inside its derived bounds.

**Result: 42/42 and 20/20.** Worst disagreement between the two code paths:
**0.0035%**.

Method 2 found two real problems that Method 1 could not have caught.

---

# 10. CURRENT STATE

```
8,156,509  raw AIS reports        one day, US coastal waters
        ↓  five filters remove 84.6%
1,256,434  usable reports
        ↓  split at 30-min silences
    5,928  voyage tracks
        ↓  aggregate 10 reports → 1 window (NOT a filter — summarisation)
  114,972  windows
        ↓  3.5% carry an injected anomaly
    4,051  anomalous windows
```

Anomalies by type: drift 1,198 · speed_lie 797 · jump 705 · dark 681 · gap 670

Eight features, scaled to [0, π], ready to encode. Both validation suites clean.

---

# 11. EVERY BUG FOUND

| # | Bug | Symptom | How it was found |
|---|---|---|---|
| 1 | deltas computed before track splitting | implied speeds of 258,249 knots | impossible values in the output |
| 2 | `gap` attack erased by the split rule | gap windows had *shorter* silences than normal | per-attack table made no sense |
| 3 | sanity filter not reapplied | normal-class mismatch inflated 9× | normal class looked worse than anomalous |
| 4 | **whole-track labelling** | **85–90% of "anomalous" labels were wrong** | features barely separated |
| 5 | attack yield imbalance 128:1 | `dark` had 16 windows | attack counts |
| 6 | `speed_lie` clipping asymmetry | 30% of windows pinned at the speed ceiling | `sog_mean` separating in reverse |
| 7 | `gap` bound derived from wrong distribution | injected gaps overlapped normal variation | **independent check** |
| 8 | **float32 coordinates** | 5% distance error at short displacements | **independent check** |

Bug 4 was the serious one — it would have quietly poisoned every result.
Bugs 7 and 8 were invisible to the property checks and only surfaced when a
second, independent implementation disagreed.

---

# 12. WHAT REMAINS

1. **Ablate `crab_mean` (0.034) and `sog_mean` (0.011).** Near chance
   individually, but univariate AUC cannot see interactions — a feature can be
   useless alone and valuable in combination. Decide from model performance.

2. **Validation anomaly rate is 2.6% vs test 4.0%.** The vessel split is not
   stratified by label. Stratify if tuning proves sensitive to it.

3. **Adjacent windows share one boundary delta** — the first delta of a window
   reaches one report back. Not independent, though never across train/test.
   State it.

4. **Anomaly rate 3.5% exceeds reality** (~0.7% industry). Deliberate, so the
   smallest label budgets have enough positives. State it.

5. **Modelling has not started.** Classical baseline → quantum model → the
   label-scarcity experiment.

---

# 13. FILE INDEX

## Documents
| File | Contains |
|---|---|
| **THE_DOCUMENT.md** | this file — everything |
| `CITATIONS.md` | verified references with DOIs and BibTeX |
| `DECISIONS.md` | every choice, tagged by how well-founded |
| `FEATURES.md` | the 8 features in full detail |
| `SENSITIVITY.md` | threshold sweep results |
| `PLAN.md` | the 11 remaining steps |
| `WALKTHROUGH.md` | narrative history, Orange onward |

## Code, in order
| File | Does |
|---|---|
| `scripts/preprocessing_QMLAIS.py` | prints the filter waterfall |
| `scripts/01_build_windows.py` | filters, builds tracks, computes features, windows |
| `scripts/02_explore_plots.py` | fig1, fig2 |
| `scripts/04_baseline_viz.py` | fig3–fig8 |
| `scripts/05_build_dataset.py` | injection, labelling, scaling, splitting |
| `scripts/06_compare_viz.py` | fig9–fig12, AUC audit |
| `scripts/07_validate_dataset.py` | validation method 1 |
| `scripts/08_sensitivity_sweep.py` | threshold sweeps |
| `scripts/09_independent_check.py` | validation method 2 |

## Data
| File | Rows |
|---|---|
| `data/raw/AIS_2023_01_01.csv` | 8,156,509 raw reports |
| `data/windows_clean.csv` | 122,371 unlabelled windows |
| `data/windows_labelled.csv` | 114,972 windows, unscaled, with labels |
| `data/dataset_train.csv` | 69,928 scaled |
| `data/dataset_val.csv` | 17,104 scaled |
| `data/dataset_test.csv` | 27,940 scaled |

## Environment
Python 3.13 in `.venv` · PennyLane 0.45.1 · scikit-learn 1.9.0 · NumPy 2.5.2 ·
pandas 3.0.5 · Matplotlib 3.11.1
