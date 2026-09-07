# Everything that happened, from Orange to now

Written so you can follow, verify, and defend every step.

---

# PART 1 — The Orange phase

## What we did

Loaded `AIS_2023_01_01.csv` — one day of US coastal AIS, **8,156,509 position
reports**, 17 columns, 15,932 vessels.

**Edit Domain widget** — fixed the variable types Orange guessed wrong:
- `VesselType`, `Status`, `Cargo` → Categorical (they're codes, not quantities;
  type 80 isn't four times type 20)
- `BaseDateTime` → Time
- Left `MMSI` as Numeric

**Select Columns widget** — set roles:
- `MMSI`, `BaseDateTime`, `VesselName` → **Metas** (carried along, never fed to
  a model). MMSI is an ID; as a number a model would learn nonsense from its
  magnitude.
- `IMO`, `CallSign`, `Cargo` → **Ignored** (47%, 12%, 28% missing respectively)

**Select Rows widget** — four filters → **6,094,912 rows**:
- `TransceiverClass = A`
- `SOG < 40`
- `LAT` between −90/90, `LON` between −180/180

## What Orange told us

**Correlations widget** found:
- `Length : Width` = **0.886** → same information twice
- `Draft : Width` = 0.675, `Draft : Length` = 0.660 → all three measure "how big"
- `Heading : Width` = −0.343 → **a fake correlation**. Heading is `511` when
  unavailable, and small vessels report 511 more often. So "small ship → huge
  number" produced a spurious result. This proved 511 must be treated as
  missing, not as a value.
- `MMSI : Width` = 0.268 → MMSI's first digits encode flag state, so this is
  "ships from certain countries are bigger". Confirmed MMSI belongs in Metas.

**Scatter plot** (LON × LAT) drew the US coastline from vessel positions —
confirming the data was what we thought.

## Why we left Orange

Orange transforms rows **independently**. It has no widget for:
- grouping reports by vessel
- sorting them by time
- comparing each report to the previous one

Every feature we need is a *change between consecutive reports*. That is
sequence work, and it has to be Python.

---

# PART 2 — Setting up Python

Your machine defaulted to **Python 3.14**, which is too new — pandas and scipy
have no prebuilt wheels for it, so pip built broken installs and PennyLane
couldn't import.

Found **Python 3.13** also installed. Built an isolated virtual environment on
it (`.venv`) and installed numpy, pandas, scipy, scikit-learn, matplotlib,
PennyLane 0.45.1. Verified with a real 4-qubit circuit using `AngleEmbedding` +
`StronglyEntanglingLayers` — the same architecture Naaman et al. used for ADS-B.

---

# PART 3 — Building the features

**Script: `01_build_windows.py`**

## Filtering (repeats Orange, plus two more)

| Step | Rows | Vessels |
|---|---:|---:|
| raw | 8,156,509 | 15,932 |
| Class A only | 6,111,077 | 8,428 |
| SOG < 40 | 6,094,912 | 8,415 |
| valid coordinates | 6,094,912 | 8,415 |
| **under way (SOG ≥ 0.5)** | **1,463,926** | 5,150 |
| vessels with ≥200 reports | 1,256,434 | 2,441 |

The moving filter removed **4.6 million** rows — 68% of all AIS reports come
from vessels moored at berths. Median speed before: **0.00 kn**. After: 7.50.

## Splitting voyages

Reports are grouped by vessel, sorted by time, and split into separate
**tracks** wherever the vessel went quiet for more than 30 minutes. A ship that
sails, ties up for two hours, then sails again is two voyages, not one.

Result: **7,935 tracks**, of which **5,928** were long enough to use.

## 🐛 BUG 1 — found and fixed

The first version computed the deltas **before** splitting tracks. So the first
report of a voyage was compared against the last report of the *previous*
voyage — hours later, hundreds of km away.

Symptoms in the output:
```
spd_mismatch_max  max = 258,249 knots
dist_total_m      max = 2,225 km inside a 10-report window
dt_max            max = 72,284 s despite a 30-min split rule
```

All physically impossible, all caused purely by the order of two operations.
Fixed by splitting first, then computing deltas within each track. Same features
now top out at 55 kn and 73 km.

## Windows

A quantum circuit needs a **fixed-size input**, so every training example must
be the same length. We take **10 consecutive reports** and summarise them into
one row.

Stride is also 10, so windows don't overlap — overlapping windows share data,
and if two overlapping windows landed on opposite sides of a train/test split
that's leakage.

**122,371 windows** from 5,928 tracks.

## 13 candidate features engineered

Speed statistics, turn rates, time gaps, distances, and three *consistency
checks* (implied vs reported speed; bearing vs course; course vs heading).

---

# PART 4 — Cutting 13 features to 6

**Script: `02_explore_plots.py`**

Ran Spearman correlation across all 122,287 windows. Seven pairs above |0.8|,
forming five clusters. Kept one from each:

| Kept | Dropped | ρ |
|---|---|---|
| `sog_mean` | `dist_total_m` | 0.966 |
| `sog_max_jump` | `sog_std` | 0.921 |
| `cog_turn_max` | `cog_turn_mean`, `brg_mismatch_mean` | 0.977, 0.929 |
| `dt_max` | `dt_mean` | 0.838 |
| `spd_mismatch_max` | `spd_mismatch_mean` | 0.940 |
| `crab_mean` | — | — |

**Why cut at all:** 6 qubits means 6 features. Two features correlating at 0.95
carry the same information, so encoding both wastes a qubit.

**One finding worth noting:** `brg_mismatch_mean` was *designed* as a spoofing
detector. It correlated 0.95 with turning — because the bearing between two
points naturally lags the reported course *during a turn*. It was measuring
manoeuvring, not deception. Dropped on evidence, not opinion.

---

# PART 5 — Baseline figures

**Script: `04_baseline_viz.py`** — eight figures on clean, unlabelled data.

The important finding, from the box plots:

**Normal vessels are internally consistent to within 0.32 knots.** The median
`spd_mismatch_max` is 0.32, and three-quarters of all windows stay under 0.63.

That tightness is why the study can work at all. A signal is only detectable if
the background is quiet, and this proved the background is very quiet.

PCA also showed **PC1 means "manoeuvring"** — every irregularity measure loads
positive, steady speed loads negative. So the main variation in normal shipping
is harbour tugs versus open-water transits, not anomalies.

---

# PART 6 — Injecting anomalies

## Why inject at all

Real AIS anomalies can't be verified without port records or satellite imagery.
No ground truth means no accuracy, precision or recall — and therefore no
quantum-vs-classical comparison, which is the entire study. Naaman et al. made
the same choice for ADS-B.

## Five attack types

| Attack | What it does |
|---|---|
| `drift` | positions slowly diverge from truth — sophisticated spoofing |
| `jump` | single instantaneous displacement |
| `gap` | delete reports — brief silence inside a voyage |
| `speed_lie` | positions untouched, reported SOG falsified |
| `dark` | delete a long stretch — transponder switched off for hours |

Types 1–4 come from the industry taxonomy (Windward, MarineTraffic). Type 5 was
added because your own Govia ruleset ranks going dark as **Critical**, and the
pipeline structurally couldn't see it.

## 🐛 BUGS 2–4 — found and fixed

**Bug 2: the gap attack was erasing itself.** Injected gaps were 900–5400 s, but
the track-splitting rule breaks a voyage at 1800 s. Any gap over 30 minutes
became a *track boundary* and vanished from the window entirely.
→ Fixed by keeping injected gaps under 1700 s, and adding `dark` as a separate
*track-level* attack with its own feature (`prev_silence_s`).

**Bug 3: the normal class was contaminated.** The impossible-value filter was
applied in stage 1 but not reapplied after re-windowing. Duplicate-MMSI
teleports sat in the *normal* class, inflating its mismatch from 0.81 to 7.78 —
blurring the exact contrast we were measuring.
→ Fixed by reapplying the filter to normal windows only.

**Bug 4: the labels were wrong — the serious one.** An anomaly was injected into
a *track*, and then **every window from that track** was labelled anomalous. A
track produces ~20 windows; the anomaly touches one to three.
**So 85–90% of "anomalous" windows contained no anomaly.**
→ Fixed by tracking exactly which rows each attack modified, and labelling a
window anomalous only if it *contains* a modified row.

---

# PART 7 — The finished dataset

**Script: `05_build_dataset.py`**

Six things done in one pass:

1. **Window-level labelling** (bug 4 fix)
2. **Fifth attack type** — `dark`
3. **Log-transform** the skewed features. Angle encoding maps each feature to a
   rotation between 0° and 180°; if 99% of values crowd into the first 2°, that
   qubit carries almost no information.
4. **`crab_mean` imputation** — 19% missing due to the 511 sentinel. Filled with
   median, plus a `crab_valid` flag so the model knows which values were invented.
5. **Scale to [0, π]** — computed on **training data only**, so no test
   information leaks into training.
6. **Train/test split BY VESSEL** — no ship appears in both sides. Split
   randomly by row and the model would memorise individual vessels instead of
   learning behaviour.

## 🐛 BUG 5 — found and fixed

Window-level labelling produced a **128:1 imbalance** between attack types:
speed_lie 2,049 windows, dark 16.

Cause: mask width. `speed_lie` corrupts everything from a point to the end of a
track (hundreds of rows → dozens of windows). `jump` corrupts one discontinuity
(three rows → one window). Equal tracks, wildly unequal yields.

Also `dark` almost never fired — it picked a random start point and often ran
out of track before accumulating enough silence.

→ Fixed by allocating tracks **inversely to measured yield**, and rewriting the
dark attack to guarantee room. Imbalance now **1.8:1**, dark fires 671 times.

## 🐛 BUG 6 — found, NOT yet fixed

`speed_lie` reads `sog_mean` = 3.84 against 7.71 for normal windows.

The attack adds or subtracts up to 15 knots then clips at 0.1. On a 7-knot
vessel a negative offset bottoms out near zero while a positive one just moves
up — the effect is asymmetric and drags reported speed down.

**A model could learn "slow = anomalous", which would be my bug, not a finding.**
Fixable by making the offset multiplicative or rejecting offsets that clip.

---

# Current state

```
8,156,509  raw AIS position reports
1,256,434  after filtering
    5,928  voyage tracks
  115,095  windows  (4,110 anomalous, 3.6%)
```

| File | Rows | Vessels | Anomalous |
|---|---:|---:|---|
| `dataset_train.csv` | 81,706 | 1,705 | 2,820 (3.5%) |
| `dataset_test.csv` | 33,389 | 730 | 1,290 (3.9%) |
| vessel overlap | **0** | | |

Anomalies by type: drift 1,198 · speed_lie 858 · jump 705 · gap 678 · dark 671

Eight features, scaled to [0, π], ready to encode.

---

# Outstanding before results mean anything

1. **Fix bug 6** — the speed_lie asymmetry
2. **Add a validation set** — currently only train/test. Tuning the classical
   baseline against the test set would contaminate it.
3. **Sensitivity checks** on three thresholds I chose with no external basis:
   the 0.5 kn moving cut, the 30-minute track split, the 10-report window.

Then: classical baseline → quantum model → the label-scarcity experiment.
