# METHODS — everything, explained and justified

Written for the person writing the paper. Every number here is either
**MEASURED** from the data, **DERIVED** from a constraint, or **REASONED** from
how the physics works. Nothing is "because I said so."

---

## ⚑ CURRENT STATE — verified 2026-08-27, after the float64 rebuild

| | Windows | Vessels | Anomalous |
|---|---:|---:|---|
| train | 69,928 | 1,462 | 2,494 (3.6%) |
| validation | 17,104 | 365 | 446 (2.6%) |
| test | 27,940 | 608 | 1,111 (4.0%) |
| **total** | **114,972** | **2,435** | **4,051 (3.5%)** |

By attack: drift 1,198 · speed_lie 797 · jump 705 · dark 681 · gap 670

**Validation:** 42/42 property checks and 20/20 independent-recomputation checks
pass. Two separate code paths produce identical features (worst disagreement
0.0035%).

**Note:** validation anomaly rate is 2.6% versus 4.0% in test — the vessel-level
split is not stratified by label, so the rate varies. Acceptable but worth
stating; if it matters for tuning, stratify the vessel assignment.

---

# PART 0 — What we are doing, in plain words

**The problem.** Ships broadcast their position, speed and course over radio
(AIS). Criminals falsify these broadcasts — to hide sanctions-busting cargo
transfers, illegal fishing, or smuggling. We want to detect falsified broadcasts.

**The research question.** Quantum machine learning models have far fewer
tunable parameters than ordinary neural networks. Theory (Caro et al. 2021)
proves they can therefore learn from very few labelled examples. Maritime
anomaly detection has very few labelled examples — confirmed cases number in
the handful. **Does the theoretical advantage show up in practice?**

**The design.** Take real vessel movements. Deliberately corrupt some of them in
five documented ways. Train a quantum model and a classical model to tell the
difference. Then shrink the number of labelled examples they get — 500, 200,
100, 50, 20 — and see whether the quantum model degrades more slowly.

**Why deliberate corruption.** Real anomalies cannot be verified without port
records or satellite imagery. Without knowing the right answer, you cannot
compute accuracy, and without accuracy you cannot compare two models. Injecting
known anomalies is standard practice — Naaman et al. did the same for aircraft
data.

---

# PART 1 — Definitions

## AIS terms

| Term | Meaning |
|---|---|
| **AIS** | Automatic Identification System — the radio system ships use to broadcast their position, roughly every 2–10 seconds |
| **MMSI** | Maritime Mobile Service Identity — a vessel's 9-digit radio ID |
| **SOG** | Speed Over Ground — actual speed across the seabed, in knots |
| **COG** | Course Over Ground — the direction the ship is actually travelling |
| **Heading** | The direction the ship's bow points (differs from COG in a current) |
| **Class A / Class B** | Transponder types. Class A is mandatory for large commercial ships and reports every 2–10 s. Class B is for smaller craft and reports every 30 s–3 min |
| **Knot** | One nautical mile per hour ≈ 1.85 km/h |
| **Sentinel value** | A number meaning "no data". AIS uses `102.3` for unknown speed, `511` for unknown heading, `360` for unknown course |

## Our terms

| Term | Meaning |
|---|---|
| **Report** | One AIS broadcast from one vessel at one moment — one row of the raw file |
| **Track** | One continuous voyage segment. A vessel that sails, moors for two hours, then sails again produces two tracks |
| **Window** | 10 consecutive reports from one track, summarised into a single row. **This is one training example** |
| **Feature** | One number describing a window (e.g. "the largest speed change in this window") |
| **Label** | 0 = normal, 1 = contains an injected anomaly |
| **AUC** | Area Under the ROC Curve. 0.5 = the feature is useless, 1.0 = perfect separation. The standard metric when classes are imbalanced |

## Machine learning terms

| Term | Meaning |
|---|---|
| **Train / validation / test** | Three separate slices of data. Train teaches the model; validation tunes its settings; test measures final performance and is touched once |
| **Leakage** | When information from test data reaches the model during training, producing inflated scores that mean nothing |
| **Qubit** | The quantum equivalent of a bit. Our circuit has one per feature |
| **Angle encoding** | Turning a number into a rotation angle so a quantum circuit can accept it |

---

# PART 2 — The data source

**NOAA MarineCadastre**, file `data/raw/AIS_2023_01_01.csv` — one day of US coastal
vessel traffic. **8,156,509 reports, 15,932 vessels, 17 columns.**

**Why 2023 and not something newer** — MEASURED. I probed every year on NOAA's
server: 2018–2023 download successfully; 2024 is listed in the index but every
file returns 404; 2025 exists only as aggregated density grids, not vessel
tracks; 2026 has no index at all. **2023 is the most recent raw AIS that
publicly exists.**

**Why this source** — US federal data is public domain, so the processed subset
can be published alongside the paper. Several commercial AIS sources forbid
redistribution.

---

# PART 3 — Preprocessing, with derivations

## Filter 1 — Class A transceivers only

**What:** keep only vessels with Class A transponders. Removes 2,045,432 reports.

**Why — DERIVED.** One of our features is the largest gap between reports, used
to detect a vessel going silent. Class A reports every 2–10 seconds; Class B
every 30 seconds to 3 minutes. **A three-minute silence is alarming for Class A
and completely routine for Class B.** Mixing them makes the feature
uninterpretable.

**Honest note:** this is *scoping*, not cleaning. Class B vessels are
legitimate; we exclude them to keep one feature meaningful. State it as a scope
limitation in the paper.

## Filter 2 — `SOG < 40` knots

**What:** removes 16,165 reports.

**Why — REASONED from the AIS standard.** The AIS protocol encodes "speed
unavailable" as the literal value **102.3**. It is a sentinel, not a
measurement. 40 knots also exceeds the service speed of any commercial vessel,
so anything above is either that sentinel or corruption.

## Filter 3 — valid coordinates

**What:** latitude in [−90, 90], longitude in [−180, 180]. Removed **0 rows**.

**Why:** values outside those ranges are not coordinates. NOAA's data turns out
to be clean, so this filter is defensive rather than corrective — worth saying,
because it tells a reader the source is well-maintained.

## Filter 4 — under way only (`SOG ≥ 0.5` knots)

**What:** removes **4,630,986 reports — 76% of what reaches it.**

**Why — MEASURED.** I looked at what speed vessels report when they *declare*
themselves stationary:

| Declared status | Reports | median | p90 | p95 |
|---|---:|---:|---:|---:|
| moored | 801,384 | 0.00 | 0.10 | 2.60 |
| at anchor | 272,197 | 0.00 | **0.20** | **0.40** |

**0.5 kn sits above the 95th percentile of vessels at anchor and above the 90th
percentile of moored vessels.** The small non-zero readings (0.1–0.4 kn) are GPS
jitter at a berth, not motion. The threshold is set to exclude that jitter while
keeping anything genuinely under way.

Effect: median speed before the filter was **exactly 0.00 kn**; after, **7.50 kn**.

**Verified robust:** swept at 0.3 / 0.5 / 1.0 — detectability moved by 0.048 AUC.

## Filter 5 — vessels with ≥200 reports

**What:** removes 207,492 reports, leaving **1,256,434** from **2,441 vessels**.

**Why — REASONED.** At 10 reports per window, 200 reports yields ~20 windows per
vessel. Below that, a single vessel's idiosyncrasies could dominate a class.
2,441 vessels clear the bar — far more than the study needs — so the threshold
costs nothing.

---

# PART 4 — Building tracks

## Splitting voyages at 30 minutes of silence

**What:** group reports by vessel, sort by time, and start a new track wherever
the gap exceeds **1800 seconds**. Produces **7,935 tracks**, of which **5,928**
are long enough to use.

**Why — MEASURED.** The observed distribution of gaps between consecutive
reports for Class A vessels under way:

| median | p90 | p99 | p99.9 | above 1800 s |
|---:|---:|---:|---:|---:|
| 70 s | 85 s | 491 s | 1,409 s | **0.000%** |

**Not a single legitimate gap in 1.25 million reports reaches 1800 seconds.**
The threshold sits entirely outside the natural distribution — so anything
exceeding it is genuinely a break in the voyage, not normal transmission
variation.

## Order of operations — DERIVED, and it caused a bug

Tracks must be split **before** computing differences between consecutive
reports. Computing them first compares the first report of one voyage against
the last report of the previous one — hours later and hundreds of kilometres
away.

The first version got this wrong and produced implied speeds of **258,249
knots** and displacements of **2,225 km inside a 10-report window**. Both
physically impossible; both caused purely by the order of two operations.

---

# PART 5 — Windows

## 10 reports per window, non-overlapping

**Why windows at all — DERIVED.** A quantum circuit accepts a fixed number of
inputs. Every training example must therefore be the same size. Vessel tracks
vary from 200 to thousands of reports, so they are cut into fixed chunks.

**Why 10 — REASONED plus MEASURED.** At the observed median interval of 70
seconds, ten reports span about 12 minutes — long enough to contain a
manoeuvre, short enough that a brief anomaly is not diluted by surrounding
normal behaviour.

**Swept at 5 / 10 / 20**, and the result is a genuine trade-off:

| window | drift | jump | gap |
|---|---|---|---|
| 5 | **0.641** | 0.886 | 0.898 |
| 10 | 0.596 | 0.937 | 0.922 |
| 20 | 0.607 | **0.964** | **0.944** |

Short windows favour gradual drift; long windows favour point events. **10 was
kept because it sits in the middle and maximises nothing** — choosing 20 because
it detects jumps better would be tuning the data construction to flatter the
result.

**Why non-overlapping — DERIVED.** Overlapping windows share reports. If two
overlapping windows landed on opposite sides of a train/test split, the model
would be tested on data it partly trained on. Non-overlapping removes the
question entirely.

---

# PART 6 — The 8 features

Full definitions are in `FEATURES.md`. The design principle is what matters here:

**Four of the five anomalies are detected as internal contradictions, not as
deviations from a learned norm.**

A vessel broadcasts its position *and* its speed *and* its course. Those three
statements must agree with each other — the distance between two consecutive
positions, divided by the elapsed time, must match the reported speed. A
falsifier has to keep all of them consistent, and usually doesn't.

That is why `spd_mismatch_max` is the strongest feature: it needs no idea what
normal looks like. It only needs the vessel's own two statements to disagree.

**Measured baseline:** in unmodified data the median mismatch is **0.11 knots**,
the 99th percentile is **3.20**, the 99.9th is **11.27**. Real vessels agree
with themselves.

## Cutting 13 features to 8 — MEASURED

Thirteen were engineered, then Spearman correlation across 122,287 windows found
seven pairs above |0.8| — pairs measuring the same thing twice. One survivor was
kept from each cluster, because with 8 qubits, encoding two features that carry
identical information wastes a qubit.

One notable casualty: `brg_mismatch_mean` was *designed* as a spoofing detector,
but correlates 0.95 with turning — because the bearing between two points
naturally lags reported course *during a turn*. It was measuring manoeuvring,
not deception. Dropped on evidence.

---

# PART 7 — Anomaly injection, fully derived

## Which attacks — SOURCED

Four families come from the industry taxonomy (Windward, MarineTraffic, SkyFi):
position falsification, implausible route, transponder switch-off, false data
broadcast. A fifth (`dark`, the multi-hour blackout) was added because it is the
anomaly maritime security ranks as most critical and the pipeline could not
otherwise see it.

## How big — DERIVED from measured natural variation

Every severity range is now anchored to a measured property of real data.

### Reference measurements

| Quantity | median | p90 | p99 | p99.9 |
|---|---:|---:|---:|---:|
| distance moved between reports | 272 m | 551 m | 1,285 m | — |
| gap between reports | 70 s | 85 s | 491 s | 1,409 s |
| speed/position mismatch | 0.11 kn | 0.49 kn | 3.20 kn | 11.27 kn |

### `jump` — 500 to 20,000 m

**Derivation.** A jump is detected because it implies an impossible speed.
At the median 79-second interval:

| displacement | implied speed | verdict |
|---|---|---|
| 500 m | 12.3 kn | **plausible** — and 500 m is near the p90 of *normal* movement (551 m) |
| 1,000 m | 24.6 kn | plausible for a fast vessel |
| 20,000 m | **492 kn** | physically impossible |

**The range spans exactly from "indistinguishable from normal movement" to
"obviously impossible."** That is what makes difficulty vary rather than being
uniformly trivial.

### `drift` — 100 to 3,000 m cumulative

**Derivation.** Drift accumulates gradually across a whole track, so the
per-step displacement it adds is a small fraction of the total. The lower bound
of 100 m is **below the median single-step movement of 272 m** — the vessel's
own normal motion in one reporting interval is larger than the entire injected
drift. That is genuinely undetectable by design.

The upper bound of 3,000 m is roughly 10× a normal step, spread across ~100
reports, so per-step it remains within plausible motion. **Drift never produces
an impossible speed** — which is precisely why it is the hard case.

### `gap` — 500 to 1,700 s ⚠️ CORRECTED

**Derivation.** An injected silence must exceed natural transmission variation
to be a real signal, and must stay **below** the 1800 s track-split threshold or
the pipeline treats it as the end of a voyage and the anomaly disappears.

- **Upper bound 1,700 s** — just under the split threshold. DERIVED.
- **Lower bound: was 300 s, should be 500 s.** The 99th percentile of natural
  gaps is **491 s**, so 1.78% of legitimate gaps exceed 300 s. A 300-second
  injected gap is indistinguishable from normal transmission variation.
  **500 s sits just above the natural p99.**

**This is a real correction the derivation exposed** — the original lower bound
overlapped normal behaviour.

### `speed_lie` — ×1.8 to ×5.0

**Derivation.** The falsification must create a mismatch exceeding natural
variation. On a median 7.7-knot vessel:

| factor | falsified speed | mismatch created | vs natural |
|---|---|---|---|
| ×1.8 | 13.9 kn | 6.2 kn | above p99 (3.20), below p99.9 (11.27) |
| ×5.0 | 38.5 kn | 30.8 kn | far above p99.9 |

**The range spans from "just outside normal variation" to "far outside it."**
The multiplier is applied to whichever direction has headroom, so neither the
0.5 kn floor nor the 39 kn ceiling is hit — an earlier version clipped 30% of
cases against the ceiling and biased the whole feature.

### `dark` — 3,600 to 21,600 s

**Derivation.** Must **exceed** the 1800 s split threshold to register as a
track-level event (that is what makes it "going dark" rather than a brief
silence). The lower bound of one hour is twice the threshold, ensuring it always
splits. The upper bound of six hours is where a vessel could plausibly have left
the receiver's coverage legitimately rather than switched off.

## Attack allocation — MEASURED

Attacks produce wildly different numbers of windows because their masks differ
in width: `speed_lie` corrupts hundreds of rows, `jump` corrupts three.
Assigning tracks evenly produced a **128:1** imbalance. Tracks are now allocated
**inversely to measured yield**, giving **1.8:1**.

---

# PART 8 — Splitting the data

**Three-way, by vessel.** Train 60% / validation 15% / test 25% of *vessels* —
never of rows.

**Why by vessel — DERIVED.** Ten consecutive reports from one ship, then the
next ten minutes later, are nearly identical. Split randomly by row and those
near-twins land on opposite sides: the model recognises *the ship* rather than
learning what anomalous behaviour looks like. Accuracy looks excellent and means
nothing.

**Why a validation set — DERIVED.** Tuning a model's settings against the test
set contaminates the thing you are supposed to be measuring on. Validation
exists so test can stay untouched until the very end.

**Scaling fitted on train only — DERIVED.** Computing the min and max across all
data leaks information about test into training.

| Split | Windows | Vessels | Anomalous |
|---|---:|---:|---|
| train | 69,712 | 1,462 | 2,496 (3.6%) |
| validation | 16,511 | 365 | 528 (3.2%) |
| test | 28,884 | 608 | 1,065 (3.7%) |

Vessel overlap between any pair: **0**.

**Metric will not be accuracy — DERIVED.** At 3.6% anomalies, a model predicting
"normal" every time scores 96.4%. Report AUC, precision, recall and F1.

---

# PART 9 — Where the numbers came from

```
8,156,509  raw reports          one day of US coastal AIS
        ↓  five filters remove 84.6%
1,256,434  usable reports       Class A, under way, valid, from vessels with rich tracks
        ↓  split at 30-min silences
    5,928  voyage tracks
        ↓  aggregate 10 reports → 1 window
  115,107  windows              this is NOT a filter — it is summarisation
        ↓  3.6% carry an injected anomaly
    4,089  anomalous windows
```

---

# PART 10 — Corrections made, and what remains

## Fixed since the first build

| # | Problem | Resolution |
|---|---|---|
| 1 | Deltas computed before track splitting | reordered — produced 258,249 kn speeds |
| 2 | `gap` attack erased by the split rule | gaps confined below 1800 s; `dark` added for the track-level case |
| 3 | Sanity filter not reapplied after re-windowing | normal class was 9× contaminated |
| 4 | Whole-track labelling | window-level masks — 85–90% of labels had been wrong |
| 5 | 128:1 attack imbalance | tracks allocated inversely to measured yield → 1.8:1 |
| 6 | `speed_lie` clipping asymmetry | multiplier drawn from available headroom — ceiling pinning 30.5% → 1.1% |
| 7 | `gap` bound derived from the wrong distribution | compared against individual intervals (p99 491 s) when the feature is the *max of ten* (p99 1,318 s) |
| 8 | **float32 coordinates destroying precision** | **→ float64.** A latitude near 34.148670 exhausts float32's ~7 digits, so position *differences* lost most of their precision — 0.21% error at typical spacing, ~5% at short displacements. Silently degrading `spd_mismatch_max`, the strongest feature. |

Problems 7 and 8 were found only by the independent recomputation check, not by
the property assertions — which is why both exist.

## Resolved by those fixes

**The `sog_mean` reverse-bias has largely gone.** It was AUC 0.467 (0.033 from
chance); after the speed_lie and float64 corrections it reads **0.011 from
chance** — statistically indistinguishable from a coin flip. No longer worth
declaring as a limitation; it is simply a weak feature.

## Still outstanding

1. **Ablation on `crab_mean` (0.034) and `sog_mean` (0.011).** Both are near
   chance *individually*. They should be dropped only if a model performs no
   worse without them — univariate AUC cannot see interactions, and a feature
   can be useless alone yet valuable in combination. Decide at the modelling
   stage, not from the AUC table.

2. **Validation anomaly rate (2.6%) differs from test (4.0%).** The vessel split
   is not stratified by label. Acceptable, but if hyperparameter tuning proves
   sensitive to it, stratify the vessel assignment.

3. **Adjacent windows share one boundary delta.** The pipeline computes each
   row's delta against the previous row of the *track*, so a window's first
   delta reaches one report back. Windows are therefore not fully independent.
   Because splits are by vessel this never crosses train/test — but state it.

4. **Anomaly rate is higher than reality.** 3.5% here versus industry reports of
   ~0.7% of lost-transmission events being genuinely suspicious. Deliberate: a
   realistic rate would leave too few positives at the smallest label budgets.
   State as a limitation.
