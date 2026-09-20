# Decisions Log

Every choice made, why, and how well-founded it is.
Tags: **[DATA]** driven by measurement · **[DOMAIN]** AIS/maritime fact ·
**[DERIVED]** follows mathematically · **[JUDGMENT]** my call, no external
backing — challenge these

Last updated after the stage-5/6 rebuild (three-way split, five attack types,
window-level labelling, AUC audit).

---

## 1. Dataset

**NOAA MarineCadastre, `data/raw/AIS_2023_01_01.csv`** — **[DATA]**

Probed every year: 2018–2023 download; 2024 is listed in NOAA's index but every
file 404s; 2025 exists only as aggregated density grids, not tracks; 2026 has no
index. **2023 is the most recent raw AIS that exists publicly.**

Public domain (US federal), so the processed subset can ship with the paper.

---

## 2. Row filters

### `TransceiverClass == "A"` — **[DOMAIN]** + **[JUDGMENT]**

Class A reports every 2–10 s under way; Class B every 30 s–3 min. `dt_max` is a
feature and the gap-attack signal — mixed classes make it meaningless.

**This is scoping, not cleaning.** State it as a scope limitation.

### `SOG < 40` knots — **[DOMAIN]**
AIS encodes "speed unavailable" as **102.3**. Sentinel, not a measurement.

### `LAT ∈ [-90,90]`, `LON ∈ [-180,180]` — **[DERIVED]**
Removed 0 rows. NOAA's coordinates are clean; the filter is defensive.

### `SOG >= 0.5` ("under way") — **[JUDGMENT] ⚠️ OPEN**
Removed 4,630,986 rows — 76% of what reached it. Median speed before the filter
was **exactly 0.00 kn**; after, 7.50.

**Threshold is mine, no source.** 0.3 or 1.0 equally arguable. GPS jitter at a
berth registers ~0.1–0.3 kn, which is the loose reasoning — unverified.

**→ Needs a sensitivity sweep (0.3 / 0.5 / 1.0).**

---

## 3. Track construction

### Split at gaps > 1800 s (30 min) — **[JUDGMENT] ⚠️ OPEN**
Must exceed the normal Class A interval and fall short of a real blackout.
Anything 10–60 min defensible.

Weak corroboration: the author's own Govia Systems ruleset independently chose
>30 min for "going dark". **Team-decided, not researched — not citable**, but two
groups landing on the same scale is mild evidence it's natural.

**→ Needs a sensitivity sweep.**

### Deltas computed within (MMSI, track_id) — **[DERIVED]**
Not a preference. 🐛 **BUG 1**: the first version computed deltas before
splitting, comparing the first report of one voyage to the last of the previous —
producing `spd_mismatch_max = 258,249 kn` and 2,225 km inside a 10-report window.
Fixed by reordering. Same features now cap at 55 kn / 73 km.

### `MIN_REPORTS = 200` per vessel — **[JUDGMENT]**
Ensures ~20 windows per vessel so no single ship dominates. 2,441 vessels clear
it — far more than needed, so the threshold costs nothing.

### `MAX_IMPLIED_KN = 60` — **[DOMAIN]**
Above 60 kn is not a vessel. In practice these are duplicate MMSIs — two ships
sharing an identifier, whose interleaved positions look like teleportation.
Removed 0.1%.

🐛 **BUG 3**: this filter was applied at stage 1 but not reapplied after
re-windowing, so corrupt rows sat in the **normal** class and inflated its
mismatch from 0.81 to 7.78. Fixed — now reapplied to normal windows only
(anomalous windows legitimately produce extreme values).

---

## 4. Window shape

### `WINDOW = 10` reports — **[JUDGMENT] ⚠️ OPEN**
~2–15 min at Class A rates. Long enough to show behaviour, short enough that an
anomaly isn't diluted. **No source. → ablation at 5 / 10 / 20.**

### `STRIDE = 10` (non-overlapping) — **[JUDGMENT]**
Overlapping windows share reports; two overlapping windows on opposite sides of
a split is leakage that vessel-level splitting would not catch. Costs sample
count, and at 115,095 windows against a need of a few thousand that is free.

---

## 5. Feature selection

**13 candidates → 7, by correlation clustering** — **[DATA]**

Spearman across 122,287 windows found seven pairs above |0.8| in five clusters;
one survivor each.

| Kept | Dropped | ρ |
|---|---|---|
| `sog_mean` | `dist_total_m` | 0.966 |
| `sog_max_jump` | `sog_std` | 0.921 |
| `cog_turn_max` | `cog_turn_mean`, `brg_mismatch_mean` | 0.977, 0.929 |
| `dt_max` | `dt_mean` | 0.838 |
| `spd_mismatch_max` | `spd_mismatch_mean` | 0.940 |
| `crab_mean` | — | — |
| `prev_silence_s` | — | added later, see §6 |

Plus `crab_valid` (missingness flag) = **8 model inputs**.

**`brg_mismatch_mean` was designed as a spoofing detector** and dropped on
evidence: it correlates 0.95 with turning, because bearing between two points
lags reported course *during a turn*. It measured manoeuvring, not deception.

### Univariate AUC audit — **[DATA]**

| feature | overall | best at |
|---|---|---|
| `spd_mismatch_max` | **0.735** | speed_lie 0.982, jump 0.932 |
| `dt_max` | 0.598 | **gap 0.915** |
| `cog_turn_max` | 0.597 | gap 0.719 |
| `sog_max_jump` | 0.596 | gap 0.702 |
| `prev_silence_s` | 0.577 | **dark 0.778** |
| `crab_mean` | 0.536 | weak everywhere |
| `sog_mean` | **0.467** | separates in reverse |

**Nothing detects `drift`** — best single feature is 0.583. That is the hard case.

**`crab_mean` and `sog_mean` are near-chance individually but retained**, because
univariate AUC cannot see interactions and a nonlinear model may use them in
combination. Decide by ablation at the modelling stage, not from this table.

---

## 6. Anomaly injection

### Synthetic, not real anomalies — **[DOMAIN practice]**
Real AIS anomalies cannot be verified without port records or satellite imagery.
No ground truth → no accuracy/precision/recall → no quantum-vs-classical
comparison. Naaman et al. (2509.15991) made the same choice for ADS-B.

### Five attack types — **[industry taxonomy]** + **[JUDGMENT]**
Types 1–4 follow the Windward / MarineTraffic / SkyFi taxonomy. Type 5 (`dark`)
was added because the author's Govia ruleset ranks going dark as Critical and
the pipeline structurally could not see it.

**Severity ranges are NOT sourced.** Published magnitudes are vague ("hundreds
of nautical miles"). Mitigated by drawing severity from a range and recording it
per window, so detection can be reported as a function of subtlety.

### 🐛 BUG 2 — gap attack erased itself
Injected gaps of 900–5400 s exceeded the 1800 s track-split rule, so they became
track boundaries and vanished. Fixed: gaps now 300–1700 s (inside a track), and
`dark` handles the multi-hour case at track level.

**Scope limitation:** this design detects short-to-medium silences within a
voyage. Multi-hour blackouts are a track-level problem, handled separately via
`prev_silence_s`.

### 🐛 BUG 4 — whole-track labelling (the serious one)
An anomaly was injected into a track, then **every window of that track** was
labelled anomalous. Tracks yield ~20 windows; an anomaly touches 1–3.
**85–90% of "anomalous" windows contained no anomaly.**

Fixed: each attack now returns a mask of exactly which rows it modified, and a
window is anomalous only if it *contains* one.

### 🐛 BUG 5 — 128:1 attack imbalance
Window-level labelling exposed wildly unequal yields: `speed_lie` corrupts
hundreds of rows (dozens of windows), `jump` corrupts three (one window).
`dark` fired only 16 times because it picked a random start and ran out of track.

Fixed: tracks allocated **inversely to measured yield**, and `dark` rewritten to
guarantee room. Imbalance now **2.3:1**.

### 🐛 BUG 6 — speed_lie clipping asymmetry
Original: added/subtracted knots then clipped at 0.1 — a −15 kn offset on a
7-knot vessel bottomed out while +15 did not. Dragged anomalous `sog_mean` to
3.84 vs 7.71 normal. **A model could have learned "slow = anomalous."**

First fix (multiplicative) flipped the bias: 30.5% of windows pinned at the 39.9
ceiling, `sog_mean` 24.62.

Final fix: the multiplier is chosen from the **headroom actually available** in
each direction, so neither bound is hit. Ceiling pinning 30.5% → **1.1%**;
share above the normal median 75.1% → **56.2%**.

**⚠️ Residual:** `sog_mean` AUC 0.467 — still separates faintly in reverse.
Some shift is legitimate (a speed spoof *does* change reported speed); the
artefact was the clipping, which is fixed. Declare the residual as a limitation.

---

## 7. Study design

### Binary classification — **[DERIVED from the hypothesis]**
The hypothesis (Caro et al., 2111.05292) concerns generalisation from **few
labelled examples**, which presupposes labels → supervised.

### Model class: variational QNN, not quantum kernel — **[DERIVED]**
Caro's bound is stated for *"variationally optimizing a parameterized quantum
circuit"* with *"T trainable gates."* Quantum kernels have **no trainable gates**,
so the bound does not apply and a kernel would not test the hypothesis.
Secondary: kernels need N² circuit evaluations — 16M at 4,000 samples.

### Three-way split BY VESSEL — **[DERIVED]**
No ship in more than one side. Random row splitting would let the model memorise
individual vessels; windows from one voyage are near-duplicates.

Validation set exists so hyperparameters can be tuned without touching test.

| | windows | vessels | anomalous |
|---|---:|---:|---|
| train | 69,712 | 1,462 | 2,496 (3.6%) |
| val | 16,511 | 365 | 528 (3.2%) |
| test | 28,884 | 608 | 1,065 (3.7%) |

Overlaps: 0 / 0 / 0.

### Scaling fitted on train only — **[DERIVED]**
Computing min/max across all data leaks test information into training.

### Metric: NOT accuracy — **[DERIVED]**
Anomaly rate is 3.6%, so predicting "normal" always scores 96.4%. Report AUC,
precision, recall, F1.

---

## Outstanding

1. **Sensitivity sweeps** for `SOG ≥ 0.5`, the 1800 s split, `WINDOW = 10`
2. **Residual `sog_mean` reverse-bias** (AUC 0.467) — declare as limitation
3. **Ablation** on `crab_mean` and `sog_mean` — keep or drop, decided by model
   performance rather than univariate AUC
