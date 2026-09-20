# Injection Parameters & Data Timeline

---

# TABLE 1 — What each attack changes, and on what basis

## The honest framing first

Two different things were decided, with very different levels of justification:

- **WHICH attacks to simulate** — taken from the industry taxonomy
  (Windward, MarineTraffic, SkyFi describe four families; the fifth was added
  because the author's Govia ruleset ranks going dark as Critical). **Sourced.**
- **HOW BIG each attack is** — the severity ranges. **These are mine.** Published
  descriptions are qualitative ("hundreds of nautical miles", "signal silence").
  No paper gives distributions.

Mitigation: severity is **drawn from a range and recorded per window**, so
detection can be reported as a function of subtlety rather than resting on one
invented constant.

## The five attacks

| Attack | Field(s) modified | Mechanism | Severity range | Realised median | Basis for the range |
|---|---|---|---|---|---|
| **drift** | `LAT`, `LON` | positions offset along a random bearing, growing as `(i/n)^1.5` across the track. Reported SOG untouched. | 100–3,000 m | 1,651 m | **[JUDGMENT]** — sub-km to a few km keeps the track plausible, which is the point of gradual spoofing |
| **jump** | `LAT`, `LON` | single instantaneous displacement at a random point; the track continues from the new position | 500–20,000 m | 9,368 m | **[JUDGMENT]** — spans "barely visible" to "obviously wrong" so difficulty varies |
| **gap** | rows deleted | consecutive reports removed until the target silence accumulates | 300–1,700 s | 957 s | **[CONSTRAINED]** — must stay **below** `GAP_BREAK_S`=1800 s or the pipeline splits the track and the anomaly disappears |
| **speed_lie** | `SOG` | reported speed multiplied or divided by a factor; positions untouched | ×1.8–5.0 | ×3.2 | **[JUDGMENT]** — factor chosen from available headroom so neither the 0.5 kn floor nor 39 kn ceiling is hit |
| **dark** | rows deleted | long stretch removed, **exceeding** the split threshold, so the track breaks and the second half carries a large preceding silence | 3,600–21,600 s | 13,042 s | **[CONSTRAINED]** — must **exceed** `GAP_BREAK_S` to register as a track-level event |

## Why each attack is detectable — the causal chain

You asked how I knew which parameters must change for something to count as an
anomaly. The answer is that each attack breaks a *physical consistency* that
normal vessel behaviour maintains:

| Attack | What breaks | Feature that catches it | AUC |
|---|---|---|---|
| **jump** | positions imply a speed the vessel never reported | `spd_mismatch_max` | **0.932** |
| **speed_lie** | reported speed contradicts the vessel's own positions | `spd_mismatch_max` | **0.982** |
| **gap** | reporting interval violates the Class A 2–10 s norm | `dt_max` | **0.915** |
| **dark** | a voyage begins after hours of unexplained silence | `prev_silence_s` | **0.778** |
| **drift** | *almost nothing* — that is the design | best is 0.583 | **hard case** |

The key insight is that **four of five are internal contradictions**, not
deviations from a learned norm. A spoofer must keep position, speed, course and
timing mutually consistent; failing to do so is detectable without knowing what
"normal" looks like.

`drift` is deliberately the exception: positions move slowly enough that implied
speed stays plausible at every step. It is the attack that requires a model
rather than a threshold, and it is where the quantum-vs-classical comparison
will actually be decided.

## Attack allocation

Attacks differ enormously in how many *windows* one attacked track produces,
because their masks differ in width — `speed_lie` corrupts hundreds of rows,
`jump` corrupts three. Assigning tracks evenly gave a **128:1** window
imbalance, so tracks are allocated **inversely to measured yield**:

| Attack | Relative track weight | Windows produced |
|---|---:|---:|
| speed_lie | 1.0 | 835 |
| drift | 2.5 | 1,198 |
| jump | 12.0 | 705 |
| gap | 16.0 | 678 |
| dark | 22.0 | 673 |

Final imbalance: **1.8:1**.

---

# TABLE 2 — Full data timeline

| # | Stage | Rows / units | Change | % of raw | Vessels |
|---|---|---:|---:|---:|---:|
| 0 | Raw NOAA file `data/raw/AIS_2023_01_01.csv` | **8,156,509** reports | — | 100.0% | 15,932 |
| 1 | Class A transceivers only | 6,111,077 | −2,045,432 | 74.9% | 8,428 |
| 2 | `SOG < 40` kn (drops 102.3 sentinel) | 6,094,912 | −16,165 | 74.7% | 8,415 |
| 3 | Valid coordinates | 6,094,912 | 0 | 74.7% | 8,415 |
| 4 | **Under way (`SOG ≥ 0.5`)** | **1,463,926** | **−4,630,986** | 17.9% | 5,150 |
| 5 | Vessels with ≥200 reports | 1,256,434 | −207,492 | 15.4% | 2,441 |
| 6 | Split into voyages at 30-min silences | 7,935 tracks | — | — | 2,441 |
| 7 | Tracks with ≥10 reports | 5,928 tracks | −2,007 | — | 2,441 |
| 8 | **Windowed** (10 reports, stride 10) | **~122,371 windows** | ÷10 | — | 2,441 |
| 9 | After injection (gap/dark delete rows) | 115,107 windows | −7,264 | — | 2,435 |
| 10 | Split by vessel | 115,107 | 0 | — | 2,435 |

## Step-by-step notes

**Step 4 is the biggest cut** — 4.6 million rows, 76% of what reached it. Two
thirds of all AIS traffic is vessels moored at berths. Median speed before this
filter: **exactly 0.00 kn**. After: 7.50 kn.

**Step 8 is not a filter.** Ten position reports become one window — the count
divides by ten because data is *aggregated*, not discarded.

**Step 9** is the injection itself: `gap` and `dark` work by deleting reports,
so attacked tracks yield slightly fewer windows.

## Final splits

| Split | Windows | Vessels | Anomalous |
|---|---:|---:|---|
| train | 69,712 | 1,462 | 2,496 (3.6%) |
| val | 16,511 | 365 | 528 (3.2%) |
| test | 28,884 | 608 | 1,065 (3.7%) |
| **total** | **115,107** | **2,435** | **4,089 (3.6%)** |

Vessel overlap between any two splits: **0**.

## Compression summary

```
8,156,509 raw reports
        ↓  filtering removes 84.6%
1,256,434 usable reports
        ↓  aggregation: 10 reports → 1 window
  115,107 windows
        ↓  3.6% carry an injected anomaly
    4,089 anomalous windows
```

One day of US coastal AIS yields roughly **4,000 labelled anomaly examples** —
against a study that needs at most 500 for training. Ample headroom.
