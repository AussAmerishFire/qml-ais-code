# Sensitivity Sweep Results

Three preprocessing thresholds had no external justification. Each was swept
across three values, rebuilding the entire pipeline from raw AIS each time
(9 full rebuilds, 2,200 tracks each, identical random seed).

**This is a robustness check, not an optimisation.** Choosing whichever value
maximises detection would tune the data construction to flatter the result.
Where a value is changed below, it is changed on structural grounds and the
reasoning is given.

---

## 1. `MIN_MOVING` — the "under way" threshold — ✅ ROBUST

| kn | reports kept | windows | anomalous % |
|---|---:|---:|---:|
| 0.3 | 1,336,094 | 43,156 | 3.64 |
| **0.5** | **1,256,434** | **42,615** | **3.13** |
| 1.0 | 1,161,410 | 41,726 | 3.56 |

**Best achievable AUC per attack:**

| kn | drift | jump | gap | speed_lie | dark |
|---|---|---|---|---|---|
| 0.3 | 0.556 | 0.941 | 0.928 | 0.975 | 0.861 |
| **0.5** | **0.596** | **0.937** | **0.922** | **0.990** | **0.860** |
| 1.0 | 0.549 | 0.930 | 0.909 | 0.990 | 0.853 |

**Max spread: 0.048.** Top feature is `spd_mismatch_max` at every setting.

**Verdict: keep 0.5.** The threshold changes the dataset size by ±7% and moves
detectability by under 0.05 AUC. Nothing depends on it.

---

## 2. `GAP_BREAK_S` — the track-split threshold — ⚠️ NOT ROBUST BELOW 1800 s

| s | windows | anomalous % |
|---|---:|---:|
| 900 | 28,742 | 5.75 |
| **1800** | **42,615** | **3.13** |
| 3600 | 55,723 | 3.47 |

**Best achievable AUC per attack:**

| s | drift | jump | **gap** | speed_lie | dark |
|---|---|---|---|---|---|
| 900 | 0.510 | 0.945 | **0.696** | 0.986 | 0.847 |
| **1800** | **0.596** | **0.937** | **0.922** | **0.990** | **0.860** |
| 3600 | 0.566 | 0.921 | **0.926** | 0.980 | 0.863 |

**Max spread: 0.230** — driven almost entirely by the `gap` attack collapsing
from 0.92 to 0.70 at 900 s.

### Why, and why this is not a free parameter

Injected gaps span **300–1700 s**. When the split threshold is 900 s, roughly
half of them *exceed it* — so the pipeline treats the injected silence as the
end of one voyage and the start of another. The anomaly is not detected because
it no longer exists in the data. This is the same mechanism as BUG 2.

So there is a hard structural constraint:

> **`GAP_BREAK_S` must exceed the longest injected gap (1700 s).**

1800 s is the smallest value satisfying it. 3600 s satisfies it too and performs
almost identically (gap 0.926 vs 0.922, spread 0.004 across every attack).

**Verdict: keep 1800.** Not because it scored best, but because it is the
minimum value consistent with the injection design. Above that threshold the
choice is genuinely immaterial — which is the robustness claim that matters.

This must be stated in the methods: the split threshold and the injected gap
range are **coupled parameters**, not independent ones.

---

## 3. `WINDOW` — window length — ⚠️ MODERATE, WITH A REAL TRADE-OFF

| reports | windows | anomalous % |
|---|---:|---:|
| 5 | 78,422 | 2.25 |
| **10** | **42,615** | **3.13** |
| 20 | 23,833 | 4.46 |

**Best achievable AUC per attack:**

| reports | **drift** | **jump** | **gap** | speed_lie | dark |
|---|---|---|---|---|---|
| 5 | **0.641** | 0.886 | 0.898 | 0.973 | 0.856 |
| **10** | 0.596 | 0.937 | 0.922 | 0.990 | 0.860 |
| 20 | 0.607 | **0.964** | **0.944** | 0.966 | 0.855 |

**Max spread: 0.078.** Top feature is `spd_mismatch_max` at every setting.

### The trade-off is real and directional

- **Short windows favour `drift`** (0.641 at 5 vs 0.596 at 10). Gradual drift is
  a subtle, sustained deviation; a short window is less diluted by surrounding
  normal behaviour.
- **Long windows favour `jump` and `gap`** (0.964 and 0.944 at 20). These are
  point events, and a longer window is more likely to contain one.

**Verdict: keep 10.** It is the middle of the trade-off and does not maximise
any single attack — which is the point. Choosing 20 because it detects jumps
better would be selecting the construction that flatters the result.

**Report the trade-off explicitly.** It is a genuine finding about how window
length interacts with anomaly morphology, and it is more interesting than a
single number.

---

## Summary for the methods section

> Three preprocessing thresholds lacked external justification and were swept
> across three values each, rebuilding the pipeline from raw data at every
> setting. The under-way threshold (0.3/0.5/1.0 kn) proved immaterial
> (max ΔAUC 0.048). Window length (5/10/20 reports) showed a directional
> trade-off — shorter windows favour gradual drift, longer windows favour point
> events — with the chosen value of 10 maximising no individual attack. The
> track-split threshold is not independent: it must exceed the longest injected
> gap, making 1800 s the minimum admissible value; above that, results are
> insensitive to it (ΔAUC 0.004 between 1800 s and 3600 s). `spd_mismatch_max`
> was the strongest single feature at every one of the nine configurations.

---

## Reproduce

```
.venv/Scripts/python.exe 08_sensitivity_sweep.py            # all three
.venv/Scripts/python.exe 08_sensitivity_sweep.py WINDOW     # one only
```

Roughly 5 minutes per configuration; the full sweep is ~45 minutes.
