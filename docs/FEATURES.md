# The 8 Features — full definitions

> **Status (2026-10-02).** Describes the **original 8 features**. The study uses **12**; their definitions, as computed, are in `METHODS.md` Part 6. Kept for provenance — do not cite feature counts or definitions from this file.

Every feature is computed over one **window** = 10 consecutive position reports
from one vessel. Raw AIS gives us position, speed, course and heading per
report; these features describe how those changed across the window.

Normal medians come from the 110,985 unattacked windows.

---

## 1. `sog_mean` — average speed

**SOG = Speed Over Ground**, the vessel's actual speed across the seabed in knots.

**Computed:** average of the SOG field across the 10 reports.

**Why it's here:** baseline context. A harbour tug at 3 knots and a container
ship at 18 behave completely differently, and the model needs to know which it
is looking at before "unusual" means anything.

**Normal median:** 7.71 kn

**⚠️ Known issue:** reads 3.84 for `speed_lie`-attacked windows. That is a bug
in my injection code, not a real signature — see DECISIONS.md.

---

## 2. `sog_max_jump` — biggest speed change

**Computed:** the largest `|SOG[i] − SOG[i−1]|` between any two consecutive
reports in the window, in knots.

**Why it's here:** ships have enormous mass. A loaded tanker cannot gain 10
knots in eight seconds. A large jump is either a data error or fabricated data.

**Normal median:** 0.20 kn — normal vessels barely change speed between reports.

---

## 3. `cog_turn_max` — sharpest turn

**COG = Course Over Ground**, the compass direction the vessel is actually
travelling.

**Computed:** the largest angular change in COG between consecutive reports,
in degrees (0–180).

**Why it's here:** ships have turning radii measured in hundreds of metres. An
instantaneous 90° turn is physically impossible for a large vessel.

**Normal median:** 6.7°

---

## 4. `dt_max` — longest silence within the window

**Computed:** the largest time gap between consecutive reports, in seconds.

**Why it's here:** Class A transponders report every 2–10 seconds under way. A
sudden long gap means the vessel stopped transmitting — briefly going quiet.

**Normal median:** 79 s. **Gap attack:** 975 s.

---

## 5. `spd_mismatch_max` — speed contradicting position ⭐

**This is the most important feature in the study.**

**Computed, in three steps:**
1. Distance between consecutive positions (great-circle / haversine)
2. Divide by elapsed time → **implied speed**: how fast the vessel must have
   been going for those positions to be real
3. `|implied speed − reported SOG|`, take the maximum across the window

**Why it's here:** a spoofer fabricating positions must keep the reported speed
field consistent with them, and usually doesn't. If a vessel claims 6 knots but
its own positions moved 2 km in 60 seconds, those two facts cannot both be true.

**It is an internal consistency check** — it doesn't need to know what normal
looks like, only that the vessel is contradicting itself.

**Normal median:** 0.32 kn (normal vessels are self-consistent to within a
third of a knot). **Jump attack:** 187. **Speed_lie:** 6.15.

---

## 6. `crab_mean` — bow direction vs travel direction

**Heading** = where the bow points. **COG** = where the ship actually goes.
These differ when a current or wind pushes the vessel sideways — sailors call
it crabbing.

**Computed:** average `|COG − Heading|` across the window, in degrees.

**Why it's here:** a real vessel's crab angle is small and physically explained.
Fabricated data often doesn't bother keeping heading consistent with course.

**Normal median:** 2.76°

**⚠️ Missing 19% of the time**, because AIS transmits `511` when the heading
sensor isn't reporting. Missing values are filled with the median — hence
feature 8.

---

## 7. `prev_silence_s` — how long the vessel was dark before this voyage

**Computed:** the time gap that caused this track to be split off from the
previous one. A gap over 30 minutes ends a voyage segment and starts a new one;
this records how long that gap was.

**Why it's here:** this is the **"going dark"** detector — a vessel switching
off its transponder for hours and reappearing elsewhere. That is the anomaly
maritime security cares most about (sanctions evasion, illegal fishing).

**Normal:** 0 s. **Dark attack:** 5,370 s.

**Note:** it is a *track-level* signal, not a within-window one. Every window in
a voyage segment carries the same value.

---

## 8. `crab_valid` — was heading actually measured?

**Computed:** 1 if the heading was real, 0 if it was the 511 "unavailable"
sentinel and `crab_mean` had to be imputed.

**Why it's here:** two reasons. Honesty — the model should know when a value was
invented rather than measured. And information — a vessel that stops reporting
heading mid-voyage may itself be doing something worth noticing.

---

## Summary table

| # | Feature | Plain meaning | Normal median | Best at catching |
|---|---|---|---|---|
| 1 | `sog_mean` | average speed | 7.71 kn | context |
| 2 | `sog_max_jump` | biggest speed change | 0.20 kn | impossible acceleration |
| 3 | `cog_turn_max` | sharpest turn | 6.7° | impossible manoeuvres |
| 4 | `dt_max` | longest gap in window | 79 s | brief silences |
| 5 | **`spd_mismatch_max`** | **speed vs position contradiction** | **0.32 kn** | **spoofing** |
| 6 | `crab_mean` | bow vs travel direction | 2.76° | inconsistent fabrication |
| 7 | `prev_silence_s` | silence before this voyage | 0 s | going dark |
| 8 | `crab_valid` | was heading real? | 1 | missing-data pattern |

## Where they came from

13 candidate features were engineered, then reduced to these by correlation
analysis (fig2). Seven pairs correlated above 0.8 — measuring the same thing
twice — and one survivor was kept from each cluster. With only 6–8 qubits
available, encoding two features that say the same thing wastes a qubit.

Dropped: `sog_std`, `cog_turn_mean`, `dt_mean`, `dist_total_m`,
`spd_mismatch_mean`, `brg_mismatch_mean`, `heading_valid_frac`.
They are still present in the CSV files, unscaled, in case you want to revisit
that decision.
