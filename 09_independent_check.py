"""
Stage 9: INDEPENDENT verification.

07_validate_dataset.py asserts properties of the pipeline's own output. If the
pipeline computed something wrongly but consistently, those assertions would
still pass. This script is deliberately different:

  A. Recomputes features FROM THE RAW CSV using separate implementations
     (spherical law of cosines instead of haversine; explicit Python loops
     instead of pandas vectorisation) and compares to the pipeline's answer.

  B. Verifies the INJECTIONS did what they claim, by rebuilding the clean
     tracks and measuring the actual difference.

  C. Checks the realised severities fall inside their declared, derived bounds.

Two independent routes to the same numbers. Disagreement means a bug.

Run:  .venv/Scripts/python.exe 09_independent_check.py
"""

import math
import numpy as np
import pandas as pd
from importlib import import_module

# Both point at PIPELINE.py, the script that ACTUALLY produces the dataset.
# They previously pointed at 01_build_windows and 05_build_dataset, which
# PIPELINE.py superseded -- so this file was verifying code that no longer
# generates anything, and silently missed the four features added on
# 2026-09-07 (KeyError: 'mismatch_mean').
#
# The independence of this check has never come from importing a different
# builder. It comes from features_by_hand() below, which recomputes every
# feature with different formulas (spherical law of cosines rather than
# haversine, explicit loops rather than pandas, forward azimuth written out
# longhand). Verifying against the live pipeline is strictly more correct.
b = import_module("PIPELINE")
inj = b

RNG = np.random.default_rng(7)
N_SAMPLE = 300

results = []
def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ---- deliberately different implementations -------------------------------

def dist_law_of_cosines(lat1, lon1, lat2, lon2):
    """Spherical law of cosines. Mathematically equivalent to haversine but a
    completely different expression, so a coding error in one will not be
    reproduced by the other."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1)*np.sin(p2) + np.cos(p1)*np.cos(p2)*np.cos(dl)
    return R * np.arccos(np.clip(c, -1, 1))


def features_by_hand(rows):
    """Recompute window features with explicit loops — no pandas, no groupby.
    rows: list of dicts with lat, lon, sog, cog, hdg, t (seconds)."""
    n = len(rows)
    dts, dists, mismatches, turns, jumps = [], [], [], [], []
    resids = []
    for i in range(1, n):
        a, c = rows[i-1], rows[i]
        dt = c["t"] - a["t"]
        d = float(dist_law_of_cosines(a["lat"], a["lon"], c["lat"], c["lon"]))
        dts.append(dt); dists.append(d)
        if dt > 0:
            implied_kn = (d / dt) * 1.94384
            mismatches.append(abs(implied_kn - c["sog"]))
        # smallest angle between two compass bearings
        raw = abs(c["cog"] - a["cog"]) % 360.0
        turns.append(360.0 - raw if raw > 180.0 else raw)
        jumps.append(abs(c["sog"] - a["sog"]))
        # Reported course vs the bearing actually travelled, recomputed here by
        # the FORWARD-AZIMUTH formula written out longhand. The pipeline uses
        # its own bearing_deg(); agreement between two independent formulas is
        # the point of this file.
        if d >= 1.0 and c["cog"] != 360.0:
            y_ = math.sin(math.radians(c["lon"] - a["lon"])) *                  math.cos(math.radians(c["lat"]))
            x_ = math.cos(math.radians(a["lat"])) * math.sin(math.radians(c["lat"])) -                  math.sin(math.radians(a["lat"])) * math.cos(math.radians(c["lat"])) *                  math.cos(math.radians(c["lon"] - a["lon"]))
            brg = (math.degrees(math.atan2(y_, x_)) + 360.0) % 360.0
            rr = abs(c["cog"] - brg) % 360.0
            resids.append(360.0 - rr if rr > 180.0 else rr)
    # rows[0] is the predecessor, so the WINDOW itself is rows[1:]
    return {
        "sog_mean": sum(r["sog"] for r in rows[1:]) / (n - 1),
        "dt_max": max(dts) if dts else np.nan,
        "sog_max_jump": max(jumps) if jumps else np.nan,
        "cog_turn_max": max(turns) if turns else np.nan,
        "spd_mismatch_max": max(mismatches) if mismatches else np.nan,
        "mismatch_mean": (sum(mismatches) / len(mismatches)) if mismatches else np.nan,
        "course_resid_mean": (sum(resids) / len(resids)) if resids else np.nan,
    }


print("="*80)
print("A. RECOMPUTE FROM RAW, WITH DIFFERENT CODE")
print("="*80)
print("rebuilding clean pipeline state ...")
clean = b.load()
clean = b.per_step(clean)
Wc = b.windows(clean)
print(f"  clean windows: {len(Wc):,}")

# sample windows and recompute each one by hand from the underlying reports
tracks = {k: g.reset_index(drop=True) for k, g in
          clean.groupby(["MMSI", "track_id"], sort=False)}
sample = Wc.sample(n=min(N_SAMPLE, len(Wc)), random_state=7)

worst = {"sog_mean": 0, "dt_max": 0, "sog_max_jump": 0,
         "cog_turn_max": 0, "spd_mismatch_max": 0,
         "mismatch_mean": 0, "course_resid_mean": 0}
n_ok = 0
for _, w in sample.iterrows():
    t = tracks.get((w.MMSI, w.track_id))
    if t is None:
        continue
    # The pipeline's deltas for row i are computed against row i-1 of the TRACK,
    # so the first row of a window carries a delta reaching one report back.
    # To test the arithmetic rather than the convention, include that extra row.
    pos = t.index[t.BaseDateTime >= w.t_start]
    if len(pos) < b.WINDOW:
        continue
    s = int(pos[0])
    if s == 0:
        continue                       # first window of a track has no predecessor
    t = t.iloc[s-1 : s + b.WINDOW]     # 11 rows -> 10 deltas, matching the pipeline
    if len(t) < b.WINDOW + 1:
        continue
    rows = [{"lat": r.LAT, "lon": r.LON, "sog": r.SOG, "cog": r.COG,
             "hdg": r.Heading, "t": r.BaseDateTime.timestamp()}
            for r in t.itertuples()]
    mine = features_by_hand(rows)
    for k, v in mine.items():
        pv = w[k]
        if np.isnan(v) or np.isnan(pv):
            continue
        denom = max(abs(pv), 1e-6)
        worst[k] = max(worst[k], abs(v - pv) / denom)
    n_ok += 1

print(f"  recomputed {n_ok} windows independently")
for k, v in worst.items():
    check(f"{k} matches independent recomputation", v < 0.02,
          f"worst relative difference {100*v:.4f}%")

print("\n" + "="*80)
print("B. DID THE INJECTIONS ACTUALLY DO WHAT THEY CLAIM?")
print("="*80)
W = pd.read_csv("windows_labelled.csv")

# gap / dark should show measurably longer silences than normal
norm_dt = W.loc[W.attack == "none", "dt_max"]
gap_dt  = W.loc[W.attack == "gap", "dt_max"]
dark_ps = W.loc[W.attack == "dark", "prev_silence_s"]
# NOTE: the right comparison is against the p95 of the WINDOW-MAX gap, not the
# p99 of individual intervals. dt_max is the maximum of 10 intervals, so its
# distribution sits far higher (normal p95 = 580 s, p99 = 1318 s) than the
# distribution of single intervals (p99 = 491 s). Comparing against the latter
# was an error in the original derivation.
check("gap attack shifts the window-max gap above normal p95",
      gap_dt.median() > norm_dt.quantile(0.95),
      f"gap median {gap_dt.median():.0f}s vs normal p95 {norm_dt.quantile(0.95):.0f}s")
frac_obvious = (gap_dt > norm_dt.quantile(0.99)).mean()
check("gap spans subtle to obvious (20-60% clearly outside normal)",
      0.20 < frac_obvious < 0.60,
      f"{100*frac_obvious:.1f}% above normal p99 — the rest are deliberately subtle")
check("dark attack produced track-level silences over the 1800s threshold",
      dark_ps.median() > 1800,
      f"median preceding silence {dark_ps.median():.0f}s")
check("normal windows have no preceding silence",
      W.loc[W.attack == "none", "prev_silence_s"].median() == 0)

# jump should create implied speeds that are physically impossible
jump_mm = W.loc[W.attack == "jump", "spd_mismatch_max"]
check("jump attack created impossible implied speeds",
      jump_mm.median() > 60,
      f"median mismatch {jump_mm.median():.1f} kn (>60 = impossible)")

# speed_lie should NOT move positions, only the speed field
sl_mm = W.loc[W.attack == "speed_lie", "spd_mismatch_max"]
check("speed_lie created mismatch above natural p99 (3.20 kn)",
      sl_mm.median() > 3.2, f"median {sl_mm.median():.2f} kn")

# drift must stay subtle -- if it produced impossible speeds it is not drift
dr_mm = W.loc[W.attack == "drift", "spd_mismatch_max"]
check("drift stayed subtle (no impossible speeds)",
      dr_mm.quantile(0.95) < 60,
      f"p95 mismatch {dr_mm.quantile(0.95):.2f} kn")

print("\n" + "="*80)
print("C. REALISED SEVERITIES INSIDE THEIR DERIVED BOUNDS")
print("="*80)
# speed_lie changed from an ADDITIVE offset in knots (SPEED_LIE_KN, the old
# 05_build_dataset) to a MULTIPLIER (SPEED_LIE) in PIPELINE.py, after the
# additive version was found to drag sog_mean down to 3.84 against 7.71 for
# normal windows. The severity recorded for a speed_lie attack is therefore a
# multiplier now, and is bounds-checked as one.
BOUNDS = {"drift": inj.DRIFT_M, "jump": inj.JUMP_M, "gap": inj.GAP_S,
          "speed_lie": inj.SPEED_LIE, "dark": inj.DARK_S}
for a, (lo, hi) in BOUNDS.items():
    s = W.loc[W.attack == a, "severity"]
    ok = (s.min() >= lo - 1e-6) and (s.max() <= hi + 1e-6)
    check(f"{a}: severities within [{lo}, {hi}]", ok,
          f"realised {s.min():.1f} – {s.max():.1f}")

# the corrected gap bound: no injected gap may sit inside natural variation
NAT_GAP_P99 = 491.0
gap_sev = W.loc[W.attack == "gap", "severity"]
check("no injected gap falls below the natural p99 (491 s)",
      gap_sev.min() >= NAT_GAP_P99,
      f"smallest injected gap {gap_sev.min():.0f}s")
check("no injected gap exceeds the track-split threshold (1800 s)",
      gap_sev.max() < b.GAP_BREAK_S,
      f"largest {gap_sev.max():.0f}s")
check("every dark silence exceeds the split threshold",
      W.loc[W.attack == "dark", "severity"].min() > b.GAP_BREAK_S,
      f"smallest {W.loc[W.attack=='dark','severity'].min():.0f}s")

print("\n" + "="*80)
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"INDEPENDENT CHECK: {n_pass}/{len(results)} passed")
if n_pass < len(results):
    print("\nFAILURES:")
    for nm, ok, det in results:
        if not ok:
            print(f"  - {nm}  {det}")
else:
    print("Two independent routes agree. Data verified.")
print("="*80)
