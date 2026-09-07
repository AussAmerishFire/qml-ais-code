"""
Stage 5: the finished dataset. Replaces 03_inject_anomalies.py.

Implements PLAN.md steps 1-6:
  1. WINDOW-level labelling (was: whole-track, ~85% mislabelled)
  2. fifth attack type: track-level "going dark"
  3. log-transform skewed features
  4. crab_mean: median-impute + validity flag
  5. scale to [0, pi], fitted on TRAIN ONLY
  6. train/test split BY VESSEL (no ship in both sides)

Outputs:
  windows_labelled.csv  - every window, unscaled, with labels + attack type
  dataset_train.csv     - scaled, ready for a model
  dataset_test.csv      - scaled with the TRAIN scaler

Run:  .venv/Scripts/python.exe 05_build_dataset.py
"""

import numpy as np
import pandas as pd
from importlib import import_module

b = import_module("01_build_windows")

RNG          = np.random.default_rng(42)

# ANOMALY_FRAC: share of TRACKS attacked (not windows -- yields differ, see below).
# DERIVED from what the experiment needs: the label-scarcity curve trains on up
# to 500 labelled anomalies and draws 5 independent seeds, so ~2,500 anomalous
# training windows is the requirement. 0.45 of tracks delivers ~2,500.
# NOTE: the resulting 3.6% window-level anomaly rate is HIGHER than reality
# (industry reports ~0.7% of lost-transmission events as genuinely suspicious).
# Deliberate: a realistic 0.7% would leave too few positives to train on at the
# smallest label budgets. State this as a limitation.
ANOMALY_FRAC = 0.45
N_TRACKS     = 5900      # essentially every track available
TEST_FRAC    = 0.25      # standard hold-out proportion
VAL_FRAC     = 0.15      # standard; leaves 60% for training

# Attacks differ enormously in how many WINDOWS one attacked track produces,
# because their masks differ in width. Measured on the first run (120 tracks each):
#
#   speed_lie 2049 windows -> 17.1 per track     jump 162 -> 1.35
#   drift      861         ->  7.2               gap  101 -> 0.84
#                                                dark  16 -> 0.13 (since fixed)
#
# Assigning tracks evenly therefore produced a 128:1 imbalance. Instead we
# allocate tracks INVERSELY to measured yield, so each attack contributes a
# comparable number of windows. Weights are relative track counts.
YIELD_WEIGHT = {"speed_lie": 1.0, "drift": 2.5, "jump": 12.0,
                "gap": 16.0, "dark": 22.0}

# ---------------------------------------------------------------------------
# SEVERITY RANGES — every bound derived from MEASURED properties of the real
# data, not chosen. Reference measurements from 1,256,434 unmodified reports:
#
#   distance between consecutive reports : median 272 m, p90 551 m, p99 1,285 m
#   gap between consecutive reports      : median  70 s, p99 491 s, p99.9 1,409 s
#   natural speed/position mismatch      : median 0.11 kn, p99 3.20, p99.9 11.27
#   median reporting interval            : 79 s
# ---------------------------------------------------------------------------

# DRIFT: lower bound is BELOW the median single-step movement (272 m), so the
# vessel's own normal motion in one interval exceeds the whole injected drift --
# undetectable by construction. Upper is ~10x a normal step, spread across the
# track, so per-step motion stays plausible and no impossible speed is produced.
DRIFT_M      = (100,  3000)

# JUMP: detected via implied speed = displacement / interval. At the median 79 s:
#   500 m -> 12.3 kn  (plausible; and 500 m is near the p90 of NORMAL movement)
#  20000 m -> 492 kn  (physically impossible)
# The range therefore spans "indistinguishable from normal" to "obviously fake".
JUMP_M       = (500, 20000)

# GAP: must exceed natural transmission variation to be a real signal, and stay
# BELOW GAP_BREAK_S (1800 s) or the pipeline splits the track and the anomaly
# ceases to exist.
#   lower 500 s : just above the p99 of natural gaps (491 s). At the previous
#                 300 s, 1.78% of LEGITIMATE gaps exceeded it -- injected gaps
#                 were indistinguishable from normal behaviour.
#   upper 1700 s: just under the split threshold.
GAP_S        = (500,  1700)

# SPEED_LIE: multiplier (x or ÷). On a median 7.7 kn vessel:
#   x1.8 -> 13.9 kn, creating a 6.2 kn mismatch (above natural p99 of 3.20)
#   x5.0 -> 38.5 kn, creating a 30.8 kn mismatch (far above p99.9 of 11.27)
# Spans "just outside natural variation" to "far outside it".
SPEED_LIE_KN = (1.8,   5.0)

# DARK: must EXCEED GAP_BREAK_S to register as a track-level blackout rather
# than a brief silence. Lower bound 1 h = 2x the threshold, guaranteeing a split.
# Upper 6 h is where a vessel could plausibly have left receiver coverage
# legitimately rather than switched its transponder off.
DARK_S       = (3600, 21600)

ATTACKS = ["drift", "jump", "gap", "speed_lie", "dark"]

# ---------------------------------------------------------------- attacks
# Each returns (modified_track, boolean mask of AFFECTED ROWS).
# The mask is what fixes step 1: a window is anomalous only if it actually
# contains an affected row, instead of inheriting the label from its track.

def m_to_deg(m, lat):
    return m / 111_320.0, m / (111_320.0 * max(np.cos(np.radians(lat)), 0.01))


def a_drift(t, sev):
    n = len(t)
    ramp = np.linspace(0, 1, n) ** 1.5
    br = RNG.uniform(0, 2*np.pi)
    dlat, dlon = m_to_deg(sev, t.LAT.mean())
    t["LAT"] = t.LAT.values + ramp * dlat * np.sin(br)
    t["LON"] = t.LON.values + ramp * dlon * np.cos(br)
    # drift only becomes detectable once it has accumulated; mark the second half
    mask = np.zeros(n, bool); mask[n//2:] = True
    return t, mask


def a_jump(t, sev):
    n = len(t)
    k = int(RNG.integers(n//4, max(n//4+1, 3*n//4)))
    br = RNG.uniform(0, 2*np.pi)
    dlat, dlon = m_to_deg(sev, t.LAT.mean())
    t.iloc[k:, t.columns.get_loc("LAT")] = t.LAT.values[k:] + dlat*np.sin(br)
    t.iloc[k:, t.columns.get_loc("LON")] = t.LON.values[k:] + dlon*np.cos(br)
    mask = np.zeros(n, bool); mask[max(0,k-1):min(n,k+2)] = True   # the discontinuity
    return t, mask


def a_gap(t, sev):
    n = len(t)
    if n < 8:
        return t, np.zeros(n, bool)
    k = int(RNG.integers(2, n-4))
    dt = t.BaseDateTime.diff().dt.total_seconds().fillna(60).values
    drop, acc = [], 0.0
    for i in range(k, n-2):
        acc += dt[i]; drop.append(i)
        if acc >= sev: break
    keep = np.ones(n, bool); keep[drop] = False
    t2 = t[keep].reset_index(drop=True)
    mask = np.zeros(len(t2), bool)
    j = min(k, len(t2)-1)
    mask[max(0, j-1):min(len(t2), j+1)] = True   # rows straddling the silence
    return t2, mask


def a_speed_lie(t, sev):
    """Positions untouched; the reported SOG field is falsified.

    BUG FIX: the first version added or subtracted a fixed number of knots and
    clipped at 0.1. On a 7-knot vessel a -15 kn offset bottomed out at 0.1 while
    +15 simply moved to 22 -- asymmetric, and it dragged mean reported speed from
    7.71 down to 3.84. A model could have learned "slow = anomalous", which would
    have been my artefact rather than a spoofing signature.

    Now the falsification is MULTIPLICATIVE and symmetric in log space: the speed
    is multiplied or divided by the same factor with equal probability, so
    neither direction is favoured and clipping is rare.
    """
    FLOOR, CEIL = 0.5, 39.0          # stay inside the valid, under-way range
    n = len(t)
    k = int(RNG.integers(0, max(n//2, 1)))
    s = t.SOG.values[k:]

    # Headroom available in each direction WITHOUT hitting a bound. Multiplying
    # a 12-knot vessel by 5 would hit the 39.9 ceiling, so 30% of windows used
    # to pin there while division never clipped -- that asymmetry, not the
    # falsification itself, was the artefact.
    up   = CEIL / max(s.max(), 0.1)
    down = max(s.min(), 0.1) / FLOOR

    dirs = []
    if up   > 1.2: dirs.append(("up",   min(sev, up)))
    if down > 1.2: dirs.append(("down", min(sev, down)))
    if not dirs:                       # no room either way -- skip this track
        return t, np.zeros(n, bool)

    which, mag = dirs[int(RNG.integers(len(dirs)))]
    factor = mag if which == "up" else 1.0 / mag
    t.iloc[k:, t.columns.get_loc("SOG")] = np.clip(s * factor, FLOOR, CEIL)
    mask = np.zeros(n, bool); mask[k:] = True
    return t, mask


def a_dark(t, sev):
    """Delete a long stretch (> the 30-min split threshold) so the track breaks
    in two and the second half carries a large preceding silence. This is the
    'transponder switched off' anomaly -- and it is a TRACK-level event, which
    is why prev_silence_s exists as a feature.

    Earlier version fired on only 16 of 120 attacked tracks: it picked a random
    start point and often ran out of track before accumulating enough silence.
    Now the start point is chosen to guarantee room, and the target duration is
    capped at what the track can actually supply.
    """
    n = len(t)
    if n < 3 * b.WINDOW:                       # need room either side of the gap
        return t, np.zeros(n, bool)
    dt = t.BaseDateTime.diff().dt.total_seconds().fillna(60).values
    total = float(np.nansum(dt))
    if total < 2 * b.GAP_BREAK_S:              # voyage too short to hide a gap in
        return t, np.zeros(n, bool)

    # cap the requested silence at 40% of the voyage so the track survives it
    target = min(sev, 0.40 * total)
    if target < b.GAP_BREAK_S * 1.1:
        return t, np.zeros(n, bool)

    # start early enough that WINDOW rows remain after the deletion
    k = int(RNG.integers(b.WINDOW, max(b.WINDOW + 1, n - 2 * b.WINDOW)))
    drop, acc = [], 0.0
    for i in range(k, n - b.WINDOW):
        acc += dt[i]; drop.append(i)
        if acc >= target: break
    if acc < b.GAP_BREAK_S:
        return t, np.zeros(n, bool)

    keep = np.ones(n, bool); keep[drop] = False
    t2 = t[keep].reset_index(drop=True)
    mask = np.zeros(len(t2), bool)
    j = min(k, len(t2) - 1)
    mask[j:min(len(t2), j + b.WINDOW)] = True   # windows right after it reappears
    return t2, mask


APPLY = {"drift": (a_drift, DRIFT_M), "jump": (a_jump, JUMP_M),
         "gap": (a_gap, GAP_S), "speed_lie": (a_speed_lie, SPEED_LIE_KN),
         "dark": (a_dark, DARK_S)}

# ---------------------------------------------------------------- features
CORE = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
        "spd_mismatch_max", "crab_mean", "prev_silence_s"]
# step 3: these span orders of magnitude -> log before scaling, or angle
# encoding wastes the qubit (99% of values would map to nearly the same angle)
LOGGED = ["sog_max_jump", "cog_turn_max", "dt_max", "spd_mismatch_max",
          "crab_mean", "prev_silence_s"]


def main():
    df = b.load()
    df = b.per_step(df)

    keys = df.groupby(["MMSI", "track_id"]).size()
    keys = keys[keys >= b.WINDOW].index.to_list()
    RNG.shuffle(keys)
    keys = keys[:N_TRACKS]
    n_anom = int(len(keys) * ANOMALY_FRAC)
    # allocate attacked tracks inversely to measured window yield
    w = np.array([YIELD_WEIGHT[a] for a in ATTACKS], float)
    quota = np.floor(n_anom * w / w.sum()).astype(int)
    assign = {}
    pos = 0
    for a, q in zip(ATTACKS, quota):
        for key in keys[pos:pos+q]:
            assign[tuple(key)] = a
        pos += q
    print(f"\nsampled {len(keys):,} tracks; {len(assign):,} attacked")
    print("  track allocation: " +
          "  ".join(f"{a}={q}" for a, q in zip(ATTACKS, quota)))

    df = df.astype({"LAT": "float64", "LON": "float64", "SOG": "float64"})
    groups = {k: g for k, g in df.groupby(["MMSI", "track_id"], sort=False)}

    out = []
    for key in keys:
        t = groups[key].copy().reset_index(drop=True)
        atk, sev = "none", 0.0
        mask = np.zeros(len(t), bool)
        if tuple(key) in assign:
            atk = assign[tuple(key)]
            fn, rng_ = APPLY[atk]
            sev = float(RNG.uniform(*rng_))
            t, mask = fn(t, sev)
            if not mask.any():
                atk, sev = "none", 0.0     # attack could not be applied
        t["anom_row"] = mask               # <-- step 1 lives here
        t["attack"], t["severity"] = atk, sev
        out.append(t)
    D = pd.concat(out, ignore_index=True)

    print("recomputing features on modified tracks ...")
    D = D.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)
    D = b.per_step(D.drop(columns=[c for c in
        ["dt_s","dist_m","implied_kn","d_sog","d_cog","bearing","spd_mismatch",
         "brg_mismatch","crab","track_id","prev_silence_s"] if c in D.columns]))

    W = b.windows(D)
    # step 1: a window is anomalous only if it CONTAINS an affected row
    lab = (D.assign(_i=np.arange(len(D)))
             .groupby(["MMSI","track_id"])
             .apply(lambda g: g, include_groups=False))
    W = attach_window_labels(D, W)

    # sanity filter, normal class only
    n0 = len(W)
    W = W[~((W.spd_mismatch_max > b.MAX_IMPLIED_KN) & (W.label == 0))].reset_index(drop=True)
    print(f"  dropped {n0-len(W):,} corrupt normal windows")

    W.to_csv("windows_labelled.csv", index=False)
    report(W)
    split_and_scale(W)


def attach_window_labels(D, W):
    """Recompute window labels by walking the same windowing logic and asking
    whether any row inside the window was actually modified."""
    rows = []
    for (mmsi, tid), t in D.groupby(["MMSI", "track_id"], sort=False):
        if len(t) < b.WINDOW:
            continue
        for s in range(0, len(t) - b.WINDOW + 1, b.STRIDE):
            w = t.iloc[s:s+b.WINDOW]
            rows.append({"MMSI": mmsi, "track_id": tid,
                         "t_start": w.BaseDateTime.iloc[0],
                         "label": int(w.anom_row.any()),
                         "attack": w.attack.iloc[0] if w.anom_row.any() else "none",
                         "severity": w.severity.iloc[0] if w.anom_row.any() else 0.0})
    L = pd.DataFrame(rows)
    return W.drop(columns=[c for c in ["label","attack","severity"] if c in W.columns]) \
            .merge(L, on=["MMSI","track_id","t_start"], how="left")


def report(W):
    print(f"\nwindows      : {len(W):,}")
    print(f"anomalous    : {int(W.label.sum()):,} ({100*W.label.mean():.1f}%)")
    print(f"vessels      : {W.MMSI.nunique():,}")
    print("\nwindows per attack type:")
    print(W.attack.value_counts().to_string())
    print("\nfeature medians, normal vs anomalous:")
    c = W.groupby("label")[CORE].median().T
    c.columns = ["normal", "anomalous"]
    c["ratio"] = (c.anomalous / c.normal.replace(0, np.nan)).round(2)
    print(c.round(3).to_string())
    print("\nper-attack medians (RQ3 table):")
    print(W.groupby("attack")[CORE].median().round(2).to_string())


def split_and_scale(W):
    """Steps 4, 5, 6."""
    # step 4: impute crab_mean, keep a flag -- a vessel that stops reporting
    # heading mid-voyage may itself be informative
    W = W.copy()
    W["crab_valid"] = (~W.crab_mean.isna()).astype(int)
    W["crab_mean"] = W.crab_mean.fillna(W.crab_mean.median())

    # step 6: THREE-WAY split BY VESSEL. No ship may appear in more than one
    # side, or the model memorises individual vessels instead of learning
    # behaviour. Validation exists so hyperparameters can be tuned without
    # touching test -- tuning against test contaminates the thing you are
    # supposed to be measuring on.
    # STRATIFIED vessel split. A plain random split left validation at 2.6%
    # anomalous against test at 4.0%, because vessels differ in how many
    # anomalies they carry and random assignment lets the rate drift.
    #
    # Method: bin vessels by how many anomalous windows they contain, then split
    # EACH BIN 60/15/25. Every split therefore receives the same mix of
    # anomaly-rich and anomaly-free vessels. Vessels stay whole, so there is
    # still no leakage.
    per_v = (W.groupby("MMSI")
               .agg(n=("label", "size"), k=("label", "sum"))
               .reset_index())

    def bin_of(k):
        if k == 0:  return 0
        if k <= 2:  return 1
        if k <= 8:  return 2
        return 3
    per_v["bin"] = per_v.k.map(bin_of)

    assign_split = {}
    for b, grp in per_v.groupby("bin"):
        v = grp.MMSI.to_numpy().copy()
        RNG.shuffle(v)
        n = len(v)
        n_te = int(round(n * TEST_FRAC))
        n_va = int(round(n * VAL_FRAC))
        for m in v[:n_te]:               assign_split[m] = "test"
        for m in v[n_te:n_te + n_va]:    assign_split[m] = "val"
        for m in v[n_te + n_va:]:        assign_split[m] = "train"

    split_col = W.MMSI.map(assign_split)
    tr = W[split_col == "train"].reset_index(drop=True)
    va = W[split_col == "val"].reset_index(drop=True)
    te = W[split_col == "test"].reset_index(drop=True)

    feats = CORE + ["crab_valid"]

    # step 3: log-transform the skewed ones
    for d in (tr, va, te):
        for c in LOGGED:
            d[c] = np.log10(d[c].clip(lower=0) + 1e-2)

    # step 5: scale to [0, pi] using TRAIN statistics only (no leakage into
    # validation or test)
    lo, hi = tr[feats].min(), tr[feats].max()
    rng_ = (hi - lo).replace(0, 1)
    for d in (tr, va, te):
        d[feats] = ((d[feats] - lo) / rng_).clip(0, 1) * np.pi

    tr.to_csv("dataset_train.csv", index=False)
    va.to_csv("dataset_val.csv", index=False)
    te.to_csv("dataset_test.csv", index=False)

    print("\n" + "="*70)
    print("THREE-WAY SPLIT (by vessel — no ship in more than one side)")
    print("="*70)
    for name, d in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name:5s}  windows={len(d):>6,}  vessels={d.MMSI.nunique():>5,}  "
              f"anomalous={int(d.label.sum()):>5,} ({100*d.label.mean():.1f}%)")
    ov = [("train/val", set(tr.MMSI) & set(va.MMSI)),
          ("train/test", set(tr.MMSI) & set(te.MMSI)),
          ("val/test", set(va.MMSI) & set(te.MMSI))]
    for nm, s in ov:
        print(f"  vessel overlap {nm:11s}: {len(s)}  {'OK' if not s else 'LEAKAGE!'}")
    print(f"\n  {len(feats)} features scaled to [0, pi]:")
    print(tr[feats].describe().T[["min","50%","max"]].round(3).to_string())


if __name__ == "__main__":
    main()
