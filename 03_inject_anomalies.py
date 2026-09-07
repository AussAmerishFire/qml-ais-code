"""
Stage 3: inject synthetic anomalies into real vessel tracks, then re-window.

WHY INJECTION AT ALL
--------------------
Real AIS anomalies cannot be verified without port records, enforcement data or
satellite imagery. With no ground truth there is no accuracy/precision/recall,
and therefore no quantum-vs-classical comparison -- which is the whole study.
Naaman et al. (arXiv 2509.15991) made the same choice for ADS-B.

WHY AT REPORT LEVEL, NOT FEATURE LEVEL
--------------------------------------
Features are window aggregates. Corrupting an aggregate is not the same as
corrupting the underlying track and recomputing: a real spoof perturbs positions,
and the *downstream* effects on implied speed, bearing and turn rate are exactly
the signal a detector should pick up. So we modify raw reports, then re-derive.

ATTACK TAXONOMY  [industry-sourced, see DECISIONS.md]
----------------------------------------------------
Windward / MarineTraffic / SkyFi describe four families:
  1. position falsification (GNSS spoofing)  -> DRIFT
  2. implausible route                       -> JUMP
  3. AIS switch-off ("going dark")           -> GAP
  4. false data broadcast                    -> SPEED_LIE
Industry note that sophisticated spoofing "looks continuous and plausible,
unlike a gap" -- hence DRIFT is gradual, not a teleport.

SEVERITY IS SWEPT, NOT ASSUMED
------------------------------
Published magnitudes are vague ("hundreds of nautical miles"). Rather than
invent a constant, each anomaly draws severity from a range and the realised
severity is RECORDED per window, so detection can be reported as a function of
subtlety instead of resting on one arbitrary number.

Run:  .venv/Scripts/python.exe 03_inject_anomalies.py
"""

import numpy as np
import pandas as pd

from importlib import import_module
b = import_module("01_build_windows")   # reuse load / per_step / windows

RNG          = np.random.default_rng(42)
OUT          = "windows_labelled.csv"
ANOMALY_FRAC = 0.20      # share of tracks that receive an anomaly
N_TRACKS     = 3000      # tracks sampled to keep the working set manageable

# severity ranges -- SWEPT, not fixed. low end deliberately subtle.
DRIFT_M      = (100,  3000)    # metres of cumulative positional drift
JUMP_M       = (500, 20000)    # metres of instantaneous displacement
SPEED_LIE_KN = (2,      15)    # knots of falsified speed offset

# GAP: must stay BELOW the track-split threshold (b.GAP_BREAK_S = 1800 s).
# A silence longer than that is, by construction, a track boundary -- the
# pipeline splits there and the anomaly disappears from the window entirely.
# Genuine multi-hour "going dark" is therefore a TRACK-level detection problem,
# surfaced separately via the prev_silence_s feature, not a window-level one.
GAP_S        = (300,  1700)    # 5 - 28 min of silence, inside one track

ATTACKS = ["drift", "jump", "gap", "speed_lie"]


def m_to_deg(m, lat):
    """Convert metres to degrees lat/lon at a given latitude."""
    dlat = m / 111_320.0
    dlon = m / (111_320.0 * max(np.cos(np.radians(lat)), 0.01))
    return dlat, dlon


def apply_drift(t, sev):
    """Gradual positional offset accumulating across the track.
    Positions slowly diverge from truth; reported SOG is left untouched, so
    implied speed (from positions) stops agreeing with it."""
    n = len(t)
    ramp = np.linspace(0, 1, n) ** 1.5          # slow start, growing
    brg = RNG.uniform(0, 2 * np.pi)
    dlat, dlon = m_to_deg(sev, t.LAT.mean())
    t.loc[:, "LAT"] = t.LAT.values + ramp * dlat * np.sin(brg)
    t.loc[:, "LON"] = t.LON.values + ramp * dlon * np.cos(brg)
    return t


def apply_jump(t, sev):
    """Single instantaneous displacement mid-track, then continues from there."""
    n = len(t)
    k = RNG.integers(n // 4, max(n // 4 + 1, 3 * n // 4))
    brg = RNG.uniform(0, 2 * np.pi)
    dlat, dlon = m_to_deg(sev, t.LAT.mean())
    t.iloc[k:, t.columns.get_loc("LAT")] = t.LAT.values[k:] + dlat * np.sin(brg)
    t.iloc[k:, t.columns.get_loc("LON")] = t.LON.values[k:] + dlon * np.cos(brg)
    return t


def apply_gap(t, sev):
    """Delete a run of reports -- the vessel goes dark, then reappears."""
    n = len(t)
    if n < 6:
        return t
    k = RNG.integers(2, n - 3)
    # drop however many reports fall inside the silence window
    dt = t.BaseDateTime.diff().dt.total_seconds().fillna(60).values
    drop, acc = [], 0.0
    for i in range(k, n - 1):
        acc += dt[i]
        drop.append(i)
        if acc >= sev:
            break
    return t.drop(t.index[drop])


def apply_speed_lie(t, sev):
    """Positions untouched; the reported SOG field is falsified.
    Creates a direct contradiction between reported speed and what the
    positions imply -- the classic 'forgot to keep the fields consistent' spoof."""
    n = len(t)
    k = RNG.integers(0, max(n // 2, 1))
    sign = RNG.choice([-1, 1])
    t.iloc[k:, t.columns.get_loc("SOG")] = np.clip(
        t.SOG.values[k:] + sign * sev, 0.1, 39.9)
    return t


APPLY = {"drift": (apply_drift, DRIFT_M),
         "jump":  (apply_jump,  JUMP_M),
         "gap":   (apply_gap,   GAP_S),
         "speed_lie": (apply_speed_lie, SPEED_LIE_KN)}


def main():
    df = b.load()
    df = b.per_step(df)

    # sample tracks (a track = one continuous voyage segment)
    keys = df.groupby(["MMSI", "track_id"]).size()
    keys = keys[keys >= b.WINDOW].index.to_list()
    RNG.shuffle(keys)
    keys = keys[:N_TRACKS]
    print(f"\nsampled {len(keys):,} tracks of >= {b.WINDOW} reports")

    n_anom = int(len(keys) * ANOMALY_FRAC)
    anom_keys = set(map(tuple, [keys[i] for i in range(n_anom)]))

    # float32 (from the loader) cannot hold float64 injection results under
    # pandas 3.x, so widen the columns we are about to modify.
    df = df.astype({"LAT": "float64", "LON": "float64", "SOG": "float64"})

    # pre-group once instead of filtering the full frame per track (much faster)
    groups = {k: g for k, g in df.groupby(["MMSI", "track_id"], sort=False)}

    out = []
    for key in keys:
        t = groups[key].copy().reset_index(drop=True)
        label, atk, sev = 0, "none", 0.0
        if tuple(key) in anom_keys:
            atk = RNG.choice(ATTACKS)
            fn, rng_ = APPLY[atk]
            sev = float(RNG.uniform(*rng_))
            t = fn(t, sev)
            label = 1
        t["label"], t["attack"], t["severity"] = label, atk, sev
        out.append(t)
    D = pd.concat(out, ignore_index=True)

    # recompute deltas on the MODIFIED tracks, then re-window
    print("recomputing features on modified tracks ...")
    D = D.sort_values(["MMSI", "track_id", "BaseDateTime"]).reset_index(drop=True)
    D = b.per_step(D.drop(columns=[c for c in
        ["dt_s","dist_m","implied_kn","d_sog","d_cog","bearing",
         "spd_mismatch","brg_mismatch","crab","track_id"] if c in D.columns]))

    lab = D.groupby(["MMSI", "track_id"])[["label"]].max()
    W = b.windows(D)
    W = W.merge(lab, on=["MMSI", "track_id"], how="left")
    # carry attack type + severity through
    am = D.groupby(["MMSI","track_id"])[["attack","severity"]].first()
    W = W.merge(am, on=["MMSI","track_id"], how="left")

    # Reapply the stage-1 sanity filter. Without this, duplicate-MMSI teleports
    # sit in the NORMAL class and inflate its spd_mismatch ~9x, blurring the
    # very contrast the study measures.
    n0 = len(W)
    keep = ~((W.spd_mismatch_max > b.MAX_IMPLIED_KN) & (W.label == 0))
    W = W[keep].reset_index(drop=True)
    print(f"\n  dropped {n0-len(W):,} corrupt NORMAL windows "
          f"({100*(n0-len(W))/n0:.1f}%) — anomalous windows are never dropped, "
          f"since injection legitimately produces extreme values")

    W.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}")
    print(f"  windows        : {len(W):,}")
    print(f"  anomalous      : {int(W.label.sum()):,} ({100*W.label.mean():.1f}%)")
    print(f"  vessels        : {W.MMSI.nunique():,}")
    print("\nwindows per attack type:")
    print(W.attack.value_counts().to_string())

    FEATS = ["sog_mean","sog_max_jump","cog_turn_max","dt_max",
             "spd_mismatch_max","crab_mean","prev_silence_s"]
    print("\nfeature means, normal vs anomalous:")
    cmp = W.groupby("label")[FEATS].mean().T
    cmp.columns = ["normal", "anomalous"]
    cmp["ratio"] = (cmp.anomalous / cmp.normal).round(2)
    print(cmp.round(3).to_string())

    print("\nper-attack-type feature means (RQ3 table):")
    print(W.groupby("attack")[FEATS].mean().round(2).to_string())


if __name__ == "__main__":
    main()
