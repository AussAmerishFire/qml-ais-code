"""
scripts/17_feature_probe.py
===================
Does adding trend/direction features break the ~0.85 ceiling?

Measured facts this probe is responding to:
  * Capacity is NOT the constraint. A 1,000,000-parameter random forest scored
    0.8159 on validation; the tuned 300-tree depth-6 forest scored 0.8675. A
    40,000-parameter MLP scored 0.8354, a 51-parameter one 0.8120. Making the
    models bigger makes them worse.
  * drift is 37.7% of test anomalies and every model detects it at ~0.60 --
    including the zero-parameter threshold rule, which is the BEST on drift.
    Nothing learned is finding drift signal that one threshold doesn't have.

Why drift is invisible, read off the injection code (scripts/PIPELINE.py a_drift):

    ramp = np.linspace(0, 1, n) ** 1.5
    t["LAT"] += ramp * dlat * np.sin(br)     # position slides along bearing br
    t["LON"] += ramp * dlon * np.cos(br)     # SOG, COG and Heading UNTOUCHED

The vessel is pushed sideways while its reported course stays put. So its
ACTUAL direction of travel stops matching its REPORTED course. Every existing
feature misses this:
  * spd_mismatch_max takes a MAX over 10 reports -- a gradual ramp never
    produces a large single step.
  * crab_mean is COG vs Heading; drift changes neither.
  * bearing (the compass direction actually travelled) IS computed, at
    scripts/PIPELINE.py:232, and then discarded -- it never becomes a feature.

The four features added here:

  course_resid_mean   ang_diff(COG, bearing) -- reported course versus the
                      direction actually travelled. The direct drift signature.
  mismatch_slope      least-squares slope of spd_mismatch across the window.
                      The injection ramp is monotone and accelerating, so drift
                      is a TREND; a max cannot see a trend.
  straightness        net displacement / total path length. Drift bends the
                      track away from a straight run.
  mismatch_mean       mean rather than max: small consistent errors accumulate,
                      random noise does not.

Nothing is committed on the basis of this probe. It measures, on VALIDATION
only, and prints whether drift moves. Deciding to rebuild the dataset comes
after seeing the number.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import PIPELINE as P

ATTACKS = ["speed_lie", "jump", "gap", "dark", "drift"]
OLD = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
       "spd_mismatch_max", "crab_mean", "prev_silence_s"]
NEW = ["course_resid_mean", "mismatch_slope", "straightness", "mismatch_mean"]

# Least-squares slope against evenly spaced x, precomputed: for x = 0..n-1,
# slope = sum((x - xbar) * y) / sum((x - xbar)^2).
_X = np.arange(P.WINDOW, dtype=float)
_XC = _X - _X.mean()
_DEN = (_XC ** 2).sum()


def _slope(v):
    v = np.asarray(v, dtype=float)
    if np.isnan(v).all():
        return 0.0
    v = np.nan_to_num(v, nan=np.nanmean(v))
    return float((_XC * (v - v.mean())).sum() / _DEN)


_orig_per_step = P.per_step
_orig_windows = P.windows


def per_step_v2(df):
    df = _orig_per_step(df)
    # Reported course versus the bearing actually travelled. Both sentinels
    # (COG 360, and the NaN bearing on each track's first report) must drop out
    # or they would masquerade as huge residuals.
    df["course_resid"] = np.where(
        (df.COG == P.COG_NA) | df.bearing.isna() | (df.dist_m < 1.0),
        np.nan, P.ang_diff(df.COG, df.bearing))
    return df


def windows_v2(df):
    rows = []
    for (mmsi, tid), t in df.groupby(["MMSI", "track_id"], sort=False):
        if len(t) < P.WINDOW:
            continue
        for s in range(0, len(t) - P.WINDOW + 1, P.STRIDE):
            w = t.iloc[s:s + P.WINDOW]
            net = P.haversine_m(w.LAT.iloc[0], w.LON.iloc[0],
                                w.LAT.iloc[-1], w.LON.iloc[-1])
            path = float(np.nansum(w.dist_m.values))
            rows.append({
                "MMSI": mmsi, "track_id": tid,
                "t_start": w.BaseDateTime.iloc[0],
                "lat_mean": w.LAT.mean(), "lon_mean": w.LON.mean(),
                "vessel_type": w.VesselType.iloc[0], "length": w.Length.iloc[0],
                # --- the original eight ---
                "sog_mean": w.SOG.mean(),
                "sog_max_jump": w.d_sog.max(),
                "cog_turn_max": w.d_cog.max(),
                "dt_max": w.dt_s.max(),
                "spd_mismatch_max": w.spd_mismatch.max(),
                "crab_mean": w.crab.mean(),
                "prev_silence_s": w.prev_silence_s.iloc[0],
                # --- the four candidates ---
                "course_resid_mean": w.course_resid.mean(),
                "mismatch_slope": _slope(w.spd_mismatch.values),
                "straightness": (net / path) if path > 1.0 else 1.0,
                "mismatch_mean": w.spd_mismatch.mean(),
                # carried through for labelling
                "anom_row": w.anom_row.any() if "anom_row" in w else False,
                "attack": w.attack.iloc[0] if "attack" in w else "none",
                "severity": w.severity.iloc[0] if "severity" in w else 0.0,
            })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    P.per_step, P.windows = per_step_v2, windows_v2

    print("rebuilding with 4 candidate features added ...", flush=True)
    df = P.load()
    df = P.per_step(df)
    df = df.astype({"LAT": "float64", "LON": "float64", "SOG": "float64"})
    D = P.inject(df)
    D = D.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)
    D = P.per_step(D.drop(columns=[c for c in
        ["dt_s", "dist_m", "implied_kn", "d_sog", "d_cog", "bearing",
         "spd_mismatch", "crab", "course_resid", "track_id",
         "prev_silence_s"] if c in D.columns]))
    W = P.windows(D)
    W["label"] = W.anom_row.astype(int)
    W.loc[W.label == 0, ["attack", "severity"]] = ["none", 0.0]
    W = W.drop(columns=["anom_row"])
    W = W[~((W.spd_mismatch_max > P.MAX_IMPLIED_KN) & (W.label == 0))]
    W = W.reset_index(drop=True)

    for c in NEW:
        W[c] = W[c].fillna(W[c].median())
    W["crab_valid"] = (~W.crab_mean.isna()).astype(int)
    W["crab_mean"] = W.crab_mean.fillna(W.crab_mean.median())
    W.to_csv("data/windows_labelled_v2.csv", index=False)
    print(f"  {len(W):,} windows, {int(W.label.sum()):,} anomalous "
          f"({W.label.mean()*100:.2f}%)  [{time.time()-t0:.0f}s]\n", flush=True)

    # Vessel-wise split, same principle as the real pipeline: no MMSI may appear
    # on both sides, or the comparison is contaminated.
    rng = np.random.default_rng(0)
    v = W.MMSI.unique()
    rng.shuffle(v)
    tr_v = set(v[:int(0.75 * len(v))])
    tr, va = W[W.MMSI.isin(tr_v)], W[~W.MMSI.isin(tr_v)]
    print(f"train {len(tr):,} / val {len(va):,}  (vessel-disjoint)\n")

    print("UNIVARIATE AUC -- how much does each feature separate, on its own?")
    print(f"  {'feature':<20}{'overall':>9}{'drift':>9}   (0.5 = useless)")
    order = OLD + ["crab_valid"] + NEW
    for c in order:
        d = va[(va.label == 0) | (va.attack == "drift")]
        a = roc_auc_score(va.label, va[c])
        ad = roc_auc_score(d.label, d[c])
        star = "  <-- NEW" if c in NEW else ""
        print(f"  {c:<20}{max(a,1-a):>9.4f}{max(ad,1-ad):>9.4f}{star}")

    print("\nRANDOM FOREST -- old features vs old + new (300 trees, depth 6)")
    for label, feats in (("old 8", OLD + ["crab_valid"]),
                         ("old + new 12", OLD + ["crab_valid"] + NEW)):
        m = RandomForestClassifier(n_estimators=300, max_depth=6,
                                   random_state=0, n_jobs=-1).fit(tr[feats], tr.label)
        s = m.predict_proba(va[feats])[:, 1]
        pa = " ".join(
            f"{a[:5]}:{roc_auc_score(va.label[(va.label==0)|(va.attack==a)], s[((va.label==0)|(va.attack==a)).values]):.3f}"
            for a in ATTACKS)
        print(f"  {label:<14} AUC {roc_auc_score(va.label, s):.4f}   {pa}")

    print(f"\n[{time.time()-t0:.0f}s]  wrote data/windows_labelled_v2.csv")


if __name__ == "__main__":
    main()
