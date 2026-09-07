"""
Stage 8: sensitivity sweep over the three unjustified thresholds.

PURPOSE — read this before interpreting the output.

This is NOT an optimisation. Choosing the threshold that maximises detection
would mean tuning the data construction to produce the best-looking answer,
which is a form of overfitting and exactly how QML papers manufacture fake
advantage.

The question is: DO THE CONCLUSIONS SURVIVE A DIFFERENT CHOICE?
If feature rankings and per-attack detectability hold across all tested values,
the arbitrary constant does not matter and the result is robust to it.

Where a principled choice exists, it comes from the DATA's own structure
(distribution breaks, reporting intervals), never from downstream performance.

Swept:
  MIN_MOVING   0.3 / 0.5 / 1.0 kn      "under way" threshold
  GAP_BREAK_S  900 / 1800 / 3600 s     track-split threshold
  WINDOW       5 / 10 / 20 reports     window length

Run:  .venv/Scripts/python.exe 08_sensitivity_sweep.py
"""

import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

b = importlib.import_module("01_build_windows")
inj = importlib.import_module("05_build_dataset")

FEATS = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
         "spd_mismatch_max", "crab_mean", "prev_silence_s"]
ATTACKS = ["drift", "jump", "gap", "speed_lie", "dark"]

# smaller sample keeps 7 full rebuilds tractable; we are measuring RELATIVE
# stability, not publishing these absolute numbers
inj.N_TRACKS = 2200

DEFAULTS = dict(MIN_MOVING=0.5, GAP_BREAK_S=1800, WINDOW=10)


def run(min_moving, gap_break, window):
    """Rebuild the whole dataset at one setting; return summary metrics."""
    b.MIN_MOVING  = min_moving
    b.GAP_BREAK_S = gap_break
    b.WINDOW      = window
    b.STRIDE      = window          # keep windows non-overlapping
    inj.RNG = np.random.default_rng(42)   # same seed every run

    df = b.load()
    df = b.per_step(df)
    keys = df.groupby(["MMSI", "track_id"]).size()
    keys = keys[keys >= window].index.to_list()
    inj.RNG.shuffle(keys)
    keys = keys[:inj.N_TRACKS]

    n_anom = int(len(keys) * inj.ANOMALY_FRAC)
    w = np.array([inj.YIELD_WEIGHT[a] for a in inj.ATTACKS], float)
    quota = np.floor(n_anom * w / w.sum()).astype(int)
    assign, pos = {}, 0
    for a, q in zip(inj.ATTACKS, quota):
        for k in keys[pos:pos+q]:
            assign[tuple(k)] = a
        pos += q

    df = df.astype({"LAT": "float64", "LON": "float64", "SOG": "float64"})
    groups = {k: g for k, g in df.groupby(["MMSI", "track_id"], sort=False)}
    out = []
    for key in keys:
        t = groups[key].copy().reset_index(drop=True)
        atk, sev = "none", 0.0
        mask = np.zeros(len(t), bool)
        if tuple(key) in assign:
            atk = assign[tuple(key)]
            fn, rng_ = inj.APPLY[atk]
            sev = float(inj.RNG.uniform(*rng_))
            t, mask = fn(t, sev)
            if not mask.any():
                atk, sev = "none", 0.0
        t["anom_row"], t["attack"], t["severity"] = mask, atk, sev
        out.append(t)
    D = pd.concat(out, ignore_index=True)
    D = D.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)
    D = b.per_step(D.drop(columns=[c for c in
        ["dt_s","dist_m","implied_kn","d_sog","d_cog","bearing","spd_mismatch",
         "brg_mismatch","crab","track_id","prev_silence_s"] if c in D.columns]))
    W = b.windows(D)
    W = inj.attach_window_labels(D, W)
    W = W[~((W.spd_mismatch_max > b.MAX_IMPLIED_KN) & (W.label == 0))].reset_index(drop=True)

    res = {"rows": len(df), "vessels": int(W.MMSI.nunique()),
           "windows": len(W), "anom_pct": 100 * W.label.mean()}
    for f in FEATS:
        v = W[f].fillna(W[f].median())
        res[f"auc_{f}"] = roc_auc_score(W.label, v)
    for a in ATTACKS:
        m = (W.attack == a) | (W.label == 0)
        if W.label[m].nunique() < 2:
            res[f"det_{a}"] = np.nan
            continue
        res[f"det_{a}"] = max(
            roc_auc_score(W.label[m], W[f][m].fillna(W[f].median())) for f in FEATS)
    return res


def sweep(param, values):
    print(f"\n{'='*80}\nSWEEPING {param}\n{'='*80}")
    rows = []
    for v in values:
        cfg = dict(DEFAULTS); cfg[param] = v
        print(f"  running {param}={v} ...", flush=True)
        r = run(**{"min_moving": cfg["MIN_MOVING"],
                   "gap_break": cfg["GAP_BREAK_S"],
                   "window": cfg["WINDOW"]})
        r[param] = v
        rows.append(r)
    T = pd.DataFrame(rows).set_index(param)

    print(f"\n  dataset shape:")
    print(T[["rows", "vessels", "windows", "anom_pct"]].round(2).to_string())

    print(f"\n  feature AUC (does the RANKING change?):")
    ac = T[[f"auc_{f}" for f in FEATS]]
    ac.columns = FEATS
    print(ac.round(3).to_string())
    orders = [tuple(ac.loc[i].sort_values(ascending=False).index) for i in ac.index]
    print(f"  ranking identical across all values: {len(set(orders)) == 1}")
    top = {o[0] for o in orders}
    print(f"  top feature always the same: {len(top) == 1}  ({top})")

    print(f"\n  best achievable AUC per attack:")
    dc = T[[f"det_{a}" for a in ATTACKS]]
    dc.columns = ATTACKS
    print(dc.round(3).to_string())
    print(f"  max spread across settings: "
          f"{(dc.max() - dc.min()).max():.3f}")
    return T


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    grid = {"MIN_MOVING": [0.3, 0.5, 1.0],
            "GAP_BREAK_S": [900, 1800, 3600],
            "WINDOW": [5, 10, 20]}
    out = {}
    for k, vals in grid.items():
        if which in ("all", k):
            out[k] = sweep(k, vals)

    for k, T in out.items():
        T.to_csv(f"sensitivity_{k}.csv")
        print(f"wrote sensitivity_{k}.csv")
