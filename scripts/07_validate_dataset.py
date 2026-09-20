"""
Stage 7: full integrity check before any modelling.

Every test that could catch a silent corruption. Prints PASS/FAIL per check.
Run this after ANY pipeline change.

Run:  .venv/Scripts/python.exe scripts/07_validate_dataset.py
"""

import numpy as np
import pandas as pd

FEATS = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
         "spd_mismatch_max", "crab_mean", "prev_silence_s",
         # added 2026-09-07 -- these must face the same scaling, leakage and
         # discriminative-power checks as the original seven, or they enter the
         # study less verified than the features they sit beside.
         "course_resid_mean", "mismatch_slope", "straightness", "mismatch_mean",
         "crab_valid"]
ATTACKS = ["drift", "jump", "gap", "speed_lie", "dark"]

tr = pd.read_csv("data/dataset_train.csv")
va = pd.read_csv("data/dataset_val.csv")
te = pd.read_csv("data/dataset_test.csv")
W  = pd.read_csv("data/windows_labelled.csv")
SPLITS = {"train": tr, "val": va, "test": te}

results = []
def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

print("=" * 78)
print("1. LEAKAGE")
print("=" * 78)
tv = set(tr.MMSI) & set(va.MMSI)
tt = set(tr.MMSI) & set(te.MMSI)
vt = set(va.MMSI) & set(te.MMSI)
check("no vessel in train AND val",  len(tv) == 0, f"{len(tv)} shared")
check("no vessel in train AND test", len(tt) == 0, f"{len(tt)} shared")
check("no vessel in val AND test",   len(vt) == 0, f"{len(vt)} shared")

# a window is identified by (MMSI, track_id, t_start); none may repeat across splits
key = lambda d: set(zip(d.MMSI, d.track_id, d.t_start))
kt, kv, ke = key(tr), key(va), key(te)
check("no duplicated window across splits",
      len(kt & kv) == 0 and len(kt & ke) == 0 and len(kv & ke) == 0)
check("no duplicated window WITHIN a split",
      all(len(key(d)) == len(d) for d in SPLITS.values()))

print("\n" + "=" * 78)
print("2. FEATURE VALUES")
print("=" * 78)
for n, d in SPLITS.items():
    v = d[FEATS].values
    check(f"{n}: no NaN", not np.isnan(v).any(), f"{int(np.isnan(v).sum())} NaN")
    check(f"{n}: no inf", np.isfinite(v).all())
    check(f"{n}: within [0, pi]", v.min() >= -1e-9 and v.max() <= np.pi + 1e-9,
          f"min {v.min():.4f} max {v.max():.4f}")

# scaler was fitted on train, so train should span the full range and
# val/test may fall slightly inside it -- but must not be wildly compressed
for n, d in (("val", va), ("test", te)):
    span = (d[FEATS].max() - d[FEATS].min()).min()
    check(f"{n}: no feature collapsed to a constant", span > 1e-6,
          f"narrowest span {span:.4f}")

print("\n" + "=" * 78)
print("3. LABELS")
print("=" * 78)
for n, d in SPLITS.items():
    check(f"{n}: labels are 0/1 only", set(d.label.unique()) <= {0, 1})
    check(f"{n}: both classes present", d.label.nunique() == 2,
          f"{int(d.label.sum())} anomalous of {len(d)}")
    # every anomalous window must name an attack; every normal one must not
    bad_a = ((d.label == 1) & (d.attack == "none")).sum()
    bad_n = ((d.label == 0) & (d.attack != "none")).sum()
    check(f"{n}: label agrees with attack column", bad_a == 0 and bad_n == 0,
          f"{bad_a} anomalous-but-none, {bad_n} normal-but-attacked")

print("\n" + "=" * 78)
print("4. ATTACK COVERAGE")
print("=" * 78)
for n, d in SPLITS.items():
    present = set(d[d.label == 1].attack.unique())
    missing = set(ATTACKS) - present
    check(f"{n}: all 5 attack types present", not missing,
          f"missing {missing}" if missing else "")
    vc = d[d.label == 1].attack.value_counts()
    if len(vc):
        check(f"{n}: attack imbalance under 5x", vc.max()/vc.min() < 5,
              f"{vc.max()/vc.min():.1f}x  ({vc.min()}-{vc.max()})")

print("\n" + "=" * 78)
print("5. SANITY OF THE UNDERLYING (unscaled) DATA")
print("=" * 78)
check("speeds physically plausible", W.sog_mean.between(0, 40).all(),
      f"range {W.sog_mean.min():.2f}-{W.sog_mean.max():.2f} kn")
check("turns within 0-180 deg", W.cog_turn_max.between(0, 180).all(),
      f"max {W.cog_turn_max.max():.1f}")
check("in-window gaps never exceed the 1800s split rule",
      W.dt_max.max() <= 1800.001, f"max {W.dt_max.max():.0f} s")
check("no negative durations", (W.dt_max >= 0).all() and (W.prev_silence_s >= 0).all())
norm_mm = W.loc[W.label == 0, "spd_mismatch_max"]
check("normal-class mismatch stays under the 60 kn cap", norm_mm.max() <= 60,
      f"max {norm_mm.max():.1f} kn")

print("\n" + "=" * 78)
print("6. SIGNAL IS PRESENT (would a model have anything to learn?)")
print("=" * 78)
from sklearn.metrics import roc_auc_score
au = {f: roc_auc_score(W.label, W[f].fillna(W[f].median())) for f in FEATS if f in W}
best_f, best_v = max(au.items(), key=lambda kv: abs(kv[1] - .5))
check("at least one feature beats chance", abs(best_v - .5) > .05,
      f"best is {best_f} at AUC {best_v:.3f}")
for a in ATTACKS:
    m = (W.attack == a) | (W.label == 0)
    aucs = {f: roc_auc_score(W.label[m], W[f][m].fillna(W[f].median())) for f in FEATS if f in W}
    bf, bv = max(aucs.items(), key=lambda kv: abs(kv[1] - .5))
    check(f"{a} is detectable by something", abs(bv - .5) > .03,
          f"{bf} AUC {bv:.3f}")

print("\n" + "=" * 78)
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"RESULT: {n_pass}/{len(results)} checks passed")
if n_pass < len(results):
    print("\nFAILURES:")
    for nm, ok, det in results:
        if not ok:
            print(f"  - {nm}  {det}")
else:
    print("Dataset is ready for modelling.")
print("=" * 78)
