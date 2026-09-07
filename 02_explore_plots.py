"""
Stage 2: exploratory figures on the clean (unlabelled) windows.

These two answer the only question that can be answered before labels exist:
which of the 13 candidate features are worth spending qubits on?

Run:  .venv/Scripts/python.exe 02_explore_plots.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = pd.read_csv("windows_clean.csv")

FEATURES = ["sog_mean","sog_std","sog_max_jump","cog_turn_max","cog_turn_mean",
            "dt_max","dt_mean","dist_total_m","spd_mismatch_max",
            "spd_mismatch_mean","brg_mismatch_mean","crab_mean","heading_valid_frac"]

# ---------------------------------------------------------------- fig 1: distributions
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for ax, f in zip(axes.ravel(), FEATURES):
    v = W[f].dropna()
    # log scale for the heavily skewed ones so the shape is visible at all
    if v.min() >= 0 and v.max() > 0 and (v.max() / max(v.median(), 1e-9)) > 50:
        ax.hist(np.log10(v + 1e-3), bins=60, color="#2b6cb0", alpha=.85)
        ax.set_xlabel("log10 value")
    else:
        ax.hist(v, bins=60, color="#2b6cb0", alpha=.85)
    ax.set_title(f, fontsize=10)
    ax.tick_params(labelsize=8)
for ax in axes.ravel()[len(FEATURES):]:
    ax.axis("off")
fig.suptitle(f"Candidate feature distributions  (n={len(W):,} windows, "
             f"{W.MMSI.nunique():,} vessels)", fontsize=13)
fig.tight_layout()
fig.savefig("fig1_distributions.png", dpi=140)
print("wrote fig1_distributions.png")

# ---------------------------------------------------------------- fig 2: correlation
C = W[FEATURES].corr(method="spearman")   # spearman: robust to the heavy skew above
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=9)
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        val = C.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(val) > 0.55 else "black")
fig.colorbar(im, ax=ax, shrink=.8, label="Spearman rho")
ax.set_title("Feature redundancy — pairs above |0.8| waste qubit budget", fontsize=12)
fig.tight_layout()
fig.savefig("fig2_correlation.png", dpi=140)
print("wrote fig2_correlation.png")

# ---------------------------------------------------------------- redundant pairs
print("\nredundant pairs (|rho| >= 0.8):")
seen = set()
pairs = []
for i, a in enumerate(FEATURES):
    for j, b in enumerate(FEATURES):
        if i < j and abs(C.iloc[i, j]) >= 0.8:
            pairs.append((abs(C.iloc[i, j]), a, b, C.iloc[i, j]))
for r, a, b, signed in sorted(pairs, reverse=True):
    print(f"  {signed:+.3f}  {a:22s} : {b}")
if not pairs:
    print("  none — all 13 features carry distinct information")

print("\nspread check (features with near-zero variance are useless):")
for f in FEATURES:
    v = W[f].dropna()
    print(f"  {f:22s} cv={v.std()/max(abs(v.mean()),1e-9):6.2f}  "
          f"unique={v.nunique():>7,}  missing={100*W[f].isna().mean():.1f}%")
