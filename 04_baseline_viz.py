"""
Baseline visualisations of the CLEAN (pre-injection) windows.

Purpose: establish what normal maritime behaviour looks like across every view,
so the same six figures can be regenerated after injection and compared. Any
structure that appears in the 'after' version and not here is the anomaly.

Run:  .venv/Scripts/python.exe 04_baseline_viz.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RNG = np.random.default_rng(0)
W = pd.read_csv("windows_clean.csv")

FEATS = ["sog_mean", "sog_max_jump", "cog_turn_max",
         "dt_max", "spd_mismatch_max", "crab_mean"]
NICE = {"sog_mean": "mean speed (kn)", "sog_max_jump": "max speed jump (kn)",
        "cog_turn_max": "max turn (deg)", "dt_max": "max gap (s)",
        "spd_mismatch_max": "speed/position mismatch (kn)",
        "crab_mean": "course-heading diff (deg)"}

# heavily skewed features get log-transformed for the projection methods
LOGGED = ["sog_max_jump", "dt_max", "spd_mismatch_max", "cog_turn_max", "crab_mean"]

D = W[FEATS].copy()
D["crab_mean"] = D.crab_mean.fillna(D.crab_mean.median())     # 19% missing
X = D.copy()
for c in LOGGED:
    X[c] = np.log10(X[c] + 1e-2)
Xs = StandardScaler().fit_transform(X)

print(f"windows: {len(W):,}   vessels: {W.MMSI.nunique():,}   tracks: {len(W.groupby(['MMSI','track_id'])):,}")

# ---------------------------------------------------------------- fig 3: box plots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, f in zip(axes.ravel(), FEATS):
    v = D[f].dropna()
    ax.boxplot(v, vert=True, widths=.5, showfliers=True,
               flierprops=dict(marker=".", markersize=2, alpha=.15, markerfacecolor="#c53030",
                               markeredgecolor="none"),
               medianprops=dict(color="#c53030", linewidth=2),
               boxprops=dict(facecolor="#bee3f8", color="#2b6cb0"), patch_artist=True)
    if f in LOGGED:
        ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title(NICE[f], fontsize=11)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=.3)
    q1, med, q3 = v.quantile([.25, .5, .75])
    ax.text(.02, .97, f"med {med:.2f}\nIQR {q1:.2f}-{q3:.2f}", transform=ax.transAxes,
            va="top", fontsize=8, family="monospace",
            bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
fig.suptitle("BASELINE (no anomalies) — feature spread", fontsize=13)
fig.tight_layout(); fig.savefig("fig3_baseline_box.png", dpi=140); plt.close(fig)
print("wrote fig3_baseline_box.png")

# ---------------------------------------------------------------- fig 4: PCA
p = PCA(n_components=6).fit(Xs)
Z = p.transform(Xs)
fig = plt.figure(figsize=(15, 5.5))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.3, 1.3, 1])
ax = fig.add_subplot(gs[0])
h = ax.hexbin(Z[:, 0], Z[:, 1], gridsize=70, cmap="Blues", mincnt=1, bins="log")
fig.colorbar(h, ax=ax, label="windows (log)")
ax.set_xlabel(f"PC1 ({100*p.explained_variance_ratio_[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({100*p.explained_variance_ratio_[1]:.1f}%)")
ax.set_title("PCA density — normal behaviour occupies one blob")
ax2 = fig.add_subplot(gs[1])
h2 = ax2.hexbin(Z[:, 0], Z[:, 2], gridsize=70, cmap="Blues", mincnt=1, bins="log")
fig.colorbar(h2, ax=ax2, label="windows (log)")
ax2.set_xlabel("PC1"); ax2.set_ylabel(f"PC3 ({100*p.explained_variance_ratio_[2]:.1f}%)")
ax2.set_title("PC1 vs PC3")
ax3 = fig.add_subplot(gs[2])
ax3.bar(range(1, 7), 100*p.explained_variance_ratio_, color="#2b6cb0")
ax3.plot(range(1, 7), 100*np.cumsum(p.explained_variance_ratio_), "o-", color="#c53030")
ax3.set_xlabel("component"); ax3.set_ylabel("% variance")
ax3.set_title("scree — how many dimensions really matter")
ax3.grid(alpha=.3)
for i, c in enumerate(np.cumsum(p.explained_variance_ratio_)):
    ax3.text(i+1, 100*c+2, f"{100*c:.0f}", ha="center", fontsize=8)
fig.suptitle("BASELINE — PCA structure", fontsize=13)
fig.tight_layout(); fig.savefig("fig4_baseline_pca.png", dpi=140); plt.close(fig)
print("wrote fig4_baseline_pca.png")
print("  PCA loadings (PC1):")
for f, l in sorted(zip(FEATS, p.components_[0]), key=lambda t: -abs(t[1])):
    print(f"    {f:20s} {l:+.3f}")

# ---------------------------------------------------------------- fig 5: t-SNE
n = 4000
idx = RNG.choice(len(Xs), size=min(n, len(Xs)), replace=False)
print(f"\nrunning t-SNE on {len(idx):,} sampled windows ...")
E = TSNE(n_components=2, perplexity=35, init="pca", random_state=0).fit_transform(Xs[idx])
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].scatter(E[:, 0], E[:, 1], s=4, alpha=.35, c="#2b6cb0", edgecolors="none")
axes[0].set_title("t-SNE — normal windows only")
for ax, f in zip(axes[1:], ["sog_mean", "spd_mismatch_max"]):
    v = D[f].values[idx]
    c = np.log10(v + 1e-2) if f in LOGGED else v
    s = ax.scatter(E[:, 0], E[:, 1], s=4, c=c, cmap="viridis", alpha=.6, edgecolors="none")
    fig.colorbar(s, ax=ax, label=("log10 " if f in LOGGED else "") + NICE[f])
    ax.set_title(f"coloured by {NICE[f]}")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("BASELINE — t-SNE embedding", fontsize=13)
fig.tight_layout(); fig.savefig("fig5_baseline_tsne.png", dpi=140); plt.close(fig)
print("wrote fig5_baseline_tsne.png")

# ---------------------------------------------------------------- fig 6: pair scatter
fig, axes = plt.subplots(6, 6, figsize=(16, 16))
sub = RNG.choice(len(D), size=min(8000, len(D)), replace=False)
for i, fi in enumerate(FEATS):
    for j, fj in enumerate(FEATS):
        ax = axes[i, j]
        if i == j:
            v = X[fi].values[sub]
            ax.hist(v, bins=45, color="#2b6cb0", alpha=.85)
        else:
            ax.scatter(X[fj].values[sub], X[fi].values[sub], s=1.5, alpha=.12,
                       c="#2b6cb0", edgecolors="none")
        if i == 5: ax.set_xlabel(fj, fontsize=8)
        if j == 0: ax.set_ylabel(fi, fontsize=8)
        ax.tick_params(labelsize=6)
fig.suptitle("BASELINE — pairwise structure (log-scaled where skewed)", fontsize=14)
fig.tight_layout(); fig.savefig("fig6_baseline_pairs.png", dpi=110); plt.close(fig)
print("wrote fig6_baseline_pairs.png")

# ---------------------------------------------------------------- fig 7: geography + context
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
h = axes[0].hexbin(W.lon_mean, W.lat_mean, gridsize=110, cmap="magma_r", mincnt=1, bins="log")
fig.colorbar(h, ax=axes[0], label="windows (log)")
axes[0].set_xlabel("longitude"); axes[0].set_ylabel("latitude")
axes[0].set_title("where the windows are")
vt = W.vessel_type.dropna().astype(int)
bins = {"fishing (30)": (30, 30), "tug/pilot (31-52)": (31, 52), "passenger (60-69)": (60, 69),
        "cargo (70-79)": (70, 79), "tanker (80-89)": (80, 89), "other": (0, 29)}
counts = {k: int(((vt >= a) & (vt <= b_)).sum()) for k, (a, b_) in bins.items()}
axes[1].barh(list(counts), list(counts.values()), color="#2b6cb0")
axes[1].set_title("vessel types"); axes[1].set_xlabel("windows")
axes[1].grid(axis="x", alpha=.3)
axes[2].hist(W.sog_mean, bins=70, color="#2b6cb0", alpha=.85)
axes[2].set_xlabel("mean speed in window (kn)"); axes[2].set_ylabel("windows")
axes[2].set_title("speed profile of transiting traffic")
axes[2].grid(alpha=.3)
fig.suptitle("BASELINE — dataset context", fontsize=13)
fig.tight_layout(); fig.savefig("fig7_baseline_context.png", dpi=140); plt.close(fig)
print("wrote fig7_baseline_context.png")

# ---------------------------------------------------------------- fig 8: real tracks
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
keys = list(W.groupby(["MMSI", "track_id"]).groups.keys())
pick = [keys[i] for i in RNG.choice(len(keys), 8, replace=False)]
for ax, k in zip(axes.ravel(), pick):
    t = W[(W.MMSI == k[0]) & (W.track_id == k[1])].sort_values("t_start")
    ax.plot(t.lon_mean, t.lat_mean, "-o", ms=2.5, lw=1, color="#2b6cb0")
    ax.scatter(t.lon_mean.iloc[0], t.lat_mean.iloc[0], c="green", s=35, zorder=3, label="start")
    ax.set_title(f"MMSI {k[0]}  ({len(t)} windows)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(useOffset=False, style="plain")
fig.suptitle("BASELINE — eight real vessel tracks (window centroids)", fontsize=13)
fig.tight_layout(); fig.savefig("fig8_baseline_tracks.png", dpi=140); plt.close(fig)
print("wrote fig8_baseline_tracks.png")

print("\nbaseline summary (median / IQR):")
for f in FEATS:
    q1, m, q3 = D[f].quantile([.25, .5, .75])
    print(f"  {f:20s} {m:9.2f}   [{q1:.2f} - {q3:.2f}]")
