"""
Stage 6: baseline views regenerated with anomalies visible, plus a per-feature
audit of which features actually earn their qubit.

Fixes from the first version:
  - prev_silence_s is INCLUDED. Leaving it out made the 'dark' attack look
    undetectable when in fact it is the only feature that detects it.
  - adds univariate AUC per feature, overall and per attack type, so feature
    selection is measured rather than eyeballed.

Run:  .venv/Scripts/python.exe scripts/06_compare_viz.py
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
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
W = pd.read_csv("data/windows_labelled.csv")

FEATS = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
         "spd_mismatch_max", "crab_mean", "prev_silence_s"]
NICE = {"sog_mean": "mean speed (kn)", "sog_max_jump": "max speed jump (kn)",
        "cog_turn_max": "max turn (deg)", "dt_max": "max gap (s)",
        "spd_mismatch_max": "speed/position mismatch (kn)",
        "crab_mean": "course-heading diff (deg)",
        "prev_silence_s": "silence before voyage (s)"}
LOGGED = ["sog_max_jump", "dt_max", "spd_mismatch_max", "cog_turn_max",
          "crab_mean", "prev_silence_s"]
ATK_COL = {"drift": "#d69e2e", "jump": "#c53030", "gap": "#805ad5",
           "speed_lie": "#dd6b20", "dark": "#2f855a"}
ATTACKS = list(ATK_COL)

D = W[FEATS].copy()
D["crab_mean"] = D.crab_mean.fillna(D.crab_mean.median())
y = W.label.values
print(f"windows {len(W):,}   normal {int((y==0).sum()):,}   anomalous {int(y.sum()):,}")

X = D.copy()
for c in LOGGED:
    X[c] = np.log10(X[c].clip(lower=0) + 1e-2)
Xs = StandardScaler().fit_transform(X)

# ---------------------------------------------------------------- feature audit
print("\n" + "="*84)
print("UNIVARIATE AUC — how well does each feature ALONE separate normal from anomalous?")
print("0.50 = useless (coin flip).  1.00 = perfect.  <0.50 means it separates in reverse.")
print("="*84)
rows = []
for f in FEATS:
    v = D[f].values
    overall = roc_auc_score(y, v)
    r = {"feature": f, "overall": overall}
    for a in ATTACKS:
        m = (W.attack.values == a) | (y == 0)      # this attack vs all normals
        r[a] = roc_auc_score(y[m], v[m])
    rows.append(r)
A = pd.DataFrame(rows).set_index("feature")
# report distance from 0.5 so reverse-separating features are not hidden
disp = A.copy()
print(disp.round(3).to_string())
print("\ndiscriminative power = |AUC - 0.5| (higher is better):")
print((A - 0.5).abs().round(3).to_string())

best = (A - 0.5).abs().overall.sort_values(ascending=False)
print("\nranked by overall power:")
for f, v in best.items():
    verdict = "KEEP" if v > 0.03 else "DROP — no better than chance"
    print(f"  {f:20s} {v:.3f}   {verdict}")

# ---------------------------------------------------------------- fig 9: box, split
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, f in zip(axes.ravel(), FEATS):
    a = D[f][y == 0].dropna(); b_ = D[f][y == 1].dropna()
    bp = ax.boxplot([a, b_], widths=.55, showfliers=True, patch_artist=True,
                    tick_labels=["normal", "anomalous"],
                    flierprops=dict(marker=".", ms=2, alpha=.12, mfc="#718096", mec="none"),
                    medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor("#bee3f8"); bp["boxes"][1].set_facecolor("#feb2b2")
    if f in LOGGED:
        ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title(f"{NICE[f]}\nAUC {A.loc[f,'overall']:.3f}", fontsize=10)
    ax.grid(axis="y", alpha=.3)
    ax.text(.02, .97, f"norm {a.median():.2f}\nanom {b_.median():.2f}",
            transform=ax.transAxes, va="top", fontsize=8, family="monospace",
            bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
for ax in axes.ravel()[len(FEATS):]:
    ax.axis("off")
fig.suptitle("AFTER INJECTION — normal vs anomalous, all 7 features  (compare to fig3)", fontsize=13)
fig.tight_layout(); fig.savefig("figures/earlier/fig9_after_box.png", dpi=140); plt.close(fig)
print("\nwrote figures/earlier/fig9_after_box.png")

# ---------------------------------------------------------------- fig 10: PCA
p = PCA(n_components=len(FEATS)).fit(Xs)
Z = p.transform(Xs)
fig = plt.figure(figsize=(16, 5.5))
gs = GridSpec(1, 3, figure=fig)
ax = fig.add_subplot(gs[0])
ax.hexbin(Z[y == 0, 0], Z[y == 0, 1], gridsize=70, cmap="Blues", mincnt=1, bins="log")
ax.set_title("normal only"); ax.set_xlabel(f"PC1 ({100*p.explained_variance_ratio_[0]:.1f}%)")
ax.set_ylabel("PC2")
ax2 = fig.add_subplot(gs[1], sharex=ax, sharey=ax)
ax2.hexbin(Z[y == 0, 0], Z[y == 0, 1], gridsize=70, cmap="Greys", mincnt=1, bins="log", alpha=.55)
ax2.scatter(Z[y == 1, 0], Z[y == 1, 1], s=6, c="#c53030", alpha=.5, edgecolors="none")
ax2.set_title("anomalies overlaid"); ax2.set_xlabel("PC1")
ax3 = fig.add_subplot(gs[2], sharex=ax, sharey=ax)
ax3.hexbin(Z[y == 0, 0], Z[y == 0, 1], gridsize=70, cmap="Greys", mincnt=1, bins="log", alpha=.35)
for a_, c in ATK_COL.items():
    m = (W.attack == a_).values
    ax3.scatter(Z[m, 0], Z[m, 1], s=7, c=c, alpha=.7, edgecolors="none", label=a_)
ax3.legend(markerscale=2.5, fontsize=8, loc="best")
ax3.set_title("by attack type"); ax3.set_xlabel("PC1")
fig.suptitle("AFTER INJECTION — PCA, all 7 features  (compare to fig4)", fontsize=13)
fig.tight_layout(); fig.savefig("figures/earlier/fig10_after_pca.png", dpi=140); plt.close(fig)
print("wrote figures/earlier/fig10_after_pca.png")

c0 = Z[y == 0][:, :2].mean(0)
d0 = np.linalg.norm(Z[y == 0][:, :2] - c0, axis=1)
d1 = np.linalg.norm(Z[y == 1][:, :2] - c0, axis=1)
thr = np.percentile(d0, 99)
print(f"\n  normal centroid distance: median {np.median(d0):.2f}, 99th pct {thr:.2f}")
print(f"  anomalous median {np.median(d1):.2f}")
print(f"  {100*(d1 > thr).mean():.1f}% of anomalies beyond the normal 99th percentile")
for a_ in ATTACKS:
    m = (W.attack == a_).values
    print(f"    {a_:10s} {100*(np.linalg.norm(Z[m][:, :2]-c0, axis=1) > thr).mean():5.1f}%")

# ---------------------------------------------------------------- fig 11: t-SNE
idx = RNG.choice(len(Xs), size=min(6000, len(Xs)), replace=False)
idx = np.unique(np.concatenate([idx, np.where(y == 1)[0][:2500]]))
print(f"\nrunning t-SNE on {len(idx):,} windows ...")
E = TSNE(n_components=2, perplexity=35, init="pca", random_state=0).fit_transform(Xs[idx])
yy = y[idx]
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5))
axes[0].scatter(E[yy == 0, 0], E[yy == 0, 1], s=4, c="#a0aec0", alpha=.35, edgecolors="none")
axes[0].set_title("normal only")
axes[1].scatter(E[yy == 0, 0], E[yy == 0, 1], s=4, c="#cbd5e0", alpha=.3, edgecolors="none")
axes[1].scatter(E[yy == 1, 0], E[yy == 1, 1], s=7, c="#c53030", alpha=.6, edgecolors="none")
axes[1].set_title("anomalies in red")
axes[2].scatter(E[yy == 0, 0], E[yy == 0, 1], s=4, c="#e2e8f0", alpha=.3, edgecolors="none")
for a_, c in ATK_COL.items():
    m = (W.attack.values[idx] == a_)
    axes[2].scatter(E[m, 0], E[m, 1], s=8, c=c, alpha=.75, edgecolors="none", label=a_)
axes[2].legend(markerscale=2.5, fontsize=8)
axes[2].set_title("by attack type")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("AFTER INJECTION — t-SNE, all 7 features  (compare to fig5)", fontsize=13)
fig.tight_layout(); fig.savefig("figures/earlier/fig11_after_tsne.png", dpi=140); plt.close(fig)
print("wrote figures/earlier/fig11_after_tsne.png")

# ---------------------------------------------------------------- fig 12: AUC heatmap
fig, ax = plt.subplots(figsize=(9, 5))
M = A[ATTACKS].values
im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ATTACKS))); ax.set_xticklabels(ATTACKS, rotation=20, ha="right")
ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=9)
for i in range(len(FEATS)):
    for j in range(len(ATTACKS)):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(M[i, j]-.5) > .28 else "black")
fig.colorbar(im, ax=ax, label="AUC (0.5 = no signal)")
ax.set_title("Which feature detects which attack  —  RQ3 table")
fig.tight_layout(); fig.savefig("figures/earlier/fig12_feature_attack_auc.png", dpi=140); plt.close(fig)
print("wrote figures/earlier/fig12_feature_attack_auc.png")
