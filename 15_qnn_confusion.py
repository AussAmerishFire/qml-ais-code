"""
15_qnn_confusion.py
===================
Confusion matrices for the three current quantum architectures and their
depth- and parameter-matched classical twins.

RUN THIS AFTER 19_final.py -- it trains six models and would otherwise compete
with that run for the same CPU cores.

*** VALIDATION-SET MATRICES. *** 19_final.py owns the test-set numbers; this
file exists to show the SHAPE of each model's errors, which a single confusion
matrix can do and an aggregate AUC cannot.

Architectures come from 16_sweep.py, the script that actually produced the
results. An earlier version imported them from 13_architectures.py -- the
superseded 8-feature screen -- and crashed with "Features must be of length 8
or less; got length 12", because those variants feed features straight into
AngleEmbedding with no classical pre-layer. That is the same failure mode as
09_independent_check.py importing 01_build_windows: a file pointing at
definitions that something else replaced.

Seed 0 throughout. A confusion matrix is one draw by construction, so it is
useful for reading error structure and USELESS for ranking models -- the
ranking lives in RESULTS_SWEEP.md and final_results.csv, at 5 and 10 seeds.
"""

import importlib.util

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sw = _load("16_sweep.py", "sw")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
BLUES = LinearSegmentedColormap.from_list(
    "seqblue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9.5, "figure.dpi": 150,
})

TITLE = {
    "Q_pre":      "Q_pre — Dense(12→8) → circuit",
    "MLP_pre":    "MLP(8,5) — depth-matched twin",
    "Q_reup":     "Q_reup — + data re-uploading",
    "MLP_reup":   "MLP(8,6) — depth-matched twin",
    "Q_frozen":   "Q_frozen — gates NEVER trained",
    "MLP_frozen": "MLP(6,4) — depth-matched twin",
}
ORDER = ["Q_pre", "MLP_pre", "Q_reup", "MLP_reup", "Q_frozen", "MLP_frozen"]


def main():
    import json
    tr = pd.read_csv("dataset_train.csv")
    va = pd.read_csv("dataset_val.csv")
    X, y = tr[sw.FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[sw.FEATS].to_numpy(float), va.label.to_numpy(int)
    best = json.load(open("sweep_best_cfg.json"))
    rate = yv.mean()
    print(f"validation {len(yv):,} windows, {int(yv.sum())} anomalous "
          f"({rate*100:.2f}%) -- alert budget set to that rate\n", flush=True)

    scores = {}
    for name in ORDER:
        if name in sw.QARCH:
            cfg = sw.QGRID[best[name]]
            p, pred = sw.q_train(sw.QARCH[name], X, y, cfg, 0,
                                 sw.SWEEP_STEPS, None)
            s = np.asarray(pred(p, jnp.array(Xv)))
            T = sw.q_params(sw.QARCH[name])
        else:
            cfg = sw.CGRID[best[name]]
            m = sw.c_train(sw.CARCH[name], X, y, cfg, 0, sw.SWEEP_STEPS, False)
            s = m.predict_proba(Xv)[:, 1]
            T = sw.CPARAMS[name]
        scores[name] = (s, T)
        print(f"  {name:<11} T={T:>4}  AUC {roc_auc_score(yv, s):.4f}", flush=True)

    k = max(1, int(round(len(yv) * rate)))
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 11))
    for ax, name in zip(axes.ravel(), ORDER):
        s, T = scores[name]
        pred = s >= np.partition(s, -k)[-k]
        cm = np.array([[int(((~pred) & (yv == 0)).sum()),
                        int((pred & (yv == 0)).sum())],
                       [int(((~pred) & (yv == 1)).sum()),
                        int((pred & (yv == 1)).sum())]])
        # Coloured by ROW rate: raw counts span 16,000 to 100, so a count ramp
        # would paint one cell dark and say nothing.
        r = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(r, cmap=BLUES, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                c = "white" if r[i, j] > 0.55 else INK
                ax.text(j, i - 0.13, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=c)
                ax.text(j, i + 0.19, f"{r[i, j]*100:.1f}% of row", ha="center",
                        va="center", fontsize=8,
                        color="white" if r[i, j] > 0.55 else INK2)
        ax.set_xticks([0, 1], ["not flagged", "flagged"], fontsize=8.5)
        ax.set_yticks([0, 1], ["actually\nnormal", "actually\nanomalous"],
                      fontsize=8.5)
        ax.set_title(f"{TITLE[name]}   T={T}\nAUC {roc_auc_score(yv, s):.4f}  ·  "
                     f"recall {r[1,1]*100:.1f}%  ·  "
                     f"precision {cm[1,1]/max(1,cm[:,1].sum())*100:.1f}%",
                     fontsize=9.5, color=INK)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.suptitle(
        "Validation confusion matrices, 12 features  ·  17,381 windows, "
        "537 anomalous\nEach quantum model sits beside its depth- and "
        "parameter-matched classical twin  ·  seed 0",
        fontsize=11.5, fontweight="bold", color=INK, y=1.005)
    fig.tight_layout()
    fig.savefig("fig_q1_confusion.png", bbox_inches="tight")
    print("\nwrote fig_q1_confusion.png")


if __name__ == "__main__":
    main()
