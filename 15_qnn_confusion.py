"""
15_qnn_confusion.py
===================
Confusion matrices for the quantum architectures and their classical twins.

*** THESE ARE VALIDATION-SET MATRICES, NOT TEST. ***

The test split has not been opened for any quantum model and must not be until
a configuration is finally selected -- the 12-config grid never completed, so
selection is unfinished. Validation can be looked at repeatedly; test cannot.
Every number here is therefore directly comparable to the validation AUCs in
QNN_RESULTS.md, and NOT to the test numbers in BASELINES.md section 6.

ALERT BUDGET
    Each model flags the top 3.09% of windows by score -- the validation split's
    own anomaly rate (537 of 17,381). Holding the budget fixed is what makes the
    matrices comparable: all six flag exactly 537 windows, and only *which*
    windows differ. Note this is 3.09%, not the 3.5% used for the test-set
    figures, because each split is held to its own true rate.

Seed 0 throughout, as with fig_b1. Seed-to-seed spread is in QNN_RESULTS.md;
a single confusion matrix is a single draw by construction.
"""

import importlib.util
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                              # noqa: E402
import matplotlib as mpl                             # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402
from sklearn.neural_network import MLPClassifier     # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


arch = _load("13_architectures.py", "arch")
ctrl = _load("14_controls.py", "ctrl")

FEATS = arch.FEATS
CFG = {"lr": 0.02, "oversample": None, "decay": 1e-4}   # as in the screen

# Style: same tokens as 11_baseline_viz.py so the figures read as one set.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
BLUES = LinearSegmentedColormap.from_list(
    "seqblue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9.5, "figure.dpi": 150,
})


def main():
    tr = pd.read_csv("dataset_train.csv")
    va = pd.read_csv("dataset_val.csv")
    X, y = tr[FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[FEATS].to_numpy(float), va.label.to_numpy(int)
    Xvj = jnp.array(Xv)
    rate = yv.mean()
    print(f"validation: {len(yv):,} windows, {int(yv.sum())} anomalous "
          f"({rate*100:.2f}%) -- alert budget set to that rate\n")

    panels = []

    # --- quantum architectures -------------------------------------------
    for key, title in (("A_vqc", "A — pure VQC (not hybrid)\nT = 50"),
                       ("E_reup", "E — hybrid + re-uploading\nT = 57"),
                       ("C_pre", "C — hybrid, classical pre-layer\nT = 122")):
        t0 = time.time()
        a = arch.ARCHS[key]
        p, _, predict = arch.train(a, X, y, CFG, 0)
        s = np.asarray(predict(p, Xvj))
        panels.append((title, s, arch.param_count(a)))
        print(f"  {key:<8} AUC {roc_auc_score(yv, s):.4f}  [{time.time()-t0:.0f}s]",
              flush=True)

    # --- the control that matters ----------------------------------------
    t0 = time.time()
    p, predict = ctrl.train_quantum("frozen", X, y, 0)
    s = np.asarray(predict(p, Xvj))
    panels.append(("C frozen — gates never trained\n74 trainable", s, 74))
    print(f"  frozen   AUC {roc_auc_score(yv, s):.4f}  [{time.time()-t0:.0f}s]",
          flush=True)

    # --- classical twins --------------------------------------------------
    for hl, npar, title in (((8, 5), 123, "MLP (8,5) — depth-matched to C\n123 params"),
                            ((5,), 51, "MLP (5,) — matched to A\n51 params")):
        m = MLPClassifier(hidden_layer_sizes=hl, alpha=1e-4,
                          learning_rate_init=0.01, max_iter=500,
                          early_stopping=True, n_iter_no_change=20,
                          random_state=0).fit(X, y)
        s = m.predict_proba(Xv)[:, 1]
        panels.append((title, s, npar))
        print(f"  MLP{str(hl):<7} AUC {roc_auc_score(yv, s):.4f}", flush=True)

    # --- draw --------------------------------------------------------------
    k = max(1, int(round(len(yv) * rate)))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4))
    for ax, (title, s, npar) in zip(axes.ravel(), panels):
        pred = s >= np.partition(s, -k)[-k]
        cm = np.array([[int(((~pred) & (yv == 0)).sum()), int((pred & (yv == 0)).sum())],
                       [int(((~pred) & (yv == 1)).sum()), int((pred & (yv == 1)).sum())]])
        # Coloured by ROW rate, not raw count: counts run from 16,000 to 100, so
        # a count ramp would paint one cell dark and say nothing.
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
        ax.set_yticks([0, 1], ["actually\nnormal", "actually\nanomalous"], fontsize=8.5)
        ax.set_title(f"{title}\nAUC {roc_auc_score(yv, s):.4f}  ·  "
                     f"recall {r[1,1]*100:.1f}%  ·  "
                     f"precision {cm[1,1]/max(1,cm[:,1].sum())*100:.1f}%",
                     fontsize=9.5, color=INK)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.suptitle(
        "VALIDATION-set confusion matrices  ·  17,381 windows, 537 anomalous  ·  "
        "every model flags the same 537\n"
        "Test split is still closed for all quantum models — these are NOT test "
        "results",
        fontsize=11.5, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig("fig_q1_confusion.png", bbox_inches="tight")
    print("\nwrote fig_q1_confusion.png")


if __name__ == "__main__":
    main()
