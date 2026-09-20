"""
18_sweep_viz.py
===============
The headline figure: validation AUC against training-set size, and the
quantum-minus-classical gap that inverts across it.

Colour: dataviz reference palette, categorical slots, validated light mode.
Quantum models are drawn solid, their matched classical twins dashed in the
SAME hue -- the pairing is the point of the figure, so the pair shares a colour
and is separated by line style rather than being given two unrelated hues.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e8e7e3"
mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 9.5, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 8.5,
    "font.size": 9.5, "figure.dpi": 150,
})

PAIRS = [("Q_pre", "MLP_pre", "#2a78d6", "pre-layer hybrid"),
         ("Q_reup", "MLP_reup", "#eb6834", "re-uploading hybrid"),
         ("Q_frozen", "MLP_frozen", "#1baf7a", "frozen-gate hybrid")]
REFS = [("iforest", "#e87ba4", "Isolation Forest (0 labels)"),
        ("rf", "#008300", "Random forest (all 12 features)")]


def tidy(ax):
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    d = pd.read_csv("sweep_results.csv")
    p2 = d[d.phase == 2]
    Ns = sorted(p2.N.unique())
    T = p2.groupby("model").T.first()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # ---- panel 1: absolute performance --------------------------------------
    ax = axes[0]
    for q, c, col, lab in PAIRS:
        for m, ls, mk in ((q, "-", "o"), (c, "--", "s")):
            mu = [p2[(p2.model == m) & (p2.N == n)].auc.mean() for n in Ns]
            sd = [p2[(p2.model == m) & (p2.N == n)].auc.std() for n in Ns]
            ax.errorbar(Ns, mu, yerr=sd, color=col, ls=ls, marker=mk, ms=5,
                        lw=2, capsize=3, elinewidth=1,
                        label=f"{'quantum' if ls=='-' else 'classical'} — {lab}")
    for m, col, lab in REFS:
        mu = [p2[(p2.model == m) & (p2.N == n)].auc.mean() for n in Ns]
        ax.plot(Ns, mu, color=col, ls=":", lw=2.5, marker="^", ms=5, label=lab)

    ax.set_xscale("log")
    ax.set_xticks(Ns, [f"{n:,}" for n in Ns])
    ax.set_xlabel("Training-set size N (log scale, 3.5% anomalies throughout)")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_title("Every supervised model collapses as data shrinks.\n"
                 "The zero-label baseline does not.")
    ax.legend(loc="lower right", ncol=1)
    tidy(ax)

    # ---- panel 2: the gap, and its inversion ---------------------------------
    ax = axes[1]
    for q, c, col, lab in PAIRS:
        diff = []
        for n in Ns:
            a = p2[(p2.model == q) & (p2.N == n)].sort_values("seed").auc.values
            b = p2[(p2.model == c) & (p2.N == n)].sort_values("seed").auc.values
            k = min(len(a), len(b))
            diff.append(a[:k].mean() - b[:k].mean())
        ax.plot(Ns, diff, color=col, lw=2.5, marker="o", ms=6,
                label=f"{lab}  (T={int(T[q])} vs {int(T[c])} params)")
    ax.axhline(0, color=INK, lw=1.2)
    ax.axhspan(-0.16, 0, color="#e34948", alpha=0.06)
    ax.axhspan(0, 0.09, color="#2a78d6", alpha=0.06)
    ax.text(600, 0.055, "quantum ahead", fontsize=9, color="#2a78d6",
            fontweight="bold")
    ax.text(600, -0.13, "classical ahead", fontsize=9, color="#e34948",
            fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(Ns, [f"{n:,}" for n in Ns])
    ax.set_xlabel("Training-set size N (log scale)")
    ax.set_ylabel("Quantum AUC − matched classical AUC")
    ax.set_title("The gap inverts — and it runs BACKWARDS\n"
                 "to the generalisation bound's prediction")
    ax.legend(loc="lower right")
    tidy(ax)

    fig.suptitle(
        "Quantum models need MORE data, not less  ·  5 seeds per point, "
        "depth- and parameter-matched twins",
        fontsize=12.5, fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, -0.04,
             "Caro et al. (2022) bound the generalisation gap as sqrt(T/N), "
             "predicting quantum models should hold up BETTER as N falls. "
             "All three\nmatched pairs show the opposite, monotonically. "
             "Below N ≈ 3,000, an unsupervised method using no labels at all "
             "beats every supervised model.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig("fig_s1_scarcity.png", bbox_inches="tight")
    print("wrote fig_s1_scarcity.png")


if __name__ == "__main__":
    main()
