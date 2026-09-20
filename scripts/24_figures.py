"""
scripts/24_figures.py
=============
Every figure the study's COMPLETED data supports, written to figures/.

    python scripts/24_figures.py

SOURCES -- completed files only
    results/sweep_results.csv           scarcity sweep, validation, 5 seeds, 5 sizes
    results/baseline_sweep_results.csv  classical baselines, same protocol
    results/final_results.csv           endpoint runs, validation + test, 10 seeds
    results/qsvm_results.csv            Quantum Support Vector Machine, validation
    results/baseline_results.csv        classical baselines, test set, full data
    dataset_*.csv               the data itself

results/full_sweep_results.csv is deliberately NOT used: it is still being written, and a
figure built on a half-finished file would silently average an incomplete set of
seeds. Figures that will change once it lands are marked in figures/README.md.

No model is trained here -- the sweep holds the CPU.

COLOUR
    dataviz reference palette, validated light mode, fixed slot order, never
    cycled. Each model owns one hue in EVERY figure. A hybrid quantum network and
    its matched classical twin share a hue -- the pairing is the point of the
    study -- and are separated by line style: solid = quantum, dashed = twin.
"""

import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ style
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e8e7e3"
C = {  # entity -> hue, fixed for every figure
    "Q_pre": "#2a78d6", "MLP_pre": "#2a78d6",
    "Q_reup": "#eb6834", "MLP_reup": "#eb6834",
    "Q_frozen": "#1baf7a", "MLP_frozen": "#1baf7a",
    "QSVM": "#4a3aa7",
    "gbt": "#eda100", "rf": "#008300", "logreg": "#e87ba4", "mlp": "#e34948",
    "iforest": "#52514e", "rule": "#8a8880",
}
NAME = {
    "Q_pre": "HQNN · classical pre-layer", "MLP_pre": "Matched MLP · pre-layer twin",
    "Q_reup": "HQNN · data re-uploading", "MLP_reup": "Matched MLP · re-uploading twin",
    "Q_frozen": "HQNN · frozen circuit", "MLP_frozen": "Matched MLP · frozen twin",
    "QSVM": "Quantum Support Vector Machine",
    "gbt": "Gradient Boosting", "rf": "Random Forest", "logreg": "Logistic Regression",
    "mlp": "Multi-Layer Perceptron (71 params)", "iforest": "Isolation Forest (no labels)",
    "rule": "Single-Threshold Rule (no labels)",
}
DASH = {m: "--" for m in ("MLP_pre", "MLP_reup", "MLP_frozen")}
DASH.update({"iforest": ":", "rule": ":"})
TWIN = {"Q_pre": "MLP_pre", "Q_reup": "MLP_reup", "Q_frozen": "MLP_frozen"}
NS = [571, 1429, 2857, 14286, 68296]
ATT = ["speed_lie", "jump", "gap", "dark", "drift"]
ATT_LAB = {"speed_lie": "Speed\nfalsification", "jump": "Position\njump",
           "gap": "Reporting\ngap", "dark": "Transponder\nswitch-off", "drift": "Gradual\ndrift"}
BLUES = LinearSegmentedColormap.from_list(
    "b", ["#f3f7fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("d", ["#b3342f", "#e34948", "#f4b5ae", "#f0efec",
                                                  "#a9c9f2", "#2a78d6", "#184f95"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "grid.linestyle": "-", "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False,
    "legend.fontsize": 8.5, "font.size": 9.5, "figure.dpi": 150,
})


def tidy(ax):
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def logx(ax):
    ax.set_xscale("log")
    ax.set_xticks(NS, [f"{n:,}" for n in NS])
    ax.minorticks_off()


def save(fig, name, source):
    # below the figure's bottom edge, so it can never sit on tick labels;
    # bbox_inches="tight" grows the canvas to include it
    fig.text(0.995, -0.01, f"Source: {source}", ha="right", va="top",
             fontsize=7, color=INK3)
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ------------------------------------------------------------------ data
sw = pd.read_csv("results/sweep_results.csv"); sw = sw[sw.phase == 2]
bs = pd.read_csv("results/baseline_sweep_results.csv")
fin = pd.read_csv("results/final_results.csv")
qs = pd.read_csv("results/qsvm_results.csv")
bres = pd.read_csv("results/baseline_results.csv").set_index("model")

VAL = pd.concat([sw[["model", "N", "seed", "auc"]],
                 bs[["model", "N", "seed", "auc"]],
                 qs.rename(columns={"val_auc": "auc"}).assign(model="QSVM")[["model", "N", "seed", "auc"]]])


def curve(m):
    g = VAL[VAL.model == m].groupby("N").auc
    mu, se = g.mean(), g.std() / np.sqrt(g.count())
    return mu.index.values, mu.values, se.fillna(0).values


# ================================================================== 1. pipeline
def fig_pipeline():
    # values from logs/pipeline_v2.log (STAGE 1); every bar counts reports, not vessels
    stages = [("Raw AIS reports", 8_156_509), ("After AIS validity filters", 6_094_912),
              ("Under way (moored removed)", 1_463_926), ("From vessels with ≥ 200 reports", 1_256_434)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    y = np.arange(len(stages))[::-1]
    vals = [v for _, v in stages]
    ax.barh(y, vals, color="#2a78d6", height=0.62)
    for yi, (lab, v) in zip(y, stages):
        ax.text(v + 90_000, yi, f"{v:,}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [s for s, _ in stages])
    ax.set_xlim(0, 10_200_000)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    ax.set_xlabel("AIS position reports")
    ax.set_title("From 8.2 million reports to 2,441 vessels\n"
                 "(2,435 of them yield at least one usable window)", fontsize=10.5)
    ax.grid(axis="y", visible=False); tidy(ax)

    ax = axes[1]
    splits = [("Train", 68_296, 2_420, 1_462), ("Validation", 17_381, 537, 364), ("Test", 29_292, 1_035, 609)]
    x = np.arange(3)
    ax.bar(x, [s[1] - s[2] for s in splits], color="#9ec5f4", width=0.58, label="normal windows")
    ax.bar(x, [s[2] for s in splits], bottom=[s[1] - s[2] for s in splits], color="#e34948",
           width=0.58, label="anomalous windows")
    for xi, s in zip(x, splits):
        ax.text(xi, s[1] + 1400, f"{s[1]:,} windows\n{s[2]:,} anomalous ({s[2]/s[1]*100:.2f}%)\n"
                f"{s[3]:,} vessels", ha="center", va="bottom", fontsize=8.2, color=INK)
    ax.set_xticks(x, [s[0] for s in splits])
    ax.set_ylim(0, 92_000)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax.set_ylabel("10-report windows")
    ax.set_title("114,969 windows, split by vessel")
    ax.legend(loc="upper right"); ax.grid(axis="x", visible=False); tidy(ax)
    fig.suptitle("The dataset  ·  zero vessels appear in more than one split", fontsize=12.5,
                 fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "01_data_pipeline.png", "scripts/PIPELINE.py run log")


# ================================================================== 2. attacks
def fig_attacks():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    splits = {"Train": pd.read_csv("data/dataset_train.csv"), "Validation": pd.read_csv("data/dataset_val.csv"),
              "Test": pd.read_csv("data/dataset_test.csv")}
    order = ["speed_lie", "jump", "gap", "dark", "drift"]
    width = 0.26
    shades = {"Train": "#184f95", "Validation": "#3987e5", "Test": "#9ec5f4"}
    x = np.arange(len(order))
    for i, (sp, d) in enumerate(splits.items()):
        a = d[d.label == 1].attack.value_counts(normalize=True)
        v = [a.get(k, 0) * 100 for k in order]
        ax.bar(x + (i - 1) * width, v, width=width - 0.03, color=shades[sp], label=sp)
    ax.set_xticks(x, [ATT_LAB[k].replace("\n", " ") for k in order])
    ax.set_ylabel("Share of that split's anomalies (%)")
    ax.set_title("Attack mix per split — gradual drift is the largest class, and the hardest")
    ax.legend(title="Split", title_fontsize=8.5); ax.grid(axis="x", visible=False); tidy(ax)
    fig.tight_layout()
    save(fig, "02_attack_composition.png", "dataset_train/val/test.csv")


# ================================================================== 3. features
def fig_features():
    tr = pd.read_csv("data/dataset_train.csv")
    feats = ["spd_mismatch_max", "mismatch_mean", "course_resid_mean", "dt_max", "sog_max_jump",
             "cog_turn_max", "prev_silence_s", "crab_mean", "mismatch_slope", "straightness",
             "sog_mean", "crab_valid"]
    lab = {"spd_mismatch_max": "Speed–position mismatch (max)", "mismatch_mean": "Speed–position mismatch (mean) †",
           "course_resid_mean": "Course vs bearing travelled †", "dt_max": "Longest reporting gap",
           "sog_max_jump": "Largest speed jump", "cog_turn_max": "Sharpest turn",
           "prev_silence_s": "Silence before window", "crab_mean": "Mean crab angle",
           "mismatch_slope": "Mismatch trend across window †", "straightness": "Path straightness †",
           "sog_mean": "Mean speed", "crab_valid": "Crab angle available"}
    cols = ["overall"] + ATT
    M = np.zeros((len(feats), len(cols)))
    for i, f in enumerate(feats):
        for j, c in enumerate(cols):
            sel = tr if c == "overall" else tr[(tr.label == 0) | (tr.attack == c)]
            a = roc_auc_score(sel.label, sel[f])
            M[i, j] = max(a, 1 - a)
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.imshow(M, cmap=BLUES, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(feats)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] > 0.78 else INK)
    ax.set_xticks(range(len(cols)), ["All\nattacks"] + [ATT_LAB[a] for a in ATT], fontsize=8.5)
    ax.set_yticks(range(len(feats)), [lab[f] for f in feats], fontsize=8.8)
    ax.xaxis.tick_top(); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("How well each feature separates each attack on its own (univariate AUC)\n"
                 "0.50 = no information · † = added mid-study", pad=44)
    fig.tight_layout()
    save(fig, "03_feature_separability.png", "data/dataset_train.csv, 68,296 windows")


# ================================================================== 4. learning curves
def fig_curves():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), sharey=True)
    panels = [
        ("Hybrid quantum networks vs their matched classical twins",
         ["Q_pre", "MLP_pre", "Q_reup", "MLP_reup", "Q_frozen", "MLP_frozen"]),
        ("Other classical models  ·  MLP here is 12-5-1, 71 parameters", ["gbt", "rf", "logreg", "mlp"]),
        ("Quantum kernel vs the best variational quantum model at N = 571", ["QSVM", "Q_reup"]),
        ("No-label methods, with every other model shown in grey", ["iforest", "rule"]),
    ]
    for ax, (title, models) in zip(axes.ravel(), panels):
        if "grey" in title:
            for m in C:
                if m in ("iforest", "rule"):
                    continue
                x, mu, _ = curve(m)
                ax.plot(x, mu, color="#d8d6d0", lw=1.2, zorder=1)
        for m in models:
            x, mu, se = curve(m)
            ls = DASH.get(m, "-")
            ax.fill_between(x, mu - se, mu + se, color=C[m], alpha=0.10, lw=0, zorder=2)
            ax.plot(x, mu, color=C[m], ls=ls, lw=2.2, marker="o", ms=5,
                    mec=SURFACE, mew=1.2, label=NAME[m], zorder=3)
        if "kernel" in title:
            ax.text(2000, 0.60, "not computable beyond N = 1,429\n"
                    "on this machine (kernel ∝ N²)", fontsize=8.5, color=C["QSVM"])
        logx(ax); ax.set_title(title, fontsize=10.5); ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(0.50, 0.92); tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel("Training-set size N (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation AUC")
    fig.suptitle("Learning curves for all 13 models  ·  anomaly rate fixed at 3.5%  ·  "
                 "bands = ±1 standard error", fontsize=12.5, fontweight="bold", color=INK, y=1.005)
    fig.tight_layout()
    save(fig, "04_learning_curves.png", "results/sweep_results.csv, results/baseline_sweep_results.csv, results/qsvm_results.csv · 5 seeds")


# ================================================================== 5. inversion
def fig_inversion():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for q, c in TWIN.items():
        d = []
        for n in NS:
            a = sw[(sw.model == q) & (sw.N == n)].sort_values("seed").auc.values
            b = sw[(sw.model == c) & (sw.N == n)].sort_values("seed").auc.values
            diff = a - b
            d.append((diff.mean(), diff.std() / np.sqrt(len(diff))))
        mu = np.array([x[0] for x in d]); se = np.array([x[1] for x in d])
        ax.fill_between(NS, mu - se, mu + se, color=C[q], alpha=0.12, lw=0)
        ax.plot(NS, mu, color=C[q], lw=2.4, marker="o", ms=6, mec=SURFACE, mew=1.3, label=NAME[q])
    ax.axhline(0, color=INK, lw=1.2)
    ax.axhspan(0, 0.12, color="#2a78d6", alpha=0.05, zorder=0)
    ax.axhspan(-0.19, 0, color="#e34948", alpha=0.05, zorder=0)
    ax.text(620, 0.09, "quantum ahead", color="#184f95", fontsize=9, fontweight="bold")
    ax.text(620, -0.17, "classical twin ahead", color="#b3342f", fontsize=9, fontweight="bold")
    logx(ax); ax.set_ylim(-0.19, 0.12)
    ax.set_xlabel("Training-set size N (log scale)"); ax.set_ylabel("Quantum AUC − matched twin AUC")
    ax.set_title("Across all five sizes (validation, 5 seeds)", fontsize=10.5)
    ax.legend(loc="lower right"); tidy(ax)

    ax = axes[1]
    xs = {571: 0, 68296: 1}
    off = {"Q_pre": -0.18, "Q_reup": 0.0, "Q_frozen": 0.18}
    for q, c in TWIN.items():
        for n in (571, 68296):
            a = fin[(fin.model == q) & (fin.N == n)].sort_values("seed").test_auc.values
            b = fin[(fin.model == c) & (fin.N == n)].sort_values("seed").test_auc.values
            diff = a - b
            ci = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            p = stats.wilcoxon(a, b).pvalue
            x = xs[n] + off[q]
            ax.errorbar(x, diff.mean(), yerr=ci, fmt="D", ms=7, color=C[q], mec=SURFACE,
                        mew=1.2, ecolor=C[q], elinewidth=1.6, capsize=4,
                        label=NAME[q] if n == 571 else None)
            # label on the far side of the interval from zero, so it never sits on the zero line
            below = diff.mean() < 0
            ax.text(x, diff.mean() - ci - 0.006 if below else diff.mean() + ci + 0.006,
                    f"p={p:.3f}" + (" ✔" if p < 0.05 else ""), ha="center",
                    va="top" if below else "bottom", fontsize=7.6, color=INK2)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks([0, 1], ["N = 571", "N = 68,296"]); ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.19, 0.12)
    ax.set_title("At the endpoints (test set, 10 seeds, 95% CI)", fontsize=10.5)
    ax.grid(axis="x", visible=False); tidy(ax)
    fig.suptitle("The gap inverts: quantum trails its twin with scarce data and leads with plenty\n"
                 "significant at both ends for 2 of 3 architectures (re-uploading: neither end)  ·  "
                 "opposite to the √(T/N) inference", fontsize=12, fontweight="bold",
                 color=INK, y=1.06)
    fig.tight_layout()
    save(fig, "05_inversion.png", "results/sweep_results.csv (left) · results/final_results.csv (right) · Wilcoxon signed-rank")


# ================================================================== 6. where quantum wins
def fig_where():
    quantum = ["Q_reup", "Q_pre", "Q_frozen", "QSVM"]
    classical = ["gbt", "rf", "logreg", "mlp", "MLP_reup", "MLP_pre", "MLP_frozen", "iforest", "rule"]
    short = {"gbt": "Gradient\nBoosting", "rf": "Random\nForest", "logreg": "Logistic\nRegression",
             "mlp": "MLP", "MLP_reup": "Twin ·\nre-upload", "MLP_pre": "Twin ·\npre-layer",
             "MLP_frozen": "Twin ·\nfrozen", "iforest": "Isolation\nForest", "rule": "Threshold\nRule"}
    mean = VAL.groupby(["model", "N"]).auc.mean()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    norm = TwoSlopeNorm(vmin=-0.30, vcenter=0, vmax=0.12)
    for ax, n in zip(axes, (571, 2857, 68296)):
        M = np.full((len(quantum), len(classical)), np.nan)
        for i, q in enumerate(quantum):
            for j, c in enumerate(classical):
                if (q, n) in mean.index and (c, n) in mean.index:
                    M[i, j] = mean[(q, n)] - mean[(c, n)]
        ax.imshow(np.ma.masked_invalid(M), cmap=DIVERGE, norm=norm, aspect="auto")
        for i in range(len(quantum)):
            for j in range(len(classical)):
                v = M[i, j]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#efeeea"))
                    ax.text(j, i, "not\nrun", ha="center", va="center", fontsize=7, color=INK3)
                else:
                    # 3 decimals: at 2, a +0.004 lead printed as a bold "+0.00"
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8.2,
                            fontweight="bold" if v > 0 else "normal",
                            color="white" if abs(v) > 0.17 else INK)
        wins = int(np.nansum(M > 0)); tot = int(np.sum(~np.isnan(M)))
        ax.set_title(f"N = {n:,}  ·  quantum ahead in {wins} of {tot} comparisons", fontsize=10.5)
        ax.set_xticks(range(len(classical)), [short[c] for c in classical], fontsize=7.8)
        ax.set_yticks(range(len(quantum)), [NAME[q] for q in quantum] if n == 571 else [""] * 4, fontsize=8.6)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Where quantum is better: quantum AUC minus classical AUC  ·  blue = quantum ahead, "
                 "red = classical ahead", fontsize=12.5, fontweight="bold", color=INK, y=1.04)
    fig.tight_layout()
    save(fig, "06_where_quantum_wins.png", "validation AUC, 5 seeds · sweep + baseline sweep + QSVM")


# ================================================================== 7. frozen vs trained
def fig_frozen():
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    d, ps = [], []
    for n in NS:
        a = sw[(sw.model == "Q_frozen") & (sw.N == n)].sort_values("seed").auc.values
        b = sw[(sw.model == "Q_pre") & (sw.N == n)].sort_values("seed").auc.values
        diff = a - b
        d.append((diff.mean(), diff.std() / np.sqrt(len(diff))))
        ps.append(stats.wilcoxon(a, b).pvalue)
    mu = np.array([x[0] for x in d]); se = np.array([x[1] for x in d])
    ax.fill_between(NS, mu - se, mu + se, color="#1baf7a", alpha=0.14, lw=0)
    ax.plot(NS, mu, color="#1baf7a", lw=2.5, marker="o", ms=7, mec=SURFACE, mew=1.3,
            label="Measured: frozen-circuit HQNN − trained HQNN")
    pred = np.sqrt(154 / np.array(NS)) - np.sqrt(106 / np.array(NS))
    ax.plot(NS, pred, color=INK3, lw=1.8, ls="--",
            label="Direction the √(T/N) inference predicts: √(154/N) − √(106/N)  (scale illustrative)")
    ax.axhline(0, color=INK, lw=1.1)
    for n, m, p in zip(NS, mu, ps):
        ax.text(n, m - 0.008, f"{m:+.3f}\np = {p:.2f}", ha="center", va="top", fontsize=8,
                color="#11704f")
    logx(ax); ax.set_xlabel("Training-set size N (log scale)")
    ax.set_ylabel("AUC difference (validation)")
    # every p here is >= 0.6, so the title claims no direction
    ax.set_title("Freezing the 48 quantum rotation angles (T: 154 → 106) makes no significant\n"
                 f"difference at any N  ·  the predicted small-data benefit does not appear  ·  "
                 f"all p ≥ {min(ps):.2f}", fontsize=10.5)
    ax.legend(loc="lower right"); tidy(ax)
    fig.tight_layout()
    save(fig, "07_frozen_vs_trained.png", "results/sweep_results.csv · validation, 5 paired seeds")


# ================================================================== 8. simple vs neural
def fig_simple_neural():
    groups = {
        "Non-neural classical": (["iforest", "logreg", "rf", "gbt", "rule"], "#008300"),
        "Neural · classical": (["mlp", "MLP_pre", "MLP_reup", "MLP_frozen"], "#e34948"),
        "Neural · quantum": (["Q_pre", "Q_reup", "Q_frozen"], "#2a78d6"),
        "Kernel · quantum": (["QSVM"], "#4a3aa7"),
    }
    mean = VAL.groupby(["model", "N"]).auc.mean()
    fig, ax = plt.subplots(figsize=(11, 5.6))
    jit = {"Non-neural classical": -0.09, "Neural · classical": -0.03,
           "Neural · quantum": 0.03, "Kernel · quantum": 0.09}
    for g, (models, col) in groups.items():
        first = True
        for m in models:
            xs, ys = [], []
            for i, n in enumerate(NS):
                if (m, n) in mean.index:
                    xs.append(i + jit[g]); ys.append(mean[(m, n)])
            ax.scatter(xs, ys, s=46, color=col, edgecolor=SURFACE, lw=1, zorder=3,
                       label=g if first else None)
            first = False
    for i, n in enumerate(NS):
        simple = max(mean[(m, n)] for m in ["iforest", "logreg", "rf"])
        neural = max(mean[(m, n)] for m in ["mlp", "MLP_pre", "MLP_reup", "MLP_frozen", "Q_pre", "Q_reup", "Q_frozen"])
        ax.annotate("", xy=(i + 0.2, neural), xytext=(i + 0.2, simple),
                    arrowprops=dict(arrowstyle="<->", color=INK3, lw=1))
        ax.text(i + 0.25, (simple + neural) / 2, f"{simple-neural:+.3f}", fontsize=8, color=INK2, va="center")
    ax.set_xticks(range(len(NS)), [f"N = {n:,}" for n in NS])
    ax.set_xlim(-0.4, len(NS) - 0.2)
    ax.set_ylabel("Validation AUC")
    ax.set_title("With little data, non-neural models beat every neural network — quantum or classical\n"
                 "Arrow = best of Isolation Forest / Logistic Regression / Random Forest minus best neural network")
    ax.legend(loc="lower right"); ax.grid(axis="x", visible=False); tidy(ax)
    fig.tight_layout()
    save(fig, "08_simple_vs_neural.png", "validation AUC, 5 seeds")


# ================================================================== 9. data hunger
def fig_hunger():
    s3 = VAL
    rows = []
    for m in C:
        a = s3[(s3.model == m) & (s3.N == 571)].auc.mean()
        b = s3[(s3.model == m) & (s3.N == 1429)].auc.mean()
        if not (np.isnan(a) or np.isnan(b)):
            rows.append((m, b - a))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[C[r[0]] for r in rows], height=0.64,
            hatch=[("///" if r[0].startswith("MLP_") else "") for r in rows], edgecolor=SURFACE, lw=0)
    for yi, (m, g) in zip(y, rows):
        # every value label sits right of the bar end or the zero line, never over the axis labels
        ax.text(max(g, 0) + 0.004, yi, f"{g:+.3f}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y, [NAME[r[0]] for r in rows], fontsize=8.8)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Validation AUC gained from N = 571 to N = 1,429 (same training windows)")
    ax.set_title("How much each model gains from more data\n"
                 "Neural networks, quantum and classical, gain the most; the quantum kernel barely moves")
    ax.grid(axis="y", visible=False); tidy(ax)
    fig.tight_layout()
    save(fig, "09_data_hunger.png", "validation AUC, 5 seeds, identical subsamples · hatched = matched classical twin")


# ================================================================== 10. endpoint seeds
def fig_seeds():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=False)
    for ax, n in zip(axes, (571, 68296)):
        pos = 0; ticks = []; labs = []
        for q, c in TWIN.items():
            a = fin[(fin.model == q) & (fin.N == n)].sort_values("seed").test_auc.values
            b = fin[(fin.model == c) & (fin.N == n)].sort_values("seed").test_auc.values
            for i in range(len(a)):
                ax.plot([pos, pos + 1], [b[i], a[i]], color=C[q], alpha=0.30, lw=1, zorder=1)
            ax.scatter(np.full(len(b), pos), b, s=30, facecolor=SURFACE, edgecolor=C[q], lw=1.4, zorder=3)
            ax.scatter(np.full(len(a), pos + 1), a, s=30, color=C[q], edgecolor=SURFACE, lw=0.8, zorder=3)
            ax.plot([pos - 0.2, pos + 0.2], [b.mean()] * 2, color=INK, lw=2)
            ax.plot([pos + 0.8, pos + 1.2], [a.mean()] * 2, color=INK, lw=2)
            p = stats.wilcoxon(a, b).pvalue
            up = sum(a > b)
            # fixed row just above the plot area, not tied to the data, so it cannot hit the title
            ax.text(pos + 0.5, 1.02,
                    f"quantum higher on {up}/10 seeds\np = {p:.3f}" + (" ✔" if p < 0.05 else ""),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7.8, color=INK2)
            ticks += [pos, pos + 1]; labs += ["twin", "quantum"]
            pos += 3
        ax.set_xticks(ticks, labs, fontsize=8.2)
        for k, q in enumerate(TWIN):
            ax.text(k * 3 + 0.5, -0.10, NAME[q].replace("HQNN · ", ""), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.8, color=C[q], fontweight="bold")
        ax.set_title(f"N = {n:,}", fontsize=11, pad=38)
        ax.set_ylabel("Test AUC"); ax.grid(axis="x", visible=False); tidy(ax)
    fig.suptitle("Every seed, paired  ·  each line joins a quantum network to its twin trained on the "
                 "same windows  ·  bar = mean", fontsize=12.5, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "10_endpoint_seeds.png", "results/final_results.csv · test set · 10 seeds · Wilcoxon signed-rank")


# ================================================================== 11. attack detection
def fig_attack():
    F = fin[fin.N == 68296]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.6), sharey=False)
    for ax, a in zip(axes, ATT):
        seen = [bres.loc[b_m, f"auc_{a}"] for b_m in ("gbt", "rf")]
        marks = []
        for k, (q, c) in enumerate(TWIN.items()):
            qa = F[F.model == q][f"test_auc_{a}"].values
            ca = F[F.model == c][f"test_auc_{a}"].values
            p = stats.wilcoxon(qa, ca).pvalue
            seen += [qa.mean(), ca.mean()]
            ax.plot([k, k], [ca.mean(), qa.mean()], color=C[q], lw=2.2, zorder=1)
            ax.scatter(k, ca.mean(), s=58, facecolor=SURFACE, edgecolor=C[q], lw=1.8, zorder=3)
            ax.scatter(k, qa.mean(), s=58, color=C[q], edgecolor=SURFACE, lw=1, zorder=3)
            if p < 0.05:
                marks.append((k, max(qa.mean(), ca.mean()), qa.mean() > ca.mean()))
        for b_m in ("gbt", "rf"):
            ax.axhline(bres.loc[b_m, f"auc_{a}"], color=C[b_m], lw=1, ls=":", zorder=0)
        # headroom sized to each panel's own range, so the ✔ stays inside the plot area
        lo, hi = min(seen), max(seen)
        span = hi - lo
        ax.set_ylim(lo - 0.08 * span, hi + 0.16 * span)
        for k, top, won in marks:
            ax.text(k, top + 0.05 * span, "✔", ha="center", va="bottom", fontsize=9,
                    color="#11704f" if won else "#b3342f")
        ax.set_xticks(range(3), ["pre-\nlayer", "re-\nupload", "frozen"], fontsize=8)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(ATT_LAB[a].replace("\n", " "), fontsize=10.5)
        ax.grid(axis="x", visible=False); tidy(ax)
    axes[0].set_ylabel("Test AUC, full data")
    fig.suptitle("Detection by attack type  ·  ● filled = HQNN, ○ open = matched twin, ✔ = p < 0.05  ·  "
                 "dotted = Gradient Boosting (yellow) and Random Forest (green)",
                 fontsize=12, fontweight="bold", color=INK, y=1.06)
    fig.tight_layout()
    save(fig, "11_attack_detection.png", "results/final_results.csv (10 seeds) · results/baseline_results.csv")


# ================================================================== 12. significance
def fig_significance():
    metrics = [("test_auc", "AUC"), ("test_ap", "PR-AUC"), ("test_prec", "Precision"),
               ("test_recall", "Recall")] + [(f"test_auc_{a}", ATT_LAB[a].replace("\n", " ")) for a in ATT]
    pts = []
    for n in (571, 68296):
        for q, c in TWIN.items():
            X = fin[(fin.model == q) & (fin.N == n)].sort_values("seed")
            Y = fin[(fin.model == c) & (fin.N == n)].sort_values("seed")
            for col, lab in metrics:
                a, b = X[col].values, Y[col].values
                try:
                    p = stats.wilcoxon(a, b).pvalue
                except ValueError:
                    p = 1.0
                pts.append((n, q, lab, a.mean() - b.mean(), p))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for n, mk, lab in ((571, "o", "N = 571"), (68296, "D", "N = 68,296")):
        sub = [p for p in pts if p[0] == n]
        ax.scatter([p[3] for p in sub], [-np.log10(p[4]) for p in sub], marker=mk, s=54,
                   c=[C[p[1]] for p in sub], edgecolor=SURFACE, lw=1, alpha=0.9, zorder=3)
        ax.scatter([], [], marker=mk, s=54, color=INK3, label=lab)
    ax.axhline(-np.log10(0.05), color=INK3, lw=1.2, ls="--")
    # x in axes fraction, y in data: always inside the plot, whatever the x range
    ax.text(0.995, -np.log10(0.05) + 0.04, "p = 0.05", transform=ax.get_yaxis_transform(),
            color=INK3, fontsize=8.5, ha="right", va="bottom")
    ax.axvline(0, color=INK, lw=1)
    sig = [p for p in pts if p[4] < 0.05]
    wins = sum(1 for p in sig if p[3] > 0); loss = sum(1 for p in sig if p[3] < 0)
    ys = [-np.log10(p[4]) for p in pts]
    ax.set_ylim(-0.1, max(ys) + 0.35)
    ax.set_xlim(min(p[3] for p in pts) - 0.03, max(p[3] for p in pts) + 0.03)
    for q in TWIN:
        ax.scatter([], [], s=54, color=C[q], label=NAME[q])
    ax.set_xlabel("Quantum minus matched twin (test set)")
    ax.set_ylabel("−log₁₀ p   (higher = stronger evidence)")
    ax.set_title(f"All {len(pts)} comparisons at once: {wins} significant quantum wins, all at full data;\n"
                 f"{loss} significant losses, all at scarce data")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8); tidy(ax)
    # layout is fixed BEFORE labels are placed, so the pixel positions checked below are final
    fig.tight_layout()
    # label placement in screen space: try positions around each point and keep the first
    # whose text box clears every data point, every placed label, and the axes edge
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    dots = ax.transData.transform([(p[3], -np.log10(p[4])) for p in pts])
    frame = ax.get_window_extent(rend)
    rad = 8  # marker radius in pixels at 150 dpi
    tries = [(0, 7, "center", "bottom"), (0, -7, "center", "top"), (7, 0, "left", "center"),
             (-7, 0, "right", "center"), (6, 6, "left", "bottom"), (-6, 6, "right", "bottom"),
             (6, -6, "left", "top"), (-6, -6, "right", "top"), (0, 20, "center", "bottom"),
             (0, -20, "center", "top")]
    boxes = []
    for p in sorted(sig, key=lambda p: -abs(p[3])):
        if abs(p[3]) <= 0.07:
            continue
        for dx, dy, ha, va in tries:
            t = ax.annotate(p[2], (p[3], -np.log10(p[4])), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=7, color=INK2)
            bb = t.get_window_extent(rend)
            clear = (bb.x0 > frame.x0 and bb.x1 < frame.x1 and bb.y1 < frame.y1
                     and not any(bb.overlaps(b) for b in boxes)
                     and not any(bb.x0 - rad < px < bb.x1 + rad and bb.y0 - rad < py < bb.y1 + rad
                                 for px, py in dots))
            if clear:
                boxes.append(bb)
                break
            t.remove()
        else:
            print(f"    fig 12: no clear spot for '{p[2]}', left unlabelled")
    save(fig, "12_significance_map.png", "results/final_results.csv · 10 paired seeds · Wilcoxon signed-rank")


# ================================================================== 13. ranking
def fig_ranking():
    F = fin[fin.N == 68296]
    te = pd.read_csv("data/dataset_test.csv")
    n = {a: int((te.attack == a).sum()) for a in ATT}
    sub = [a for a in ATT if a != "drift"]
    w = {a: n[a] / sum(n[b] for b in sub) for a in sub}
    rows = []
    for m in ["gbt", "rf", "logreg", "mlp", "iforest", "rule"]:
        r = bres.loc[m]
        rows.append((m, r["auc"], sum(w[a] * r[f"auc_{a}"] for a in sub), r["ap"]))
    for m in TWIN.keys() | set(TWIN.values()):
        g = F[F.model == m]
        rows.append((m, g.test_auc.mean(), sum(w[a] * g[f"test_auc_{a}"].mean() for a in sub), g.test_ap.mean()))
    rows.sort(key=lambda r: r[1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), sharey=True)
    y = np.arange(len(rows))
    for ax, k, title in ((axes[0], 1, "Test AUC"), (axes[1], 2, "Test AUC, gradual drift excluded"),
                         (axes[2], 3, "Test PR-AUC")):
        vals = [r[k] for r in rows]
        ax.hlines(y, min(vals) - 0.02, vals, color=GRID, lw=1.2)
        ax.scatter(vals, y, s=64, color=[C[r[0]] for r in rows],
                   marker=None, edgecolor=SURFACE, lw=1.2, zorder=3)
        for yi, v, r in zip(y, vals, rows):
            ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10.5); ax.grid(axis="y", visible=False); tidy(ax)
    axes[0].set_yticks(y, [NAME[r[0]] for r in rows], fontsize=8.6)
    fig.suptitle("Every model on the held-out test set at full data (29,292 windows, 609 unseen vessels)",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "13_model_ranking.png", "results/final_results.csv (10 seeds) · results/baseline_results.csv (5 seeds)")


# ================================================================== 14. quantum kernel
def fig_kernel():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    models = ["iforest", "logreg", "rf", "gbt", "mlp", "QSVM", "MLP_frozen", "MLP_pre", "MLP_reup",
              "Q_reup", "Q_pre", "Q_frozen"]
    non_neural = {"iforest", "logreg", "rf", "gbt"}
    s3 = VAL
    above = {}
    for i, n in enumerate((571, 1429)):
        mu = {m: s3[(s3.model == m) & (s3.N == n)].auc.mean() for m in models}
        above[n] = [m for m in models if mu[m] > mu["QSVM"]]
        order = sorted(models, key=lambda m: -mu[m])
        for j, m in enumerate(order):
            v = s3[(s3.model == m) & (s3.N == n)].auc.values
            x = i * 14 + j
            ax.bar(x, v.mean(), color=C[m], width=0.78, alpha=1 if m == "QSVM" else 0.55,
                   hatch="///" if m.startswith("MLP_") else "", edgecolor=SURFACE, lw=0)
            ax.scatter(np.full(len(v), x), v, s=12, color=INK, zorder=3)
            # label under the axis, not inside the bar: short bars cannot hold it
            ax.text(x, 0.397, NAME[m].replace("Matched MLP · ", "Twin · ").replace("HQNN · ", "HQNN ")
                    .replace(" (no labels)", "").replace("Quantum Support Vector Machine", "QSVM"),
                    rotation=90, ha="center", va="top", fontsize=7.6,
                    color=C["QSVM"] if m == "QSVM" else INK2,
                    fontweight="bold" if m == "QSVM" else "normal")
        ax.text(i * 14 + 5.5, 0.915, f"N = {n:,}", ha="center", fontsize=11, fontweight="bold", color=INK)
    # floor at 0.40: three seeds score below chance (lowest 0.430) and a 0.50 floor hid them
    ax.axhline(0.5, color=INK3, lw=0.9, ls=":", zorder=0)
    ax.text(-0.9, 0.503, "chance", fontsize=7.5, color=INK3, va="bottom")
    ax.set_ylim(0.40, 0.93); ax.set_xticks([])
    ax.set_ylabel("Validation AUC  (bar = mean, dots = seeds)")
    # title built from the numbers, so it cannot claim more than the bars show
    short = {"mlp": "71-parameter MLP", "MLP_reup": "re-uploading twin", "MLP_pre": "pre-layer twin",
             "MLP_frozen": "frozen twin", "Q_reup": "re-uploading HQNN", "Q_pre": "pre-layer HQNN",
             "Q_frozen": "frozen HQNN", "gbt": "Gradient Boosting", "rf": "Random Forest",
             "logreg": "Logistic Regression", "iforest": "Isolation Forest"}

    def who(ms):
        return ", ".join(short[m] for m in ms)
    small_neural = [m for m in above[571] if m not in non_neural]
    big_neural = [m for m in above[1429] if m not in non_neural]
    line1 = ("At N = 571 the quantum kernel beats every neural network, quantum or classical"
             if not small_neural else f"At N = 571 the quantum kernel is beaten by {who(above[571])}")
    line2 = (f"By N = 1,429 {len(big_neural)} neural network{'s' if len(big_neural) != 1 else ''} "
             f"overtake it: {who(big_neural)}" if big_neural
             else "At N = 1,429 it still beats every neural network")
    ax.set_title(f"{line1}\n{line2}", fontsize=10.5, pad=16)
    ax.grid(axis="x", visible=False); tidy(ax)
    fig.tight_layout()
    save(fig, "14_quantum_kernel.png", "identical windows, 5 seeds · hatched = matched classical twin")


if __name__ == "__main__":
    for f in (fig_pipeline, fig_attacks, fig_features, fig_curves, fig_inversion, fig_where,
              fig_frozen, fig_simple_neural, fig_hunger, fig_seeds, fig_attack,
              fig_significance, fig_ranking, fig_kernel):
        try:
            f()
        except Exception as e:
            import traceback
            print(f"  FAILED {f.__name__}: {e}")
            traceback.print_exc()
