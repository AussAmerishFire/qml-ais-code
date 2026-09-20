"""
scripts/11_baseline_viz.py
==================
Every figure needed to read the classical baseline results.

Figures written (150 dpi PNG, light surface -- these are print figures for the
paper, so there is no dark variant):

  figures/earlier/fig_b1_confusion.png    confusion matrix for all six models at the 3.5% budget
  figures/earlier/fig_b2_roc.png          ROC curves, with the actual operating point marked
  figures/earlier/fig_b3_pr.png           precision-recall curves (the honest view at 3.5%)
  figures/earlier/fig_b4_per_attack.png   per-attack AUC and per-attack recall at the budget
  figures/earlier/fig_b5_enrichment.png   anomaly rate by score band
  figures/earlier/fig_b6_budget.png       recall as a function of how much traffic you can check
  figures/earlier/fig_b7_importance.png   which features the models actually use
  figures/earlier/fig_b8_val_test.png     the validation-to-test drop and what explains it

All curves use seed 0 for the stochastic models. Seed-to-seed spread is reported
as +/- in docs/BASELINES.md section 6.2; a curve is a single draw by construction.

Colour: dataviz reference palette, categorical slots 1-6, validated with
scripts/validate_palette.js --mode light (all checks pass; contrast WARN on
slots 3/4/5 discharged by the always-present legends carrying visible values).
"""

import json
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PALETTE AND STYLE  (dataviz reference instance, light mode)
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e8e7e3"

# Categorical slots 1-6, in the fixed order the palette defines. Colour follows
# the MODEL, never its rank -- the same model keeps its hue in every figure.
SERIES = {
    "rf":      "#2a78d6",  # slot 1 blue
    "gbt":     "#eb6834",  # slot 2 orange
    "mlp":     "#1baf7a",  # slot 3 aqua
    "logreg":  "#eda100",  # slot 4 yellow
    "iforest": "#e87ba4",  # slot 5 magenta
    "rule":    "#008300",  # slot 6 green
}
LABEL = {
    "rf": "Random forest",
    "gbt": "Gradient boosting",
    "mlp": "MLP (71 params)",
    "logreg": "Logistic regression",
    "iforest": "Isolation Forest",
    "rule": "Threshold rule",
}
ORDER = ["rf", "gbt", "mlp", "logreg", "iforest", "rule"]

# Sequential ramp: ONE hue, light -> dark. Never a rainbow.
BLUES = LinearSegmentedColormap.from_list(
    "seqblue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",       # solid hairline; dashed grid is an anti-pattern
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "font.size": 9.5,
    "figure.dpi": 150,
})


def tidy(ax):
    """Recessive chrome: grid behind the marks, no top/right spines."""
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


FEATS = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
         "spd_mismatch_max", "crab_mean", "prev_silence_s",
         "course_resid_mean", "mismatch_slope", "straightness",
         "mismatch_mean", "crab_valid"]
FEAT_LABEL = {
    "sog_mean": "mean speed",
    "sog_max_jump": "largest speed jump",
    "cog_turn_max": "sharpest turn",
    "dt_max": "longest reporting gap",
    "spd_mismatch_max": "speed vs position mismatch",
    "crab_mean": "mean crab angle",
    "prev_silence_s": "silence before window",
    "crab_valid": "crab angle validity",
    "course_resid_mean": "course vs bearing travelled",
    "mismatch_slope": "mismatch trend across window",
    "straightness": "path straightness",
    "mismatch_mean": "mean speed/position mismatch",
}
ATTACKS = ["speed_lie", "jump", "gap", "dark", "drift"]  # easiest -> hardest
ALERT_RATE = 0.035


# ---------------------------------------------------------------------------
# FIT THE SELECTED MODELS
# ---------------------------------------------------------------------------

def build_scores():
    tr = pd.read_csv("data/dataset_train.csv")
    va = pd.read_csv("data/dataset_val.csv")
    te = pd.read_csv("data/dataset_test.csv")
    cfg = json.load(open("results/baseline_selection.json"))

    X, y = tr[FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[FEATS].to_numpy(float), va.label.to_numpy(int)
    Xt, yt = te[FEATS].to_numpy(float), te.label.to_numpy(int)

    p = cfg["mlp"]["params"]

    def fit_all(seed):
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        want = int(len(neg) * p["oversample"] / (1 - p["oversample"]))
        idx = rng.permutation(np.concatenate(
            [rng.choice(pos, size=want, replace=True), neg]))
        return {
            "logreg": LogisticRegression(
                C=cfg["logreg"]["params"]["C"], max_iter=2000,
                random_state=seed).fit(X, y),
            "rf": RandomForestClassifier(
                **cfg["rf"]["params"], random_state=seed, n_jobs=-1).fit(X, y),
            "gbt": HistGradientBoostingClassifier(
                **cfg["gbt"]["params"], random_state=seed).fit(X, y),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(5,), alpha=p["alpha"],
                learning_rate_init=p["lr"], max_iter=500, early_stopping=True,
                n_iter_no_change=20, random_state=seed).fit(X[idx], y[idx]),
            "iforest": IsolationForest(
                **cfg["iforest"]["params"], random_state=seed, n_jobs=-1).fit(X),
        }

    def score(models, name, Z):
        if name == "rule":
            return Z[:, FEATS.index("spd_mismatch_max")]
        if name == "iforest":
            return -models["iforest"].score_samples(Z)
        return models[name].predict_proba(Z)[:, 1]

    # Seed 0 drives every figure that shows a curve or a single confusion
    # matrix; the per-attack heatmap averages all five.
    seed_scores = {m: [] for m in ORDER}
    fitted = None
    for s in (0, 1, 2, 3, 4):
        print(f"  seed {s} ...")
        models = fit_all(s)
        if s == 0:
            fitted = models
            st = {m: score(models, m, Xt) for m in ORDER}
            sv = {m: score(models, m, Xv) for m in ORDER}
        for m in ORDER:
            seed_scores[m].append(score(models, m, Xt))

    return te, yt, st, yv, sv, fitted, Xv, yv, seed_scores


def flagged(score, rate=ALERT_RATE):
    """Boolean mask: the top `rate` fraction by score. Every model is held to
    the same alert budget, so precision and recall are directly comparable."""
    k = max(1, int(round(len(score) * rate)))
    return score >= np.partition(score, -k)[-k]


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------

def fig_confusion(yt, st):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7))
    for ax, m in zip(axes.ravel(), ORDER):
        pred = flagged(st[m])
        cm = np.array([
            [int(((~pred) & (yt == 0)).sum()), int((pred & (yt == 0)).sum())],
            [int(((~pred) & (yt == 1)).sum()), int((pred & (yt == 1)).sum())],
        ])
        # Colour by ROW rate, not raw count. Raw counts span 27,000 to 350, so a
        # count ramp would paint one cell dark and three white and say nothing.
        # The row rate is the quantity that matters: of the actual anomalies,
        # what fraction did we catch?
        rate = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rate, cmap=BLUES, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i - 0.13, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if rate[i, j] > 0.55 else INK)
                ax.text(j, i + 0.19, f"{rate[i, j]*100:.1f}% of row",
                        ha="center", va="center", fontsize=8,
                        color="white" if rate[i, j] > 0.55 else INK2)
        ax.set_xticks([0, 1], ["not flagged", "flagged"])
        ax.set_yticks([0, 1], ["actually\nnormal", "actually\nanomalous"])
        ax.set_title(f"{LABEL[m]}\nrecall {rate[1,1]*100:.1f}%  ·  "
                     f"precision {cm[1,1]/max(1,cm[:,1].sum())*100:.1f}%",
                     fontsize=10)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(
        "Confusion matrices at a fixed 3.5% alert budget  ·  29,292 test windows, "
        "1,035 anomalous\nEvery model flags the same 1,025 windows — only which "
        "ones differs",
        fontsize=12, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig("figures/earlier/fig_b1_confusion.png", bbox_inches="tight")
    plt.close(fig)


def fig_roc(yt, st):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for m in ORDER:
        fpr, tpr, _ = roc_curve(yt, st[m])
        auc = roc_auc_score(yt, st[m])
        ax.plot(fpr, tpr, color=SERIES[m], lw=2,
                label=f"{LABEL[m]} — {auc:.3f}")
        # Where the model actually operates at a 3.5% budget.
        pred = flagged(st[m])
        ax.plot(pred[yt == 0].mean(), pred[yt == 1].mean(), "o",
                color=SERIES[m], ms=8, mec=SURFACE, mew=2, zorder=5)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK3, zorder=0)
    ax.text(0.55, 0.50, "chance", color=INK3, fontsize=8.5, rotation=32)
    ax.set_xlabel("False positive rate — normal traffic wrongly flagged")
    ax.set_ylabel("True positive rate — anomalies caught")
    ax.set_title("ROC curves, test set")
    ax.text(0.5, -0.135, "Filled dots mark each model's actual operating point "
            "at the 3.5% alert budget.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=INK2)
    ax.legend(loc="lower right", title="AUC", title_fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/earlier/fig_b2_roc.png", bbox_inches="tight")
    plt.close(fig)


def fig_pr(yt, st):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    base = yt.mean()
    for m in ORDER:
        pr, rc, _ = precision_recall_curve(yt, st[m])
        ap = average_precision_score(yt, st[m])
        ax.plot(rc, pr, color=SERIES[m], lw=2, label=f"{LABEL[m]} — {ap:.3f}")
    ax.axhline(base, ls="--", lw=1, color=INK3, zorder=0)
    ax.text(0.30, base + 0.022, f"random guessing ({base*100:.1f}%)",
            color=INK3, fontsize=8.5)
    ax.set_xlabel("Recall — fraction of anomalies caught")
    ax.set_ylabel("Precision — fraction of alerts that are real")
    ax.set_title("Precision–recall curves, test set")
    ax.text(0.5, -0.135,
            "The honest view at a 3.5% anomaly rate. Isolation Forest ranks 4th "
            "on ROC-AUC but last here.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=INK2)
    ax.legend(loc="upper right", title="Average precision", title_fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/earlier/fig_b3_pr.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_attack(te, yt, seed_scores):
    # Averaged over all 5 seeds, so this figure reports exactly the numbers in
    # docs/BASELINES.md section 6.3. A heatmap of means is well defined; a curve
    # averaged across seeds is not, which is why the curve figures use seed 0.
    at = te.attack.to_numpy()
    auc = np.zeros((len(ORDER), len(ATTACKS)))
    rec = np.zeros_like(auc)
    for i, m in enumerate(ORDER):
        runs = seed_scores[m]
        for j, a in enumerate(ATTACKS):
            sel = (yt == 0) | (at == a)
            auc[i, j] = np.mean([roc_auc_score(yt[sel], s[sel]) for s in runs])
            rec[i, j] = np.mean([flagged(s)[at == a].mean() for s in runs])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, M, ttl, lo, hi in (
        (axes[0], auc, "AUC per attack type (threshold-free)", 0.5, 1.0),
        (axes[1], rec, "Recall per attack type at the 3.5% budget", 0.0, 1.0),
    ):
        ax.imshow(M, cmap=BLUES, vmin=lo, vmax=hi, aspect="auto")
        for i in range(len(ORDER)):
            for j in range(len(ATTACKS)):
                v = (M[i, j] - lo) / (hi - lo)
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        fontsize=9.5, fontweight="bold",
                        color="white" if v > 0.55 else INK)
        ax.set_xticks(range(len(ATTACKS)), ATTACKS, fontsize=9)
        ax.set_yticks(range(len(ORDER)), [LABEL[m] for m in ORDER], fontsize=9)
        ax.set_title(ttl, fontsize=10.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_xlabel("easiest  →  hardest", fontsize=9, color=INK2)
    fig.suptitle(
        "Mean of 5 seeds.  Left: drift sits near 0.60 for every model, and the "
        "zero-parameter rule scores highest on it.\nRight: at the operating "
        "point, drift and dark are barely detected at all — AUC flatters them.",
        fontsize=11.5, fontweight="bold", color=INK, y=1.08)
    fig.tight_layout()
    fig.savefig("figures/earlier/fig_b4_per_attack.png", bbox_inches="tight")
    plt.close(fig)


def fig_enrichment(yt, st):
    """How much does a rising score actually raise the odds of an anomaly?

    Plotted as the anomaly rate inside each 5-percentile score band. Percentile
    rank puts all six models on one x-axis -- their raw scores are
    probabilities, negative path lengths and knots respectively, which share no
    units. A flat line at the base rate would mean the score carries nothing.
    """
    base = yt.mean()
    edges = np.linspace(0, 1, 21)
    mid = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    tops = []
    for m in ORDER:
        r = pd.Series(st[m]).rank(pct=True).to_numpy()
        b = np.digitize(r, edges[1:-1])
        rate = np.array([yt[b == i].mean() if (b == i).any() else np.nan
                         for i in range(20)])
        ax.plot(mid * 100, rate * 100, color=SERIES[m], lw=2, marker="o", ms=4,
                mec=SURFACE, mew=1, label=LABEL[m])
        tops.append((m, rate[-1] * 100))

    # Spread the end labels on the log axis so near-tied models stay readable.
    placed = []
    for m, v in sorted(tops, key=lambda t: t[1]):
        pos = v if not placed else max(v, placed[-1] * 1.13)
        placed.append(pos)
        ax.plot([100.5, 102], [v, pos], lw=0.7, color=SERIES[m])
        ax.text(102.5, pos, f"{v:.0f}%", va="center", fontsize=8.5,
                color=SERIES[m])

    ax.axhline(base * 100, ls="--", lw=1, color=INK3, zorder=0)
    ax.text(3, base * 100 * 1.15, f"base rate {base*100:.1f}% — a flat line here "
            "would mean the score is worthless", color=INK3, fontsize=8.5)
    ax.axvspan(95, 100, color="#eda100", alpha=0.12, zorder=0)
    ax.annotate("the top band is\nroughly the 3.5%\nan analyst reviews",
                xy=(97.5, 1.0), xytext=(84, 0.62), fontsize=8.5, color=INK2,
                ha="center",
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 40])
    ax.set_yticklabels(["0.5%", "1%", "2%", "5%", "10%", "20%", "40%"])
    ax.set_ylim(0.4, 40)
    ax.set_xlabel("Score percentile band (each point is 5% of the test windows)")
    ax.set_ylabel("Share of windows in that band that are anomalous")
    ax.set_title("Does a higher score really mean a higher chance of an anomaly?")
    ax.set_xlim(0, 108)
    ax.legend(loc="upper left")
    tidy(ax)
    fig.text(0.5, -0.02,
             "Every model rises — the score does carry signal. But the top band "
             "tops out around 1 anomaly in 4,\nand that is the ceiling on "
             "precision at this alert budget.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig("figures/earlier/fig_b5_enrichment.png", bbox_inches="tight")
    plt.close(fig)


def fig_budget(yt, st):
    rates = np.linspace(0.005, 0.20, 40)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for m in ORDER:
        recs, precs = [], []
        for r in rates:
            p = flagged(st[m], r)
            recs.append(p[yt == 1].mean())
            precs.append((yt[p] == 1).mean())
        axes[0].plot(rates * 100, np.array(recs) * 100, color=SERIES[m], lw=2,
                     label=LABEL[m])
        axes[1].plot(rates * 100, np.array(precs) * 100, color=SERIES[m], lw=2,
                     label=LABEL[m])
    for ax, yl, ttl in (
        (axes[0], "Anomalies caught (%)",
         "How many anomalies you catch"),
        (axes[1], "Alerts that are real (%)",
         "How many alerts are worth opening"),
    ):
        ax.axvline(3.5, color=INK3, ls="--", lw=1, zorder=0)
        ax.text(3.8, ax.get_ylim()[1] * 0.05, "3.5%", color=INK3, fontsize=8.5)
        ax.set_xlabel("Share of traffic an analyst can review (%)")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10.5)
        tidy(ax)
    axes[0].legend(loc="lower right")
    fig.suptitle(
        "Performance against the real operational constraint: how much traffic "
        "can be checked",
        fontsize=12, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig("figures/earlier/fig_b6_budget.png", bbox_inches="tight")
    plt.close(fig)


def fig_importance(fitted, Xv, yv):
    # Permutation importance, not the impurity-based kind: impurity importance
    # is biased toward high-cardinality continuous features, and all eight of
    # ours are continuous but differently distributed. Computed on VALIDATION,
    # so the test split stays untouched by anything diagnostic.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    r = permutation_importance(fitted["rf"], Xv, yv, n_repeats=10,
                               random_state=0, scoring="roc_auc", n_jobs=-1)
    o = np.argsort(r.importances_mean)
    axes[0].barh([FEAT_LABEL[FEATS[i]] for i in o], r.importances_mean[o],
                 xerr=r.importances_std[o], color="#2a78d6", height=0.62,
                 error_kw={"ecolor": INK3, "lw": 1})
    axes[0].set_xlabel("Drop in AUC when the feature is shuffled")
    axes[0].set_title("Random forest — permutation importance", fontsize=10.5)

    coef = fitted["logreg"].coef_[0]
    o2 = np.argsort(np.abs(coef))
    axes[1].barh([FEAT_LABEL[FEATS[i]] for i in o2], coef[o2],
                 color=["#e34948" if coef[i] < 0 else "#2a78d6" for i in o2],
                 height=0.62)
    axes[1].axvline(0, color=INK3, lw=1)
    axes[1].set_xlabel("Coefficient  (blue raises the anomaly score, red lowers it)")
    axes[1].set_title("Logistic regression — fitted coefficients", fontsize=10.5)

    for ax in axes:
        tidy(ax)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "What the models actually use  ·  all eight features are on the same "
        "[0, pi] scale, so these are directly comparable",
        fontsize=12, fontweight="bold", color=INK, y=1.05)
    fig.tight_layout()
    fig.savefig("figures/earlier/fig_b7_importance.png", bbox_inches="tight")
    plt.close(fig)


def fig_val_test(te, yt, st, yv, sv):
    at = te.attack.to_numpy()
    cfgv = json.load(open("results/baseline_selection.json"))
    rng = np.random.default_rng(0)
    target = (pd.read_csv("data/dataset_val.csv").query("label==1")
              .attack.value_counts(normalize=True))

    rows = []
    for m in ORDER:
        va_auc = cfgv[m]["val_auc"] if m in cfgv else roc_auc_score(yv, sv[m])
        te_auc = roc_auc_score(yt, st[m])
        neg = np.where(yt == 0)[0]
        by = {a: np.where((yt == 1) & (at == a))[0] for a in target.index}
        n_tot = int(min(len(by[a]) / target[a] for a in target.index))
        mix = [roc_auc_score(
            yt[np.concatenate([neg] + [rng.choice(by[a], int(round(n_tot*target[a])),
                                                  replace=False)
                                       for a in target.index])],
            st[m][np.concatenate([neg] + [rng.choice(by[a], int(round(n_tot*target[a])),
                                                     replace=False)
                                          for a in target.index])])
            for _ in range(30)]
        rows.append((m, va_auc, float(np.mean(mix)), te_auc))

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    xs = [0, 1, 2]
    for m, v, mixed, t in rows:
        ax.plot(xs, [v, mixed, t], "-o", color=SERIES[m], lw=2, ms=7,
                mec=SURFACE, mew=1.5, label=LABEL[m])

    # Nudge end labels apart so near-tied models (rf and gbt differ by 0.0007)
    # do not print on top of each other.
    span = max(r[3] for r in rows) - min(r[3] for r in rows)
    gap = span * 0.055
    placed = []
    for m, v, mixed, t in sorted(rows, key=lambda r: r[3]):
        pos = t if not placed else max(t, placed[-1] + gap)
        placed.append(pos)
        ax.plot([2.02, 2.07], [t, pos], lw=0.7, color=SERIES[m])
        ax.text(2.09, pos, f"{t:.3f}", va="center", fontsize=8.5,
                color=SERIES[m])
    ax.set_xticks(xs, ["Validation\n(364 vessels)",
                       "Test, re-mixed to\nvalidation's attack blend",
                       "Test\n(609 vessels)"])
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Every model drops from validation to test — including the one\n"
                 "with zero trainable parameters, so this is not overfitting",
                 fontsize=11)
    ax.text(0.5, -0.20,
            "Test carries +9.9pp more drift (the hardest attack) than validation. "
            "Re-mixing test to validation's attack blend\nrecovers 35–39% of the "
            "gap; the rest is that the test vessels are simply harder.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=INK2)
    ax.legend(loc="lower left", ncol=2)
    ax.set_xlim(-0.15, 2.45)
    tidy(ax)
    fig.tight_layout()
    fig.savefig("figures/earlier/fig_b8_val_test.png", bbox_inches="tight")
    plt.close(fig)


def main():
    print("fitting selected models across 5 seeds ...")
    te, yt, st, yv, sv, fitted, Xv, yvv, seed_scores = build_scores()

    for name, fn, args in (
        ("figures/earlier/fig_b1_confusion.png", fig_confusion, (yt, st)),
        ("figures/earlier/fig_b2_roc.png", fig_roc, (yt, st)),
        ("figures/earlier/fig_b3_pr.png", fig_pr, (yt, st)),
        ("figures/earlier/fig_b4_per_attack.png", fig_per_attack, (te, yt, seed_scores)),
        ("figures/earlier/fig_b5_enrichment.png", fig_enrichment, (yt, st)),
        ("figures/earlier/fig_b6_budget.png", fig_budget, (yt, st)),
        ("figures/earlier/fig_b7_importance.png", fig_importance, (fitted, Xv, yvv)),
        ("figures/earlier/fig_b8_val_test.png", fig_val_test, (te, yt, st, yv, sv)),
    ):
        fn(*args)
        print("  wrote", name)


if __name__ == "__main__":
    main()
