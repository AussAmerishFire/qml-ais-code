"""
28_paper_figures.py
===================
The figures the paper needs that 24_figures.py does not draw.

    python scripts/28_paper_figures.py      (run from the repository root)

    15_method_overview.png      the whole study as one flow, with real counts
    16_circuits.png             the three hybrid quantum architectures
    17_attack_examples.png      each attack injected into one real vessel track
    18_learning_curves_test.png learning curves on the held-out TEST set
    19_inversion_test.png       quantum minus twin on TEST at all five sizes
    20_geometric_difference.png the measured ceiling on quantum-kernel advantage
    21_quantum_features.png     quantum features inside Random Forest / GBT

Same style, palette and footer as 24_figures.py, which this imports. No model
is trained. Figure 17 runs the real injection functions from PIPELINE.py on a
real track, so what is drawn is exactly what the dataset contains.
"""

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import wilcoxon

f = importlib.import_module("24_figures")          # style, palette, save(), tidy()
P = importlib.import_module("PIPELINE")             # the real attack injectors
sw = importlib.import_module("16_sweep")            # the real architectures

C, NAME, TWIN, NS = f.C, f.NAME, f.TWIN, f.NS
INK, INK2, INK3, GRID, SURFACE = f.INK, f.INK2, f.INK3, f.GRID, f.SURFACE
save, tidy, logx = f.save, f.tidy, f.logx

fs = pd.read_csv("results/full_sweep_results.csv")
fin = pd.read_csv("results/final_results.csv")

EXAMPLE_MMSI = 353798000      # cargo vessel: 164 reports, 3.2 h, ~14 kn, no natural gap > 400 s


# ======================================================================= 15
def fig_overview():
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), 17, 13.5, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc=fc, ec="none"))
        ax.text(x + 8.5, y + 11.2, title, ha="center", va="center", fontsize=10,
                fontweight="bold", color=INK)
        ax.text(x + 8.5, y + 5.4, body, ha="center", va="center", fontsize=8.2,
                color=INK2, linespacing=1.45)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     color=INK3, lw=1.2))

    data = [("Raw AIS", "8,156,509 reports\nMarineCadastre\n1 January 2023"),
            ("Filter", "valid, under way,\n≥ 200 reports per vessel\n1,256,434 reports"),
            ("Window", "10 reports per window\n12 engineered features\n114,969 windows"),
            ("Inject attacks", "speed lie · jump · gap\ndark · drift\n3.47% anomalous"),
            ("Split by vessel", "68,296 / 17,381 / 29,292\ntrain / val / test\nzero vessel overlap")]
    expt = [("Subsample", "N = 571 to 68,296\nanomaly rate held\nidentical windows"),
            ("13 models", "3 hybrid quantum nets\n3 matched classical twins\nQSVM + 6 classical"),
            ("Train", "fixed 640 gradient steps\n5 seeds per cell\n10 at the endpoints"),
            ("Evaluate", "test AUC · PR-AUC\nper-attack AUC\npaired Wilcoxon"),
            ("Follow-ups", "geometric difference\nquantum features\nin tree ensembles")]
    xs = [1, 21, 41, 61, 81]
    for x, (t, b) in zip(xs, data):
        box(x, 23, t, b, "#e6f0fc")
    for x, (t, b) in zip(xs, expt):
        box(x, 3, t, b, "#eef3ec")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + 17.4, 29.8, b - 0.4, 29.8)
        arrow(a + 17.4, 9.8, b - 0.4, 9.8)
    # elbow connector between the rows, routed through the gap so it crosses no box
    ax.plot([89.5, 89.5, 9.5], [22.6, 19.8, 19.8], color=INK3, lw=1.2)
    ax.add_patch(FancyArrowPatch((9.5, 19.8), (9.5, 16.9), arrowstyle="-|>", mutation_scale=11,
                                 color=INK3, lw=1.2))
    ax.text(0.2, 38.5, "DATA", fontsize=8.5, fontweight="bold", color="#184f95")
    ax.text(0.2, 17.6, "EXPERIMENT", fontsize=8.5, fontweight="bold", color="#11704f")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    save(fig, "15_method_overview.png", "logs/pipeline_v2.log · scripts/16_sweep.py · scripts/22_full_sweep.py")


# ======================================================================= 16
def fig_circuits():
    """Schematic of the three architectures, drawn to match 16_sweep._circuit and
    q_params exactly (checked against qml.specs before drawing)."""
    arch = [
        ("Q_pre", "HQNN · classical pre-layer", [("emb", ""), ("sel", "2 layers\n48 trainable angles")],
         "sum of ⟨Z⟩\n2 params"),
        ("Q_reup", "HQNN · data re-uploading", [("emb", ""), ("sel", "layer 1\n24 angles"),
                                               ("emb", ""), ("sel", "layer 2\n24 angles")],
         "linear 8 → 1\n9 params"),
        ("Q_frozen", "HQNN · frozen circuit (control)", [("emb", ""), ("frz", "2 layers\n48 angles FROZEN")],
         "sum of ⟨Z⟩\n2 params"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.4))
    for ax, (key, title, blocks, readout) in zip(axes, arch):
        ax.set_xlim(0, 100); ax.set_ylim(-1.5, 9.5); ax.axis("off")
        wires = np.arange(8)
        x0, x1 = 21, 82
        for w in wires:
            ax.plot([x0, x1], [w, w], color=INK3, lw=0.9, zorder=1)
            ax.text(x0 - 1, w, f"q{7 - w}", ha="right", va="center", fontsize=7.5, color=INK3)
        # classical pre-layer
        ax.add_patch(FancyBboxPatch((1, 1.2), 15, 5.6, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc="#f0efec", ec="none"))
        ax.text(8.5, 5.3, "Dense 12 → 8", ha="center", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(8.5, 3.3, "classical, trainable\n104 params", ha="center", va="center",
                fontsize=8, color=INK2)
        ax.add_patch(FancyArrowPatch((16.4, 3.5), (17.9, 3.5), arrowstyle="-|>", mutation_scale=10,
                                     color=INK3))     # stops short of the qubit labels
        # circuit blocks
        span = (x1 - x0 - 6) / len(blocks)
        for i, (kind, label) in enumerate(blocks):
            bx = x0 + 3 + i * span
            if kind == "emb":
                for w in wires:
                    ax.add_patch(Rectangle((bx + span * 0.25, w - 0.32), span * 0.5, 0.64,
                                           fc="#cde2fb", ec="#2a78d6", lw=0.8, zorder=2))
                    ax.text(bx + span * 0.5, w, f"RX(x{7 - w})", ha="center", va="center",
                            fontsize=6.4, color="#184f95", zorder=3)
                ax.text(bx + span * 0.5, 8.1, "angle\nembedding", ha="center", va="bottom",
                        fontsize=7.8, color="#184f95")
            else:
                fc, ec, tc = (("#fde3d6", C[key], "#9a3d16") if kind == "sel"
                              else ("#e7e6e2", INK3, INK2))
                ax.add_patch(FancyBboxPatch((bx + span * 0.08, -0.45), span * 0.84, 7.9,
                                            boxstyle="round,pad=0.05,rounding_size=0.4",
                                            fc=fc, ec=ec, lw=1.2, zorder=2,
                                            hatch="///" if kind == "frz" else None))
                ax.text(bx + span * 0.5, 3.5, "Rot(φ,θ,ω)\n+ CNOT ring", ha="center",
                        va="center", fontsize=7.6, color=tc, zorder=3)
                ax.text(bx + span * 0.5, 8.1, label, ha="center", va="bottom", fontsize=7.8,
                        color=tc)
        # measurement and readout
        for w in wires:
            ax.add_patch(Rectangle((x1 - 2.4, w - 0.3), 2.4, 0.6, fc=SURFACE, ec=INK2, lw=0.8, zorder=2))
            ax.text(x1 - 1.2, w, "⟨Z⟩", ha="center", va="center", fontsize=6.6, color=INK, zorder=3)
        ax.add_patch(FancyArrowPatch((x1 + 0.4, 3.5), (85.5, 3.5), arrowstyle="-|>", mutation_scale=10,
                                     color=INK3))
        ax.add_patch(FancyBboxPatch((86, 1.2), 13, 5.6, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc="#f0efec", ec="none"))
        ax.text(92.5, 4, readout, ha="center", va="center", fontsize=8.2, color=INK2)
        T = sw.q_params(sw.QARCH[key])
        ax.set_title(f"{title}   ·   T = {T} trainable parameters   ·   matched twin: "
                     f"{sw.CPARAMS[sw.TWIN[key]]}", fontsize=10.5, loc="left", color=C[key])
    fig.tight_layout(h_pad=1.2)
    save(fig, "16_circuits.png", "scripts/16_sweep.py (_circuit, q_params) · 8 qubits · structure verified with qml.specs")


# ======================================================================= 17
def _track():
    d = pd.read_csv("data/raw/AIS_RAW_first_1M.csv", usecols=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"],
                    parse_dates=["BaseDateTime"])
    t = d[d.MMSI == EXAMPLE_MMSI].sort_values("BaseDateTime").reset_index(drop=True)
    return t


def _hours(t, t0):
    return (t.BaseDateTime - t0).dt.total_seconds().values / 3600


def _shade(ax, hrs, mask, color):
    if mask.any():
        idx = np.where(mask)[0]
        ax.axvspan(hrs[idx[0]], hrs[idx[-1]] if idx[-1] > idx[0] else hrs[idx[0]] + 0.02,
                   color=color, alpha=0.10, lw=0, label="labelled anomalous")


def fig_attacks():
    t = _track()
    t0 = t.BaseDateTime.iloc[0]
    h = _hours(t, t0)
    mid = lambda r: (r[0] + r[1]) / 2           # noqa: E731 -- representative severity
    sev = {k: mid(v) for k, v in {"speed_lie": P.SPEED_LIE, "jump": P.JUMP_M, "gap": P.GAP_S,
                                  "dark": P.DARK_S, "drift": P.DRIFT_M}.items()}
    col = "#e34948"

    def run(name, seed):
        P.RNG = np.random.default_rng(seed)
        fn = {"speed_lie": P.a_speed_lie, "jump": P.a_jump, "gap": P.a_gap,
              "dark": P.a_dark, "drift": P.a_drift}[name]
        return fn(t.copy(), sev[name])

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))

    # a) speed falsification -- positions untouched, reported SOG altered
    ax = axes[0]
    t2, m = run("speed_lie", 3)
    ax.plot(h, t.SOG, color=INK3, lw=1.6, label="true")
    ax.plot(h, t2.SOG, color=col, lw=1.6, label="reported")
    _shade(ax, h, m, col)
    ax.set_ylabel("Speed over ground (kn)")
    # label what was APPLIED, not what was requested: the injector caps reported
    # speed at 39 kn, so the requested multiplier is not always reachable
    fac = float(np.median(t2.SOG.values[m] / t.SOG.values[m]))
    capped = fac < sev["speed_lie"] - 0.05
    ax.set_title(f"Speed falsification\n×{fac:.1f} applied" + (" (capped at 39 kn)" if capped else ""),
                 fontsize=10)

    # b) position jump -- map, zoomed around the discontinuity
    ax = axes[1]
    t2, m = run("jump", 3)
    k = int(np.where(m)[0][0]) + 1
    lo, hi = max(0, k - 18), min(len(t), k + 18)
    ax.plot(t.LON[lo:hi], t.LAT[lo:hi], color=INK3, lw=1.6, marker="o", ms=2.5, label="true")
    ax.plot(t2.LON[lo:hi], t2.LAT[lo:hi], color=col, lw=1.6, marker="o", ms=2.5, label="reported")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    disp = P.haversine_m(t.LAT.values[k], t.LON.values[k], t2.LAT.values[k], t2.LON.values[k])
    ax.set_title(f"Position jump\n{disp/1000:.1f} km displacement", fontsize=10)
    ax.ticklabel_format(useOffset=False)
    ax.tick_params(axis="x", labelrotation=30)

    # c) reporting gap and d) transponder switch-off -- interval between reports
    for ax, name, what in ((axes[2], "gap", "same voyage"), (axes[3], "dark", "track splits")):
        t2, m = run(name, 3)
        dt0 = t.BaseDateTime.diff().dt.total_seconds().values / 60
        dt2 = t2.BaseDateTime.diff().dt.total_seconds().values / 60
        h2 = _hours(t2, t0)
        ax.plot(h, dt0, color=INK3, lw=1.2, label="true")
        # reported intervals as points, with the silence as one vertical stem: a line
        # through the points would draw a ramp across reports that do not exist
        ax.plot(h2, dt2, "o", ms=2.2, color=col, label="reported")
        j = int(np.nanargmax(dt2))
        ax.vlines(h2[j], np.nanmedian(dt2), dt2[j], color=col, lw=1.8)
        ax.axvspan(h2[j - 1], h2[j], color=col, alpha=0.10, lw=0, label="no reports")
        ax.axhline(P.GAP_BREAK_S / 60, color=INK2, lw=0.9, ls=":")
        ax.text(0.02, P.GAP_BREAK_S / 60 * 1.12, "30 min track break", va="bottom", ha="left",
                fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
        ax.set_yscale("log"); ax.set_ylim(0.15, 400); ax.set_ylabel("Minutes since previous report")
        title = "Reporting gap" if name == "gap" else "Transponder switch-off"
        ax.set_title(f"{title}\n{dt2[j]:.0f} min silence, {what}", fontsize=10)

    # e) gradual drift -- distance between reported and true position
    ax = axes[4]
    t2, m = run("drift", 3)
    off = P.haversine_m(t.LAT.values, t.LON.values, t2.LAT.values, t2.LON.values)
    ax.plot(h, off, color=col, lw=1.8, label="reported vs true")
    _shade(ax, h, m, col)
    ax.set_ylabel("Offset from true position (m)")
    ax.set_title(f"Gradual drift\nup to {off.max()/1000:.2f} km, speed untouched", fontsize=10)

    for i, ax in enumerate(axes):
        if i != 1:
            ax.set_xlabel("Hours")
        ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9, frameon=i in (2, 3))
        tidy(ax)
    fig.suptitle("Each attack injected into the same real vessel track by the study's own injection code  "
                 "·  shading marks the reports labelled anomalous", fontsize=12, fontweight="bold",
                 color=INK, y=1.04)
    fig.tight_layout()
    save(fig, "17_attack_examples.png",
         "real cargo-vessel track, 164 reports over 3.2 h · scripts/PIPELINE.py injectors · "
         "requested severity = midpoint of each range; titles show what was actually applied")


# ======================================================================= 18
def _curve(model, col="test_auc"):
    g = fs[fs.model == model].groupby("N")[col]
    return g.mean().index.values, g.mean().values, (g.std() / np.sqrt(g.count())).values


def fig_curves_test():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    panels = [("Hybrid quantum networks vs matched twins",
               ["Q_pre", "MLP_pre", "Q_reup", "MLP_reup", "Q_frozen", "MLP_frozen"]),
              ("Supervised classical models", ["gbt", "rf", "logreg", "mlp"]),
              ("Label-free methods (others in grey)", ["iforest", "rule"])]
    for ax, (title, models) in zip(axes, panels):
        if "grey" in title:
            for m in fs.model.unique():
                if m not in models:
                    x, mu, _ = _curve(m)
                    ax.plot(x, mu, color="#d8d6d0", lw=1.1, zorder=1)
        for m in models:
            x, mu, se = _curve(m)
            ax.fill_between(x, mu - se, mu + se, color=C[m], alpha=0.10, lw=0)
            ax.plot(x, mu, color=C[m], ls=f.DASH.get(m, "-"), lw=2.2, marker="o", ms=5,
                    mec=SURFACE, mew=1.2, label=NAME[m])
        logx(ax); ax.set_title(title, fontsize=10.5); ax.legend(loc="lower right", fontsize=7.8)
        ax.set_xlabel("Training-set size N (log scale)"); tidy(ax)
    axes[0].set_ylabel("Test AUC")
    fig.suptitle("Learning curves on the held-out test set  ·  29,292 windows from 609 unseen vessels  ·  "
                 "bands = ±1 SE  ·  QSVM omitted (validation only)", fontsize=12, fontweight="bold",
                 color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "18_learning_curves_test.png", "results/full_sweep_results.csv · 5 seeds per point")


# ======================================================================= 19
def fig_inversion_test():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for q, c in TWIN.items():
        d = []
        for n in NS:
            a = fs[(fs.model == q) & (fs.N == n)].sort_values("seed").test_auc.values
            b = fs[(fs.model == c) & (fs.N == n)].sort_values("seed").test_auc.values
            d.append((np.mean(a - b), np.std(a - b) / np.sqrt(len(a))))
        mu, se = np.array([x[0] for x in d]), np.array([x[1] for x in d])
        ax.fill_between(NS, mu - se, mu + se, color=C[q], alpha=0.12, lw=0)
        ax.plot(NS, mu, color=C[q], lw=2.4, marker="o", ms=6, mec=SURFACE, mew=1.3, label=NAME[q])
        for i in range(len(NS) - 1):                       # crossover, interpolated in log N
            if mu[i] < 0 <= mu[i + 1]:
                tt = -mu[i] / (mu[i + 1] - mu[i])
                cx = np.exp(np.log(NS[i]) + tt * (np.log(NS[i + 1]) - np.log(NS[i])))
                ax.plot([cx], [0], marker="|", ms=16, mew=2.2, color=C[q])
                ax.annotate(f"{cx:,.0f}", (cx, 0), xytext=(0, 12 + 10 * list(TWIN).index(q)),
                            textcoords="offset points", ha="center", fontsize=8, color=C[q])
                break
        for n in (571, 68296):                             # 10-seed endpoint significance
            a = fin[(fin.model == q) & (fin.N == n)].sort_values("seed").test_auc.values
            b = fin[(fin.model == c) & (fin.N == n)].sort_values("seed").test_auc.values
            if wilcoxon(a, b).pvalue < 0.05:
                i = NS.index(n)
                ax.text(n, mu[i] + (0.012 if mu[i] > 0 else -0.018), "✱", ha="center", fontsize=9,
                        color=C[q])
    ax.axhline(0, color=INK, lw=1.1)
    ax.axhspan(0, 0.09, color="#2a78d6", alpha=0.05, zorder=0)
    ax.axhspan(-0.15, 0, color="#e34948", alpha=0.05, zorder=0)
    ax.text(620, 0.07, "quantum ahead", color="#184f95", fontsize=9, fontweight="bold")
    ax.text(620, -0.135, "classical twin ahead", color="#b3342f", fontsize=9, fontweight="bold")
    logx(ax); ax.set_ylim(-0.15, 0.09)
    ax.set_xlabel("Training-set size N (log scale)"); ax.set_ylabel("Test AUC: quantum − matched twin")
    # the re-uploading pair is ~0 at both ends and significant at neither, so the
    # title must not claim all three invert
    ax.set_title("On the held-out test set the pre-layer and frozen pairs invert;\n"
                 "the re-uploading pair stays near zero at both ends  ·  tick = crossover  ·  "
                 "✱ = p < 0.05 at 10 seeds", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.2); tidy(ax)
    fig.tight_layout()
    save(fig, "19_inversion_test.png", "results/full_sweep_results.csv (5 seeds) · results/final_results.csv (10 seeds, ✱)")


# ======================================================================= 20
def fig_geometric():
    gd = pd.read_csv("results/geometric_difference.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    for i, n in enumerate((571, 1429)):
        t = gd[(gd.N == n) & (gd.lam == 0.001) & gd.fits]
        g = t.groupby("seed").g_gen.min()
        ax.bar(i - 0.2, np.sqrt(n), width=0.36, color="#f0efec", hatch="///", edgecolor=INK3, lw=0,
               label="largest possible  (√N)" if i == 0 else None)
        ax.bar(i + 0.2, g.median(), width=0.36, color=C["QSVM"],
               label="measured  (median of 5 seeds, range shown)" if i == 0 else None)
        # seed spread as a range bar: individual dots at this scale read as text
        ax.errorbar(i + 0.2, g.median(), yerr=[[g.median() - g.min()], [g.max() - g.median()]],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=4)
        ax.text(i + 0.2, g.max() + 1.2, f"{g.median():.1f}", ha="center", fontsize=9, color=INK)
        ax.text(i - 0.2, np.sqrt(n) + 1, f"{np.sqrt(n):.1f}", ha="center", fontsize=9, color=INK2)
        ax.text(i, -4.2, f"{g.median()/np.sqrt(n)*100:.0f}% of maximum", ha="center", fontsize=8.5,
                color=C["QSVM"], fontweight="bold")
    ax.set_xticks([0, 1], ["N = 571", "N = 1,429"]); ax.set_ylim(-6, 44)
    ax.set_ylabel("Geometric difference g")
    ax.set_title("How far the quantum kernel sits from the best classical kernel", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.2); ax.grid(axis="x", visible=False); tidy(ax)

    ax = axes[1]
    for n, mk in ((571, "o"), (1429, "D")):
        for lam, a in ((0.01, 1.0), (0.001, 0.45)):
            t = gd[(gd.N == n) & (gd.lam == lam)]
            for s, grp in t.groupby("seed"):
                fit = grp[grp.fits]
                if not len(fit):
                    continue
                ax.scatter(grp.s_Q.iloc[0] / n, fit.s_C.min() / n, marker=mk, s=50, alpha=a,
                           color=C["QSVM"], edgecolor=SURFACE, lw=0.8)
        ax.scatter([], [], marker=mk, s=50, color=INK3, label=f"N = {n:,}")
    ax.scatter([], [], s=50, color=C["QSVM"], alpha=1.0, label="λ = 0.01")
    ax.scatter([], [], s=50, color=C["QSVM"], alpha=0.45, label="λ = 0.001")
    lim = [0, 2.8]
    ax.plot(lim, lim, color=INK, lw=1)
    ax.fill_between(lim, lim, [lim[1]] * 2, color="#2a78d6", alpha=0.05)
    ax.text(0.15, 2.45, "quantum kernel describes\nthe real labels more simply", fontsize=8.5, color="#184f95")
    ax.text(1.75, 0.35, "classical kernel\nsimpler", fontsize=8.5, color="#b3342f")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Quantum model complexity  s_Q / N"); ax.set_ylabel("Best classical complexity  s_C / N")
    ax.set_title("On the real anomaly labels: quantum simpler at N = 571 (λ = 0.01, 5/5 seeds);\n"
                 "not at N = 1,429 under light regularisation (λ = 0.001, 0/5)", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8); tidy(ax)
    fig.suptitle("A measured ceiling on quantum-kernel advantage (Huang et al. 2021)", fontsize=12.5,
                 fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "20_geometric_difference.png",
         "results/geometric_difference.csv · 5 seeds per size · only classical kernels that fit (g_tra ≤ 0.1)")


# ======================================================================= 21
def fig_qfeatures():
    q = pd.read_csv("results/qfeat_results.csv")
    arms = [("+Q", "quantum features", C["QSVM"], "-"), ("+RFF", "random Fourier features", INK2, "--"),
            ("+TANH", "random tanh features", INK3, ":"), ("+PCA", "the 8 circuit inputs", "#9ec5f4", "-.")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, model, title in ((axes[0], "rf", "Random Forest"), (axes[1], "gbt", "Gradient Boosting")):
        base = q[(q.model == model) & (q.arm == "base")].set_index(["N", "seed"]).test_auc
        for arm, lab, colr, ls in arms:
            a = q[(q.model == model) & (q.arm == arm)].set_index(["N", "seed"]).test_auc
            d = (a - base).groupby("N")
            mu, se = d.mean(), d.std() / np.sqrt(d.count())
            ax.fill_between(mu.index, mu - se, mu + se, color=colr, alpha=0.10, lw=0)
            ax.plot(mu.index, mu.values, color=colr, ls=ls, lw=2.2 if arm == "+Q" else 1.6,
                    marker="o", ms=4.5, mec=SURFACE, label=f"+ {lab}")
        ax.axhline(0, color=INK, lw=1)
        logx(ax); ax.set_xlabel("Training-set size N (log scale)")
        ax.set_title(f"{title}: no addition helps on test", fontsize=10.5)
        ax.legend(loc="lower right", fontsize=7.8); tidy(ax)
    axes[0].set_ylabel("Test AUC change vs. the plain model")

    ax = axes[2]
    for split, colr, lab in (("val_auc", "#9ec5f4", "validation"), ("test_auc", C["QSVM"], "test")):
        base = q[(q.model == "rf") & (q.arm == "base")].set_index(["N", "seed"])[split]
        a = q[(q.model == "rf") & (q.arm == "+Q")].set_index(["N", "seed"])[split]
        d = (a - base).groupby("N")
        mu, se = d.mean(), d.std() / np.sqrt(d.count())
        ax.fill_between(mu.index, mu - se, mu + se, color=colr, alpha=0.15, lw=0)
        ax.plot(mu.index, mu.values, color=colr, lw=2.4, marker="o", ms=5, mec=SURFACE, label=lab)
    ax.axhline(0, color=INK, lw=1)
    logx(ax); ax.set_xlabel("Training-set size N (log scale)")
    ax.set_title("Random Forest + quantum features: appears to help on\nvalidation at large N, hurts on test at every N",
                 fontsize=10.5)
    ax.set_ylabel("AUC change vs. the plain model")
    ax.legend(loc="lower right", fontsize=8.2); tidy(ax)
    fig.suptitle("Quantum features do not improve the models industry uses  ·  they behave like classical "
                 "random features", fontsize=12.5, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "21_quantum_features.png", "results/qfeat_results.csv · 10 paired seeds per point · bands = ±1 SE")


if __name__ == "__main__":
    for fn in (fig_overview, fig_circuits, fig_attacks, fig_curves_test, fig_inversion_test,
               fig_geometric, fig_qfeatures):
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  FAILED {fn.__name__}: {e}")
            traceback.print_exc()
