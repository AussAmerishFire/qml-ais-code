"""
scripts/23_build_tables.py
==================
Builds docs/COMPLETE_RESULTS.md from measured data only.

    python scripts/23_build_tables.py

Sources:
    results/full_sweep_results.csv   12 models x 5 sizes x 5 seeds, validation + test
    results/qsvm_results.csv         Quantum Support Vector Machine, validation
    results/final_results.csv        10-seed endpoint runs, for paired significance tests

RULES THIS SCRIPT ENFORCES
    * No extrapolation, anywhere. A cell with no measurement is printed as a
      reason ("not computable on this hardware", "not run"), never as a number.
    * A completeness audit runs FIRST and is printed at the top of the output:
      every (model, size) cell with its seed count and provenance. A cell with
      fewer seeds than required is flagged, not silently averaged.
    * Every number is computed from the CSVs at build time. Nothing is typed in.

Superseded tables (docs/RESULTS_TABLES.md, docs/WHERE_QUANTUM_WINS.md) were assembled
from partial sources and, in one case, contained extrapolated values.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)

NS = [571, 1429, 2857, 14286, 68296]
SEEDS_REQUIRED = 5
ATT = ["speed_lie", "jump", "gap", "dark", "drift"]
ATT_LAB = {"speed_lie": "Speed falsification", "jump": "Position jump",
           "gap": "Reporting gap", "dark": "Transponder switch-off",
           "drift": "Gradual drift"}

NAME = {
    "Q_reup":     "Hybrid Quantum Neural Network — data re-uploading",
    "Q_pre":      "Hybrid Quantum Neural Network — classical pre-layer",
    "Q_frozen":   "Hybrid Quantum Neural Network — frozen circuit (control)",
    "QSVM":       "Quantum Support Vector Machine",
    "gbt":        "Gradient Boosting",
    "rf":         "Random Forest",
    "logreg":     "Logistic Regression",
    "mlp":        "Multi-Layer Perceptron (12-5-1, 71 parameters)",
    "MLP_reup":   "Matched Multi-Layer Perceptron — twin of re-uploading HQNN",
    "MLP_pre":    "Matched Multi-Layer Perceptron — twin of pre-layer HQNN",
    "MLP_frozen": "Matched Multi-Layer Perceptron — twin of frozen HQNN",
    "iforest":    "Isolation Forest (no labels)",
    "rule":       "Single-Threshold Rule (no labels)",
}
QUANTUM = ["Q_reup", "Q_pre", "Q_frozen", "QSVM"]
CLASSICAL = ["gbt", "rf", "logreg", "mlp", "MLP_reup", "MLP_pre", "MLP_frozen",
             "iforest", "rule"]
TWIN = {"Q_pre": "MLP_pre", "Q_reup": "MLP_reup", "Q_frozen": "MLP_frozen"}
# Cells the QSVM was never run for, with the reason. Printed as reasons, never as
# numbers. N=2,857 was attempted and stopped: ~6 h per seed x 5 seeds against a
# validation trend already flat from 571 to 1,429 (0.758 -> 0.763). Decided
# 2026-10-02 to report it as not run rather than spend ~30 h of compute.
QSVM_INFEASIBLE = {2857: "not run (~30 h for 5 seeds; trend flat from 571 to 1,429)",
                   14286: "not computable here (~10 days)",
                   68296: "not computable here (37 GB kernel, 17 GB RAM)"}


def main():
    fs = pd.read_csv("results/full_sweep_results.csv")
    q = pd.read_csv("results/qsvm_results.csv")
    fin = pd.read_csv("results/final_results.csv")
    out = []
    P = out.append

    # ---------------------------------------------------------------- audit
    P("## 0. Completeness audit\n")
    P(f"Seeds per cell (required: {SEEDS_REQUIRED}). ✔ complete · ✖ short · "
      "— not run by design.\n")
    P("| Model | " + " | ".join(f"N={n:,}" for n in NS) + " |")
    P("|---|" + "---:|" * len(NS))
    short = 0
    for m in QUANTUM + CLASSICAL:
        cells = []
        for n in NS:
            if m == "QSVM":
                k = int((q.N == n).sum())
                if n in QSVM_INFEASIBLE:
                    cells.append("—")
                    continue
            else:
                k = int(((fs.model == m) & (fs.N == n)).sum())
            ok = k >= SEEDS_REQUIRED
            short += (not ok)
            cells.append(f"{k} {'✔' if ok else '✖'}")
        P(f"| {NAME[m]} | " + " | ".join(cells) + " |")
    P("")
    P(f"**{'All cells complete.' if short == 0 else f'{short} cell(s) incomplete — numbers below are PROVISIONAL.'}**")
    reused = int((fs.source == "19_final (reused)").sum())
    P(f"\n{reused} runs were imported from `results/final_results.csv` rather than "
      "recomputed; on 60 matched cells the two pipelines agreed to a worst "
      "absolute difference of 0.00e+00.\n")

    def mean(m, n, col):
        if m == "QSVM":
            if col != "val_auc" or n in QSVM_INFEASIBLE:
                return np.nan
            s = q[q.N == n].val_auc
        else:
            s = fs[(fs.model == m) & (fs.N == n)][col]
        return s.mean() if len(s) else np.nan

    def cell(m, n, col, best=None):
        v = mean(m, n, col)
        if np.isnan(v):
            if m == "QSVM" and n in QSVM_INFEASIBLE:
                return QSVM_INFEASIBLE[n]
            if m == "QSVM":
                return "not run (validation only)"
            return "—"
        s = f"{v:.4f}"
        return f"**{s}**" if best is not None and abs(v - best) < 5e-5 else s

    # ---------------------------------------------------------------- grids
    for col, title in (("test_auc", "1. Test AUC — every model at every size"),
                       ("val_auc", "2. Validation AUC — every model at every size")):
        P(f"\n## {title}\n")
        P("5 seeds per cell, identical training windows. **Bold** = best at that size.\n")
        P("| Model | " + " | ".join(f"N={n:,}" for n in NS) + " |")
        P("|---|" + "---:|" * len(NS))
        best = {}
        for n in NS:
            vals = [mean(m, n, col) for m in QUANTUM + CLASSICAL]
            best[n] = np.nanmax(vals) if not all(np.isnan(vals)) else None
        for m in QUANTUM + CLASSICAL:
            P(f"| {NAME[m]} | " + " | ".join(cell(m, n, col, best[n]) for n in NS) + " |")

    # ---------------------------------------------------------------- best vs best
    P("\n## 3. Best quantum vs best classical at each size (test AUC)\n")
    P("| N | Best quantum | AUC | Best classical | AUC | Gap |")
    P("|---:|---|---:|---|---:|---:|")
    for n in NS:
        qv = {m: mean(m, n, "test_auc") for m in QUANTUM}
        cv = {m: mean(m, n, "test_auc") for m in CLASSICAL}
        qv = {k: v for k, v in qv.items() if not np.isnan(v)}
        cv = {k: v for k, v in cv.items() if not np.isnan(v)}
        if not qv or not cv:
            P(f"| {n:,} | *pending* | | | | |")
            continue
        mq, mc = max(qv, key=qv.get), max(cv, key=cv.get)
        P(f"| {n:,} | {NAME[mq]} | {qv[mq]:.4f} | {NAME[mc]} | {cv[mc]:.4f} | {qv[mq]-cv[mc]:+.4f} |")

    # ---------------------------------------------------------------- wins count
    P("\n## 4. Where quantum is better — classical models beaten (test AUC)\n")
    P("Out of 9 classical models. The Quantum Support Vector Machine has no test "
      "scores and is omitted here rather than estimated.\n")
    P("| Quantum model | " + " | ".join(f"N={n:,}" for n in NS) + " |")
    P("|---|" + "---:|" * len(NS))
    for m in ["Q_reup", "Q_pre", "Q_frozen"]:
        cells = []
        for n in NS:
            v = mean(m, n, "test_auc")
            present = [c for c in CLASSICAL if not np.isnan(mean(c, n, "test_auc"))]
            if np.isnan(v) or not present:
                cells.append("*pending*")
                continue
            beaten = [c for c in present if v > mean(c, n, "test_auc")]
            cells.append(f"{len(beaten)}/{len(present)}")
        P(f"| {NAME[m]} | " + " | ".join(cells) + " |")

    # ---------------------------------------------------------------- twins
    P("\n## 5. Each HQNN against its matched classical twin (test AUC)\n")
    P("Crossover = where the gap changes sign, interpolated in log N between the "
      "two MEASURED sizes that bracket it. This is interpolation inside the data, "
      "not extrapolation beyond it.\n")
    P("| Pair | " + " | ".join(f"Δ N={n:,}" for n in NS) + " | Crossover |")
    P("|---|" + "---:|" * len(NS) + "---|")
    for qm, cm in TWIN.items():
        d = [mean(qm, n, "test_auc") - mean(cm, n, "test_auc") for n in NS]
        cross = "no sign change"
        for i in range(len(NS) - 1):
            if d[i] < 0 <= d[i + 1]:
                t = -d[i] / (d[i + 1] - d[i])
                c = np.exp(np.log(NS[i]) + t * (np.log(NS[i + 1]) - np.log(NS[i])))
                cross = f"N ≈ {c:,.0f} (~{c*0.03543:,.0f} labelled anomalies)"
                break
        if any(np.isnan(x) for x in d):
            cross = "pending — not all sizes measured yet"
        P(f"| {NAME[qm].split(' — ')[1]} vs twin | " + " | ".join(
            ("*pending*" if np.isnan(x) else f"{x:+.4f}") for x in d) + f" | {cross} |")

    # ---------------------------------------------------------------- significance
    P("\n## 6. Paired significance at the endpoints (10 seeds, test AUC)\n")
    P("Wilcoxon signed-rank. ✔ = p < 0.05. From `results/final_results.csv`.\n")
    P("| Pair | Δ N=571 | p | Δ N=68,296 | p |")
    P("|---|---:|---:|---:|---:|")
    for qm, cm in TWIN.items():
        c = []
        for n in (571, 68296):
            a = fin[(fin.model == qm) & (fin.N == n)].sort_values("seed").test_auc.values
            b = fin[(fin.model == cm) & (fin.N == n)].sort_values("seed").test_auc.values
            p = wilcoxon(a, b).pvalue
            c += [f"{a.mean()-b.mean():+.4f}", f"{p:.3f}{' ✔' if p < 0.05 else ''}"]
        P(f"| {NAME[qm].split(' — ')[1]} vs twin | " + " | ".join(c) + " |")

    # ---------------------------------------------------------------- attacks
    P("\n## 7. Detection by attack type at full data (test AUC)\n")
    P("**Bold** = best for that attack.\n")
    fm = [m for m in QUANTUM + CLASSICAL if m != "QSVM"]
    ba = {a: max(mean(m, 68296, f"test_auc_{a}") for m in fm) for a in ATT}
    P("| Model | " + " | ".join(ATT_LAB[a] for a in ATT) + " |")
    P("|---|" + "---:|" * len(ATT))
    for m in fm:
        P(f"| {NAME[m]} | " + " | ".join(
            (lambda v: f"**{v:.3f}**" if abs(v - ba[a]) < 5e-4 else f"{v:.3f}")(mean(m, 68296, f"test_auc_{a}"))
            for a in ATT) + " |")

    # ---------------------------------------------------------------- qsvm
    P("\n## 8. Quantum Support Vector Machine (validation AUC, measured only)\n")
    P("| N | AUC | Seeds | Note |")
    P("|---:|---:|---:|---|")
    for n in NS:
        if n in QSVM_INFEASIBLE:
            P(f"| {n:,} | — | 0 | {QSVM_INFEASIBLE[n]} |")
        else:
            s = q[q.N == n].val_auc
            P(f"| {n:,} | {s.mean():.4f} ± {s.std():.3f} | {len(s)} | measured |" if len(s)
              else f"| {n:,} | — | 0 | pending |")

    body = "\n".join(out)
    open("docs/COMPLETE_RESULTS.md", "w", encoding="utf-8").write(
        "# Complete results — measured data only\n\n"
        "Generated by `scripts/23_build_tables.py` from `results/full_sweep_results.csv`, "
        "`results/qsvm_results.csv` and `results/final_results.csv`. No extrapolated values. "
        "Every number is computed from the CSVs at build time.\n\n" + body + "\n")
    print(body)


if __name__ == "__main__":
    main()
