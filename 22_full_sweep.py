"""
22_full_sweep.py
================
The complete, uniformly-run grid: every model at every training-set size, scored
on BOTH validation and test, 5 seeds, identical training windows throughout.

    python 22_full_sweep.py

WHY THIS EXISTS
    Before this, the grid was assembled from four scripts with different
    coverage: the scarcity sweep (validation only), the endpoint runs (test at
    two sizes), the classical baseline sweep (validation only), and the quantum
    kernel (two sizes, three seeds, with the rest extrapolated). Every number was
    real except the extrapolations, but no single table could be built from one
    consistent protocol. This fixes that.

REUSE, VERIFIED RATHER THAN ASSUMED
    19_final.py already trained the six quantum and matched-twin models at
    N = 571 and N = 68,296 with seeds 0-9, scoring validation and test. Those
    runs were checked against the original sweep on 60 matched (model, N, seed)
    cells: worst absolute difference in validation AUC 0.00e+00. The pipeline is
    exactly deterministic, so seeds 0-4 of those runs ARE what this script would
    compute. They are imported, not recomputed, saving ~3 h.

    Everything else is run fresh.

PROTOCOLS
    Quantum and matched-twin models: exactly 16_sweep.py -- selected configs from
    sweep_best_cfg.json, 640 gradient steps at every N, no early stopping.

    Unconstrained classical baselines (Gradient Boosting, Random Forest,
    Logistic Regression, Multi-Layer Perceptron, Isolation Forest, Single-
    Threshold Rule): their own configurations tuned on the 12-feature data
    (baseline_selection.json), applied unchanged at every N. These represent
    best-effort classical performance, which is their role in the paper.

    Note: the original sweep ran Isolation Forest with 200 trees; the tuned
    12-feature configuration uses 500. This script uses the tuned one, so its
    Isolation Forest numbers are the correct reference and may differ slightly
    from sweep_results.csv.

The Quantum Support Vector Machine is NOT here -- see 20_qsvm.py. Its cost is
quadratic in N and it runs on its own schedule.
"""

import importlib.util
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

spec = importlib.util.spec_from_file_location("sw", "16_sweep.py")
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)
import jax.numpy as jnp  # noqa: E402  (after 16_sweep enables x64)

RESULTS = "full_sweep_results.csv"
SEEDS = [0, 1, 2, 3, 4]
QUANTUM_AND_TWINS = list(sw.QARCH) + list(sw.CARCH)
BASELINES = ["gbt", "rf", "logreg", "mlp", "iforest", "rule"]
REUSE_N = (571, 68296)          # already computed, verified identical


def main():
    sw.keep_awake()
    bcfg = json.load(open("baseline_selection.json"))
    qcfg = json.load(open("sweep_best_cfg.json"))

    tr = pd.read_csv("dataset_train.csv")
    va = pd.read_csv("dataset_val.csv")
    te = pd.read_csv("dataset_test.csv")
    X, y = tr[sw.FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[sw.FEATS].to_numpy(float), va.label.to_numpy(int)
    Xt, yt = te[sw.FEATS].to_numpy(float), te.label.to_numpy(int)
    av, at = va.attack.to_numpy(), te.attack.to_numpy()
    rate = y.mean()
    rule_idx = sw.FEATS.index(bcfg["rule"]["params"]["feature"])

    rows, done = [], set()
    if os.path.exists(RESULTS):
        d = pd.read_csv(RESULTS)
        rows, done = d.to_dict("records"), set(zip(d.model, d.N.astype(int), d.seed.astype(int)))

    # ---- import the verified-identical endpoint runs -----------------------
    if not any(r.get("source") == "19_final (reused)" for r in rows):
        f = pd.read_csv("final_results.csv")
        f = f[f.N.isin(REUSE_N) & f.seed.isin(SEEDS) & f.model.isin(QUANTUM_AND_TWINS)]
        for _, r in f.iterrows():
            row = {"model": r.model, "N": int(r.N), "seed": int(r.seed),
                   "val_auc": r.val_auc, "val_ap": np.nan,
                   "test_auc": r.test_auc, "test_ap": r.test_ap,
                   "source": "19_final (reused)"}
            for a in sw.ATTACKS:
                row[f"test_auc_{a}"] = r[f"test_auc_{a}"]
            rows.append(row)
            done.add((r.model, int(r.N), int(r.seed)))
        pd.DataFrame(rows).to_csv(RESULTS, index=False)
        print(f"imported {len(f)} verified endpoint runs from final_results.csv", flush=True)

    def score_row(model, N, seed, sv, st):
        row = {"model": model, "N": N, "seed": seed,
               "val_auc": roc_auc_score(yv, sv), "val_ap": average_precision_score(yv, sv),
               "test_auc": roc_auc_score(yt, st), "test_ap": average_precision_score(yt, st),
               "source": "22_full_sweep"}
        for a in sw.ATTACKS:
            m = (yt == 0) | (at == a)
            row[f"test_auc_{a}"] = roc_auc_score(yt[m], st[m])
        return row

    def baseline(name, Xs, ys, seed):
        if name == "rule":
            return Xv[:, rule_idx], Xt[:, rule_idx]
        if name == "iforest":
            p = dict(bcfg["iforest"]["params"])
            p["max_samples"] = min(p["max_samples"], len(ys))
            m = IsolationForest(**p, random_state=seed, n_jobs=-1).fit(Xs)
            return -m.score_samples(Xv), -m.score_samples(Xt)
        if name == "rf":
            m = RandomForestClassifier(**bcfg["rf"]["params"], random_state=seed,
                                       n_jobs=-1).fit(Xs, ys)
        elif name == "gbt":
            m = HistGradientBoostingClassifier(**bcfg["gbt"]["params"],
                                               random_state=seed).fit(Xs, ys)
        elif name == "logreg":
            p = bcfg["logreg"]["params"]
            m = LogisticRegression(C=p["C"], class_weight=p["class_weight"],
                                   max_iter=2000, random_state=seed).fit(Xs, ys)
        elif name == "mlp":
            p = bcfg["mlp"]["params"]
            rng = np.random.default_rng(seed)
            Xa, ya = Xs, ys
            if p["oversample"]:
                pos, neg = np.where(ys == 1)[0], np.where(ys == 0)[0]
                want = int(len(neg) * p["oversample"] / (1 - p["oversample"]))
                if len(pos) and want > len(pos):
                    pos = rng.choice(pos, size=want, replace=True)
                k = rng.permutation(np.concatenate([pos, neg]))
                Xa, ya = Xs[k], ys[k]
            m = MLPClassifier(hidden_layer_sizes=(5,), alpha=p["alpha"],
                              learning_rate_init=p["lr"], max_iter=500,
                              early_stopping=True, n_iter_no_change=20,
                              random_state=seed).fit(Xa, ya)
        return m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]

    t0 = time.time()
    total = len(sw.SWEEP_N) * len(SEEDS) * (len(QUANTUM_AND_TWINS) + len(BASELINES))
    for N in sw.SWEEP_N:
        for seed in SEEDS:
            Xs, ys = sw.subsample(X, y, N, seed, rate)

            for name in BASELINES:
                if (name, N, seed) in done:
                    continue
                sv, st = baseline(name, Xs, ys, seed)
                rows.append(score_row(name, N, seed, sv, st))
                done.add((name, N, seed))
                pd.DataFrame(rows).to_csv(RESULTS, index=False)

            for name in QUANTUM_AND_TWINS:
                if (name, N, seed) in done:
                    continue
                t1 = time.time()
                if name in sw.QARCH:
                    p, pred = sw.q_train(sw.QARCH[name], Xs, ys,
                                         sw.QGRID[qcfg[name]], seed, sw.SWEEP_STEPS, None)
                    sv = np.asarray(pred(p, jnp.array(Xv)))
                    st = np.asarray(pred(p, jnp.array(Xt)))
                else:
                    m = sw.c_train(sw.CARCH[name], Xs, ys, sw.CGRID[qcfg[name]],
                                   seed, sw.SWEEP_STEPS, False)
                    sv, st = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
                r = score_row(name, N, seed, sv, st)
                rows.append(r)
                done.add((name, N, seed))
                pd.DataFrame(rows).to_csv(RESULTS, index=False)
                print(f"  [{(time.time()-t0)/60:6.1f} min] {len(done):>3}/{total}  "
                      f"N={N:>6} s{seed} {name:<11} val {r['val_auc']:.4f} "
                      f"test {r['test_auc']:.4f}  ({time.time()-t1:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    print(f"\nDONE {len(d)}/{total} runs in {(time.time()-t0)/3600:.2f} h")
    print("\nTEST AUC, mean of 5 seeds")
    print(d.pivot_table(index="model", columns="N", values="test_auc").round(4).to_string())


if __name__ == "__main__":
    main()
