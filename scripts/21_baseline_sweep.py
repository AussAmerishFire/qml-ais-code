"""
scripts/21_baseline_sweep.py
====================
Fills the one real gap in the scarcity sweep: the four unconstrained classical
baselines -- Gradient Boosting, Logistic Regression, Multi-Layer Perceptron and
the Single-Threshold Rule -- were only ever run at full data. Without them, the
claim "quantum does better than classical at size N" can only be checked against
the matched twins, Random Forest and Isolation Forest.

These are RUN, not extrapolated. Classical models are cheap; there is no reason
to estimate what can be measured in minutes.

Protocol matches scripts/16_sweep.py exactly where it can:
  * same subsample() -- seeds 0-4 draw the SAME training windows at each N
  * anomaly rate held at the natural 3.543%
  * scored on validation, like every other sweep cell

Hyperparameters are the ones selected for these models on the 12-feature data
(results/baseline_selection.json), applied unchanged at every N. They are the models'
own tuned configurations, not the fixed 640-step protocol of the matched twins:
these are the "best-effort classical" reference, which is what they represent
in the paper.

    python scripts/21_baseline_sweep.py
"""

import importlib.util
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

spec = importlib.util.spec_from_file_location("sw", "scripts/16_sweep.py")
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)

RESULTS = "results/baseline_sweep_results.csv"
SEEDS = [0, 1, 2, 3, 4]


def main():
    cfg = json.load(open("results/baseline_selection.json"))
    tr = pd.read_csv("data/dataset_train.csv")
    va = pd.read_csv("data/dataset_val.csv")
    X, y = tr[sw.FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[sw.FEATS].to_numpy(float), va.label.to_numpy(int)
    av = va.attack.to_numpy()
    rate = y.mean()
    rule_idx = sw.FEATS.index(cfg["rule"]["params"]["feature"])

    rows, done = [], set()
    if os.path.exists(RESULTS):
        d = pd.read_csv(RESULTS)
        rows = d.to_dict("records")
        done = set(zip(d.model, d.N.astype(int), d.seed.astype(int)))

    def fit_score(name, Xs, ys, seed):
        if name == "rule":
            return Xv[:, rule_idx]                    # no fitting, no labels
        if name == "logreg":
            p = cfg["logreg"]["params"]
            return LogisticRegression(C=p["C"], class_weight=p["class_weight"],
                                      max_iter=2000, random_state=seed
                                      ).fit(Xs, ys).predict_proba(Xv)[:, 1]
        if name == "gbt":
            return HistGradientBoostingClassifier(**cfg["gbt"]["params"],
                                                  random_state=seed
                                                  ).fit(Xs, ys).predict_proba(Xv)[:, 1]
        if name == "mlp":
            p = cfg["mlp"]["params"]
            rng = np.random.default_rng(seed)
            Xa, ya = Xs, ys
            if p["oversample"]:
                pos, neg = np.where(ys == 1)[0], np.where(ys == 0)[0]
                want = int(len(neg) * p["oversample"] / (1 - p["oversample"]))
                if len(pos) and want > len(pos):
                    pos = rng.choice(pos, size=want, replace=True)
                k = rng.permutation(np.concatenate([pos, neg]))
                Xa, ya = Xs[k], ys[k]
            return MLPClassifier(hidden_layer_sizes=(5,), alpha=p["alpha"],
                                 learning_rate_init=p["lr"], max_iter=500,
                                 early_stopping=True, n_iter_no_change=20,
                                 random_state=seed
                                 ).fit(Xa, ya).predict_proba(Xv)[:, 1]

    t0 = time.time()
    for N in sw.SWEEP_N:
        for seed in SEEDS:
            Xs, ys = sw.subsample(X, y, N, seed, rate)
            for name in ("gbt", "logreg", "mlp", "rule"):
                if (name, N, seed) in done:
                    continue
                s = fit_score(name, Xs, ys, seed)
                r = {"model": name, "N": N, "seed": seed,
                     "auc": roc_auc_score(yv, s),
                     "ap": average_precision_score(yv, s)}
                for a in sw.ATTACKS:
                    m = (yv == 0) | (av == a)
                    r[f"auc_{a}"] = roc_auc_score(yv[m], s[m])
                rows.append(r)
                pd.DataFrame(rows).to_csv(RESULTS, index=False)
            print(f"  [{time.time()-t0:5.0f}s] N={N:>6} seed{seed} done", flush=True)

    d = pd.DataFrame(rows)
    print("\nvalidation AUC, mean of 5 seeds")
    print(d.pivot_table(index="model", columns="N", values="auc").round(4).to_string())


if __name__ == "__main__":
    main()
