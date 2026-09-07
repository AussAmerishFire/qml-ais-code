"""
10_classical_baselines.py
=========================
The classical side of the comparison. Five baselines, each answering a
different objection. See BASELINES.md for the justification of every choice
made here.

    1. rule    - percentile threshold on spd_mismatch_max   (0 labels, 0 params)
    2. iforest - Isolation Forest                            (0 labels)
    3. logreg  - Logistic regression                         (the linear floor)
    4. trees   - Gradient boosting + Random Forest           (the strong ceiling)
    5. mlp     - 8-5-1 neural net, 51 params                 (parameter-matched
                 to the L=2 8-qubit QNN, which has T=50)

RULES ENFORCED BY THIS SCRIPT (from BASELINES.md section 3):
  * Every hyperparameter is selected on the VALIDATION split only.
  * The TEST split is read exactly once, at the very end, for final numbers.
  * All models see the identical 8 features, identically scaled.
  * 5 seeds for every stochastic model; mean +/- std reported.
  * Metrics are ROC-AUC and average precision. Accuracy is NEVER reported --
    at a 3.5% anomaly rate, predicting "normal" for everything scores 96.5%.

Output: baseline_results.csv, baseline_per_attack.csv, and a printed report.
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# The eight features, and ONLY these eight. The dataset files also carry
# lat_mean, lon_mean, vessel_type and length -- these are metadata for
# interpreting results, NOT inputs. Feeding a model latitude and longitude
# would let it memorise the geographic locations where anomalies happened to
# be injected, which is leakage dressed up as a feature.
FEATS = [
    "sog_mean",
    "sog_max_jump",
    "cog_turn_max",
    "dt_max",
    "spd_mismatch_max",
    "crab_mean",
    "prev_silence_s",
    "crab_valid",
]

SEEDS = [0, 1, 2, 3, 4]  # 5 seeds, as declared in BASELINES.md
ATTACKS = ["speed_lie", "drift", "jump", "gap", "dark"]

# Operating point for precision/recall/F1: flag the top 3.5% of windows by
# score. Chosen to match the true anomaly rate, so every model is held to the
# same alert budget -- an analyst who can investigate 3.5% of traffic.
ALERT_RATE = 0.035


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def evaluate(y, score, attack=None):
    """All reported metrics for one set of scores.

    `score` is "how anomalous", higher = more anomalous. Every model below is
    made to produce this convention so the metrics code is shared.
    """
    out = {
        "auc": roc_auc_score(y, score),
        "ap": average_precision_score(y, score),  # PR-AUC; the honest metric
                                                  # under 3.5% class imbalance
    }

    # Fixed alert budget: flag the top ALERT_RATE fraction.
    k = max(1, int(round(len(score) * ALERT_RATE)))
    cut = np.partition(score, -k)[-k]
    pred = score >= cut
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    out["precision"] = prec
    out["recall"] = rec
    out["f1"] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Per-attack AUC. Computed against NORMALS ONLY plus that one attack type,
    # so each number answers "can this model separate normal traffic from THIS
    # kind of spoofing". Mixing attacks into one AUC hides that drift is hard
    # and jump is trivial.
    if attack is not None:
        for a in ATTACKS:
            m = (y == 0) | (attack == a)
            if (y[m] == 1).sum() >= 10:
                out[f"auc_{a}"] = roc_auc_score(y[m], score[m])
    return out


# ---------------------------------------------------------------------------
# MODEL FACTORIES
# Each returns a function fit(Xtr, ytr, seed) -> score_fn(X) -> ndarray
# ---------------------------------------------------------------------------

def make_rule(feature="spd_mismatch_max"):
    """Baseline 1 -- the operations-centre rule.

    No fitting, no labels, no parameters: the anomaly score IS the value of the
    single most discriminative physical inconsistency. spd_mismatch_max asks
    whether the vessel's reported speed agrees with the speed implied by its
    successive positions. A spoofer who edits position but not speed (or the
    reverse) breaks this.
    """
    idx = FEATS.index(feature)

    def fit(Xtr, ytr, seed):
        return lambda X: X[:, idx]

    return fit


def make_iforest(n_estimators, max_samples, contamination):
    """Baseline 2 -- Isolation Forest (Liu, Ting & Zhou 2008).

    Unsupervised: ytr is accepted and ignored. Isolation Forest builds random
    trees and measures how few splits it takes to isolate a point; outliers
    isolate quickly. Because it never sees a label, its performance is
    unaffected by label scarcity -- it is the flat reference line in the
    headline experiment.
    """

    def fit(Xtr, ytr, seed):
        m = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        ).fit(Xtr)
        # score_samples: HIGHER = more normal. Negate for our convention.
        return lambda X: -m.score_samples(X)

    return fit


def make_logreg(C, class_weight):
    """Baseline 3 -- logistic regression, the linear floor."""

    def fit(Xtr, ytr, seed):
        m = LogisticRegression(
            C=C, class_weight=class_weight, max_iter=2000, random_state=seed
        ).fit(Xtr, ytr)
        return lambda X: m.predict_proba(X)[:, 1]

    return fit


def make_gbt(max_iter, max_depth, learning_rate):
    """Baseline 4a -- gradient-boosted trees.

    Grinsztajn et al. (2022): tree ensembles still beat deep learning on
    typical tabular data. 8 features and 68k rows is exactly that regime, so
    this is the baseline most likely to win outright.
    """

    def fit(Xtr, ytr, seed):
        m = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        ).fit(Xtr, ytr)
        return lambda X: m.predict_proba(X)[:, 1]

    return fit


def make_rf(n_estimators, max_depth, class_weight):
    """Baseline 4b -- random forest."""

    def fit(Xtr, ytr, seed):
        m = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        ).fit(Xtr, ytr)
        return lambda X: m.predict_proba(X)[:, 1]

    return fit


def make_mlp(alpha, lr, oversample):
    """Baseline 5 -- THE parameter-matched comparison.

    hidden_layer_sizes=(5,) on 8 inputs with 1 output gives
        8*5 (input->hidden) + 5 (hidden bias) + 5 (hidden->output) + 1 (bias)
        = 51 trainable parameters
    against T = 50 for the 8-qubit StronglyEntanglingLayers circuit at L=2
    (48 circuit weights + 2 readout). A 2% edge to the classical model, which
    is the safe direction: we are not handicapping the baseline.

    `oversample` duplicates positives to reach the given minority fraction.
    MLPClassifier has no class_weight parameter, so this is how the classical
    net is given the same imbalance handling the tree models get for free.
    Resampling changes the training data, not the parameter count, so the
    match holds.
    """

    def fit(Xtr, ytr, seed):
        X, y = Xtr, ytr
        if oversample:
            rng = np.random.default_rng(seed)
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            want = int(len(neg) * oversample / (1 - oversample))
            if len(pos) and want > len(pos):
                pos = rng.choice(pos, size=want, replace=True)
            idx = rng.permutation(np.concatenate([pos, neg]))
            X, y = Xtr[idx], ytr[idx]
        m = MLPClassifier(
            hidden_layer_sizes=(5,),
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        ).fit(X, y)
        return lambda Z: m.predict_proba(Z)[:, 1]

    return fit


# ---------------------------------------------------------------------------
# SEARCH GRIDS
# Deliberately similar in size (8-12 configurations each) so that no model
# gets a larger tuning budget than another -- the specific failure mode
# Bowles, Ahmed & Schuld (2024) identified in the QML benchmarking literature.
# ---------------------------------------------------------------------------

GRIDS = {
    "rule": [({"feature": f}, make_rule) for f in
             ["spd_mismatch_max", "sog_max_jump", "dt_max"]],

    "iforest": [({"n_estimators": n, "max_samples": s, "contamination": c},
                 make_iforest)
                for n in (200, 500)
                for s in (256, 1024)
                for c in (0.02, 0.035, 0.05)],

    "logreg": [({"C": c, "class_weight": w}, make_logreg)
               for c in (0.01, 0.1, 1.0, 10.0, 100.0)
               for w in (None, "balanced")],

    "gbt": [({"max_iter": i, "max_depth": d, "learning_rate": r}, make_gbt)
            for i in (100, 300)
            for d in (3, 6, None)
            for r in (0.05, 0.1)],

    "rf": [({"n_estimators": n, "max_depth": d, "class_weight": w}, make_rf)
           for n in (300,)
           for d in (6, 12, None)
           for w in (None, "balanced", "balanced_subsample")],

    "mlp": [({"alpha": a, "lr": r, "oversample": o}, make_mlp)
            for a in (1e-4, 1e-2)
            for r in (0.003, 0.01)
            for o in (None, 0.2, 0.35)],
}

DETERMINISTIC = {"rule", "logreg"}  # seed changes nothing -> run one seed


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("CLASSICAL BASELINES")
    print("=" * 74)

    tr = pd.read_csv("dataset_train.csv")
    va = pd.read_csv("dataset_val.csv")
    te = pd.read_csv("dataset_test.csv")

    for name, d in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name:<6} {len(d):>7,} windows  "
              f"{int(d.label.sum()):>5,} anomalous ({d.label.mean()*100:.2f}%)  "
              f"{d.MMSI.nunique():>5,} vessels")

    # Guard: the whole design rests on vessels never crossing splits.
    assert not (set(tr.MMSI) & set(va.MMSI)), "train/val vessel overlap"
    assert not (set(tr.MMSI) & set(te.MMSI)), "train/test vessel overlap"
    assert not (set(va.MMSI) & set(te.MMSI)), "val/test vessel overlap"
    print("  [ok] zero vessel overlap between all three splits")

    Xtr, ytr = tr[FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xva, yva = va[FEATS].to_numpy(float), va.label.to_numpy(int)
    Xte, yte = te[FEATS].to_numpy(float), te.label.to_numpy(int)
    ate = te.attack.to_numpy()

    assert np.isfinite(Xtr).all() and np.isfinite(Xva).all() and np.isfinite(Xte).all()
    print(f"  [ok] {len(FEATS)} features, no NaN/inf, "
          f"range [{Xtr.min():.3f}, {Xtr.max():.3f}]")

    # ---- SELECTION PHASE: validation only, test never touched -------------
    print("\n" + "-" * 74)
    print("SELECTION  (validation split only -- test is not read here)")
    print("-" * 74)

    # Selection is expensive (the random-forest grid alone is ~7 minutes), so
    # the winning config per model is cached to disk. Deleting
    # baseline_selection.json forces a full re-search. The cache stores only
    # the validation-chosen hyperparameters -- no test information ever enters
    # it, so resuming cannot leak.
    cache_path = "baseline_selection.json"
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    chosen = {}
    for model, grid in GRIDS.items():
        factories = {json.dumps(p, sort_keys=True): fac for p, fac in grid}

        if model in cache:
            key = json.dumps(cache[model]["params"], sort_keys=True)
            chosen[model] = (cache[model]["val_auc"],
                             cache[model]["params"], factories[key])
            print(f"  {model:<8} {len(grid):>2} configs  "
                  f"best val AUC {cache[model]['val_auc']:.4f}  (cached)")
            print(f"           -> {json.dumps(cache[model]['params'])}")
            continue

        t0 = time.time()
        best = None
        for params, factory in grid:
            fit = factory(**params)
            seeds = [0] if model in DETERMINISTIC else SEEDS
            aucs = []
            for s in seeds:
                score_fn = fit(Xtr, ytr, s)
                aucs.append(roc_auc_score(yva, score_fn(Xva)))
            m = float(np.mean(aucs))
            if best is None or m > best[0]:
                best = (m, params, factory)
        chosen[model] = best
        cache[model] = {"val_auc": best[0], "params": best[1]}
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  {model:<8} {len(grid):>2} configs  "
              f"best val AUC {best[0]:.4f}  ({time.time()-t0:5.1f}s)")
        print(f"           -> {json.dumps(best[1])}")

    # ---- FINAL EVALUATION: test read once ---------------------------------
    print("\n" + "-" * 74)
    print("FINAL EVALUATION  (test split, read once)")
    print("-" * 74)

    rows, per_attack_rows = [], []
    for model, (val_auc, params, factory) in chosen.items():
        fit = factory(**params)
        seeds = [0] if model in DETERMINISTIC else SEEDS
        runs = []
        for s in seeds:
            score_fn = fit(Xtr, ytr, s)
            runs.append(evaluate(yte, score_fn(Xte), ate))

        agg = {"model": model, "config": json.dumps(params),
               "val_auc": round(val_auc, 4), "n_seeds": len(seeds)}
        for k in runs[0]:
            v = [r[k] for r in runs if k in r]
            agg[k] = round(float(np.mean(v)), 4)
            agg[k + "_std"] = round(float(np.std(v)), 4)
        rows.append(agg)

        for a in ATTACKS:
            key = f"auc_{a}"
            if key in agg:
                per_attack_rows.append({"model": model, "attack": a,
                                        "auc": agg[key], "std": agg[key + "_std"]})

    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    res.to_csv("baseline_results.csv", index=False)
    pa = pd.DataFrame(per_attack_rows)
    pa.to_csv("baseline_per_attack.csv", index=False)

    print(f"\n{'model':<9}{'test AUC':>18}{'PR-AUC':>18}"
          f"{'prec@3.5%':>12}{'recall':>10}")
    for _, r in res.iterrows():
        print(f"  {r.model:<9} {r.auc:.4f} +/- {r.auc_std:.4f}   "
              f"{r.ap:.4f} +/- {r.ap_std:.4f}   "
              f"{r.precision:>8.3f}  {r.recall:>8.3f}")

    print("\nPer-attack AUC (normals vs that attack only):")
    piv = pa.pivot(index="model", columns="attack", values="auc")
    piv = piv.reindex(columns=[a for a in ATTACKS if a in piv.columns])
    print(piv.sort_values(ATTACKS[1], ascending=False).to_string(
        float_format=lambda v: f"{v:.3f}"))

    print("\nwrote baseline_results.csv, baseline_per_attack.csv")


if __name__ == "__main__":
    main()
