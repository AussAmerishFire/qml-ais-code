"""
scripts/19_final.py
===========
The last data collection. Two things at once, checkpointed per run.

A. STATISTICAL POWER AT THE ENDPOINTS
   The sweep's headline claim -- that the quantum-minus-classical gap inverts
   between N=571 and N=68,296 -- was run at 5 seeds. A two-sided Wilcoxon
   signed-rank test on 5 pairs has a minimum attainable p-value of 2/2^5 =
   0.0625, so it CANNOT reach p<0.05 no matter what the data shows. That was a
   flaw in the protocol as written (docs/BASELINES.md section 3, rule 6), not in the
   result. At 10 pairs the minimum is 2/2^10 = 0.002.

   This re-runs both endpoints at 10 seeds for all three matched pairs.

B. THE TEST SPLIT, READ ONCE FOR THE QUANTUM MODELS
   Every quantum number so far is validation. Each endpoint run here scores
   BOTH validation and test, so the same fits serve both purposes and the test
   split is opened exactly once, after every modelling decision is locked.

   The classical baselines already read test (scripts/10_classical_baselines.py). That
   is a second read of the same split, and the paper must say so plainly. It
   does not compromise anything: no selection was ever performed on test, which
   is what the hold-out rule actually protects.

Hyperparameters, architectures and step budgets are taken unchanged from
scripts/16_sweep.py -- nothing is re-tuned here. Re-tuning after seeing the sweep would
be selection on the outcome.

    python scripts/19_final.py
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sw = _load("scripts/16_sweep.py", "sw")

ENDPOINTS = [571, 68296]
SEEDS = list(range(10))
RESULTS = "results/final_results.csv"


def load_done():
    if not os.path.exists(RESULTS):
        return set(), []
    d = pd.read_csv(RESULTS)
    return set(zip(d.model, d.N.astype(int), d.seed.astype(int))), d.to_dict("records")


def main():
    sw.keep_awake()
    tr = pd.read_csv("data/dataset_train.csv")
    va = pd.read_csv("data/dataset_val.csv")
    te = pd.read_csv("data/dataset_test.csv")
    X, y = tr[sw.FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[sw.FEATS].to_numpy(float), va.label.to_numpy(int)
    Xt, yt = te[sw.FEATS].to_numpy(float), te.label.to_numpy(int)
    at = te.attack.to_numpy()
    rate = y.mean()

    best = json.load(open("results/sweep_best_cfg.json"))
    done, rows = load_done()
    t0 = time.time()
    print(f"resuming with {len(done)} runs on disk")
    print(f"selected configs: {best}\n", flush=True)

    import jax.numpy as jnp
    from sklearn.metrics import average_precision_score, roc_auc_score

    def record(model, N, seed, T, sv, st, secs):
        k = max(1, int(round(len(st) * 0.035)))
        pred = st >= np.partition(st, -k)[-k]
        tp = int((pred & (yt == 1)).sum())
        r = {"model": model, "N": N, "seed": seed, "T": T, "secs": round(secs, 1),
             "val_auc": roc_auc_score(yv, sv),
             "test_auc": roc_auc_score(yt, st),
             "test_ap": average_precision_score(yt, st),
             "test_prec": tp / max(1, pred.sum()),
             "test_recall": tp / (yt == 1).sum()}
        for a in sw.ATTACKS:
            m = (yt == 0) | (at == a)
            r[f"test_auc_{a}"] = roc_auc_score(yt[m], st[m])
        rows.append(r)
        pd.DataFrame(rows).to_csv(RESULTS, index=False)
        done.add((model, N, seed))
        return r

    for N in ENDPOINTS:
        for seed in SEEDS:
            Xs, ys = sw.subsample(X, y, N, seed, rate)

            for name, a in sw.QARCH.items():
                if (name, N, seed) in done:
                    continue
                t1 = time.time()
                cfg = sw.QGRID[best[name]]
                p, pred = sw.q_train(a, Xs, ys, cfg, seed, sw.SWEEP_STEPS, None)
                r = record(name, N, seed, sw.q_params(a),
                           np.asarray(pred(p, jnp.array(Xv))),
                           np.asarray(pred(p, jnp.array(Xt))), time.time() - t1)
                print(f"  [{time.time()-t0:6.0f}s] N={N:>6} s{seed} {name:<10} "
                      f"val {r['val_auc']:.4f}  test {r['test_auc']:.4f} "
                      f"({r['secs']:.0f}s)", flush=True)

            for name, hl in sw.CARCH.items():
                if (name, N, seed) in done:
                    continue
                t1 = time.time()
                cfg = sw.CGRID[best[name]]
                m = sw.c_train(hl, Xs, ys, cfg, seed, sw.SWEEP_STEPS, False)
                r = record(name, N, seed, sw.CPARAMS[name],
                           m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1],
                           time.time() - t1)
                print(f"  [{time.time()-t0:6.0f}s] N={N:>6} s{seed} {name:<10} "
                      f"val {r['val_auc']:.4f}  test {r['test_auc']:.4f}",
                      flush=True)

    # ---------------- report ----------------
    d = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("PAIRED TESTS AT 10 SEEDS  (min attainable p = 0.002, vs 0.0625 at n=5)")
    print("=" * 78)
    for q, c in sw.TWIN.items():
        for N in ENDPOINTS:
            for split in ("val_auc", "test_auc"):
                A = d[(d.model == q) & (d.N == N)].sort_values("seed")[split].values
                B = d[(d.model == c) & (d.N == N)].sort_values("seed")[split].values
                k = min(len(A), len(B))
                if k < 6:
                    continue
                p = wilcoxon(A[:k], B[:k]).pvalue
                diff = A[:k].mean() - B[:k].mean()
                sig = "SIGNIFICANT" if p < 0.05 else "n.s."
                print(f"  {q:<10} - {c:<11} N={N:>6} {split:<9} "
                      f"{diff:>+.4f}  p={p:.4f}  n={k}  {sig}")

    print("\n" + "=" * 78)
    print("TEST-SET RESULTS  (read once, after all decisions locked)")
    print("=" * 78)
    for N in ENDPOINTS:
        print(f"\n  N = {N:,}")
        s = d[d.N == N].groupby("model").agg(
            T=("T", "first"), auc=("test_auc", "mean"), sd=("test_auc", "std"),
            ap=("test_ap", "mean"), drift=("test_auc_drift", "mean"),
            dark=("test_auc_dark", "mean")).sort_values("auc", ascending=False)
        print(f"    {'model':<12}{'T':>5}{'test AUC':>18}{'PR-AUC':>9}"
              f"{'drift':>8}{'dark':>8}")
        for m, r in s.iterrows():
            print(f"    {m:<12}{int(r['T']):>5}{r['auc']:>11.4f} +/-{r['sd']:.3f}"
                  f"{r['ap']:>9.4f}{r['drift']:>8.3f}{r['dark']:>8.3f}")

    print(f"\n[{(time.time()-t0)/3600:.2f} h]  wrote {RESULTS}")


if __name__ == "__main__":
    main()
