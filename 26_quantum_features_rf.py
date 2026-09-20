"""
26_quantum_features_rf.py
=========================
Do features computed by a quantum circuit improve the models industry actually
uses? Random Forest and Gradient Boosting -- the two best models in this study,
and the standard for engineered tabular features (Grinsztajn et al. 2022) --
trained with and without quantum features, against classical random features of
the same count.

    python 26_quantum_features_rf.py

QUANTUM FEATURES
    Projected quantum features (Huang et al. 2021, Nature Communications): the
    single-qubit expectation values <X_i>, <Y_i>, <Z_i> of the IQP-embedded state.
    This is the SAME 8-qubit feature map as the QSVM (20_qsvm.py) and the
    geometric-difference test (25_geometric_difference.py). Constant features are
    dropped -- with one IQP repetition every <Z_i> is exactly 0, because the
    embedding's entangling part is diagonal -- leaving D features.

ARMS -- same model, same hyperparameters, same subsample, same seed
    base   the 12 engineered features (the study's Random Forest / Gradient Boosting)
    +Q     12 + D quantum features
    +RFF   12 + D random Fourier features  cos(Wz + b)   classical random nonlinear
    +TANH  12 + D random tanh features     tanh(Wz + b)  a second classical control
    +PCA   12 + the 8 PCA inputs z the circuit receives   rules out "the rotation
                                                           alone helped"
    Every added feature is computed from the SAME 8 numbers z.

    A gain for +Q over base is credited to the quantum circuit ONLY if +Q also
    beats +RFF, +TANH and +PCA. Otherwise it is an extra-features effect.

PROTOCOL
    * Hyperparameters fixed at the tuned values in baseline_selection.json for
      every arm. No arm is re-tuned, so none gets extra selection budget.
    * Five training sizes, 10 seeds each (Wilcoxon floor p = 0.002), subsampled
      exactly as in the sweep (rng 1000 + seed, anomaly rate held fixed).
    * PCA 12 -> 8 and the [0, pi] scaling are fitted ONCE on the full training
      split, label-free; validation and test are only transformed. (The
      engineered features were already scaled on training statistics, so every
      arm sees the same unlabelled-data information.)
    * Reported on validation and test. Nothing is selected on either.
    * Checkpointed per row to qfeat_results.csv; re-running resumes.
"""

import importlib
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp          # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import pennylane as qml          # noqa: E402
from scipy.stats import wilcoxon                      # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              RandomForestClassifier)
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

sw = importlib.import_module("16_sweep")                 # FEATS, ATTACKS, subsample
import json                                              # noqa: E402

SIZES = [571, 1429, 2857, 14286, 68296]
SEEDS = list(range(10))
ARMS = ["base", "+Q", "+RFF", "+TANH", "+PCA"]
MODELS = ["rf", "gbt"]
OUT = "qfeat_results.csv"
NQ = 8
BLOCK = 8192

_dev = qml.device("default.qubit", wires=NQ)


@qml.qnode(_dev, interface="jax", diff_method=None)
def _state(x):
    qml.IQPEmbedding(x, wires=range(NQ), n_repeats=1)     # identical to 20_qsvm.py
    return qml.state()


_state_j = jax.jit(_state)


def paulis_from_state(psi):
    """<X_i>, <Y_i>, <Z_i> for every qubit, from a batch of 8-qubit states.

    Wire 0 is the most significant bit in PennyLane's ordering, so axis 1 of the
    reshaped state is wire 0.
    """
    b = psi.shape[0]
    t = psi.reshape((b,) + (2,) * NQ)
    out = []
    for i in range(NQ):
        a = np.moveaxis(t, 1 + i, -1).reshape(b, -1, 2)
        rho = np.einsum("bka,bkc->bac", a, a.conj())      # reduced state of qubit i
        out.append(np.stack([2 * rho[:, 0, 1].real,       # <X>
                             -2 * rho[:, 0, 1].imag,      # <Y>
                             (rho[:, 0, 0] - rho[:, 1, 1]).real], 1))  # <Z>
    return np.concatenate(out, 1)                          # (b, 24): X0 Y0 Z0 X1 ...


def quantum_features(Z):
    out = np.empty((len(Z), 3 * NQ))
    for s in range(0, len(Z), BLOCK):
        e = min(s + BLOCK, len(Z))
        blk = Z[s:e]
        pad = BLOCK - (e - s)
        if pad:
            blk = np.concatenate([blk, np.zeros((pad, NQ))])
        out[s:e] = paulis_from_state(np.asarray(_state_j(jnp.array(blk))))[: e - s]
    return out


def self_check(Z):
    """Verify the hand-written Pauli read-out against PennyLane on 3 inputs."""
    @qml.qnode(_dev)
    def ref(x):
        qml.IQPEmbedding(x, wires=range(NQ), n_repeats=1)
        return [qml.expval(P(i)) for i in range(NQ) for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]
    mine = quantum_features(Z[:3])
    theirs = np.array([np.array(ref(np.asarray(z))) for z in Z[:3]])
    err = np.abs(mine - theirs).max()
    assert err < 1e-8, f"Pauli read-out mismatch {err}"
    print(f"self-check: Pauli read-out matches PennyLane (max err {err:.1e})", flush=True)


def keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("sleep prevention: active", flush=True)
    except Exception:
        pass


def main():
    keep_awake()
    cfg = json.load(open("baseline_selection.json"))
    tr, va, te = (pd.read_csv(f"dataset_{s}.csv") for s in ("train", "val", "test"))
    F = sw.FEATS
    X, y = tr[F].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[F].to_numpy(float), va.label.to_numpy(int)
    Xt, yt, at = te[F].to_numpy(float), te.label.to_numpy(int), te.attack.to_numpy()
    rate = y.mean()

    # ---- the 8 inputs the circuit sees: PCA + [0, pi], fitted on train only
    pca = PCA(n_components=NQ, random_state=0).fit(X)
    P, Pv, Pt = pca.transform(X), pca.transform(Xv), pca.transform(Xt)
    lo, hi = P.min(0), P.max(0)
    sc = lambda A: np.clip((A - lo) / (hi - lo), 0, 1) * np.pi   # noqa: E731
    Z, Zv, Zt = sc(P), sc(Pv), sc(Pt)

    self_check(Z)
    t0 = time.time()
    QF = [quantum_features(A) for A in (Z, Zv, Zt)]
    keep = QF[0].std(0) > 1e-9
    QF = [A[:, keep] for A in QF]
    D = int(keep.sum())
    print(f"quantum features: {D} of {3*NQ} non-constant, "
          f"{len(Z)+len(Zv)+len(Zt):,} windows in {time.time()-t0:.0f}s\n", flush=True)

    sq = ((Z[:2000, None] - Z[None, :2000]) ** 2).sum(-1)
    gamma0 = 1.0 / np.median(sq[np.triu_indices(2000, 1)])      # median heuristic

    def extra(arm, seed, which):
        A = (Z, Zv, Zt)[which]
        if arm == "base":
            return None
        if arm == "+Q":
            return QF[which]
        if arm == "+PCA":
            return A
        rng = np.random.default_rng(50_000 + seed)                 # fresh per seed
        if arm == "+RFF":
            W = rng.normal(0, np.sqrt(2 * gamma0), (NQ, D))
            b = rng.uniform(0, 2 * np.pi, D)
            return np.sqrt(2 / D) * np.cos(A @ W + b)
        W = rng.normal(0, 1 / np.sqrt(NQ), (NQ, D))                 # +TANH
        b = rng.normal(0, 1, D)
        return np.tanh(A @ W + b)

    def build(model, seed):
        if model == "rf":
            return RandomForestClassifier(**cfg["rf"]["params"], random_state=seed, n_jobs=-1)
        return HistGradientBoostingClassifier(**cfg["gbt"]["params"], random_state=seed)

    rows, done = [], set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        rows = d.to_dict("records")
        done = set(zip(d.model, d.arm, d.N.astype(int), d.seed.astype(int)))
    print(f"resuming with {len(done)} rows on disk", flush=True)

    idx_all = np.arange(len(y))
    t_all = time.time()
    total = len(SIZES) * len(SEEDS) * len(ARMS) * len(MODELS)
    for N in SIZES:
        for seed in SEEDS:
            # same subsample as the sweep: subsample row indices, not rows
            k, _ = sw.subsample(idx_all[:, None], y, N, seed, rate)
            k = k.ravel()
            for arm in ARMS:
                parts = [extra(arm, seed, w) for w in (0, 1, 2)]
                Xs = X[k] if parts[0] is None else np.hstack([X[k], parts[0][k]])
                XV = Xv if parts[1] is None else np.hstack([Xv, parts[1]])
                XT = Xt if parts[2] is None else np.hstack([Xt, parts[2]])
                for model in MODELS:
                    if (model, arm, N, seed) in done:
                        continue
                    m = build(model, seed).fit(Xs, y[k])
                    sv, st = m.predict_proba(XV)[:, 1], m.predict_proba(XT)[:, 1]
                    r = {"model": model, "arm": arm, "N": N, "seed": seed,
                         "n_feats": Xs.shape[1],
                         "val_auc": roc_auc_score(yv, sv), "val_ap": average_precision_score(yv, sv),
                         "test_auc": roc_auc_score(yt, st), "test_ap": average_precision_score(yt, st)}
                    for a in sw.ATTACKS:
                        mm = (yt == 0) | (at == a)
                        r[f"test_auc_{a}"] = roc_auc_score(yt[mm], st[mm])
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(OUT, index=False)
            n = len(rows)
            print(f"  [{(time.time()-t_all)/60:5.1f} min] {n}/{total}  N={N} seed{seed}", flush=True)

    # ---- summary: +Q against every other arm, paired over seeds, test AUC
    d = pd.DataFrame(rows)
    print("\n" + "=" * 78 + "\nTest AUC by arm (mean of 10 seeds)\n" + "=" * 78)
    print(d.pivot_table(index=["model", "N"], columns="arm", values="test_auc").round(4)[ARMS].to_string())
    print("\n+Q minus each arm, test AUC, Wilcoxon over 10 paired seeds")
    for model in MODELS:
        for N in SIZES:
            q = d[(d.model == model) & (d.arm == "+Q") & (d.N == N)].sort_values("seed").test_auc.values
            line = f"  {model:3s} N={N:>6}"
            for arm in ("base", "+RFF", "+TANH", "+PCA"):
                o = d[(d.model == model) & (d.arm == arm) & (d.N == N)].sort_values("seed").test_auc.values
                p = wilcoxon(q, o).pvalue if np.any(q != o) else 1.0
                line += f"   vs {arm:5s} {q.mean()-o.mean():+.4f} p={p:.3f}{'*' if p < .05 else ' '}"
            print(line)
    print(f"\n[{(time.time()-t_all)/60:.1f} min]  wrote {OUT}")


if __name__ == "__main__":
    main()
