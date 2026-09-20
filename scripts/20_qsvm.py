"""
scripts/20_qsvm.py
==========
A quantum kernel method (QSVM) at the three small-N points, to test whether the
scarcity deficit found for variational models is a property of VARIATIONAL
quantum models specifically, or of quantum models generally.

    python scripts/20_qsvm.py

WHY THIS IS A DIFFERENT FAMILY, NOT A VARIANT
    The HQNNs in scripts/16_sweep.py are variational: a trainable circuit optimised by
    gradient descent, with T trainable gates. A quantum kernel has NO trainable
    gates at all -- the circuit is a fixed feature map, and the only thing that
    trains is the classical SVM's dual coefficients. That is precisely why it
    was excluded originally (docs/BASELINES.md section 4): Caro et al.'s sqrt(T/N)
    bound is stated for trainable gates, so it says nothing about a kernel, and
    T was the independent variable of the original design.

    The finding changed the question. "Quantum models need more data" is a claim
    about quantum models, and a kernel is one. If the QSVM also degrades at small
    N, the deficit is about quantum feature maps. If it does not, the deficit is
    specifically about optimising variational circuits from few examples -- which
    is a sharper and more useful claim.

WHY ONLY SMALL N
    A fidelity kernel needs one circuit evaluation per PAIR, so cost is O(N^2).
    Measured at 501 us/pair, including the validation kernel against support
    vectors, per seed:

        N =    571   ->  ~45 min        N = 14,286  ->  days
        N =  1,429   ->  ~2   h         N = 68,296  ->  weeks
        N =  2,857   ->  ~4   h

    The three small points are exactly where the scarcity finding lives, so the
    quadratic wall costs nothing scientifically here. It is itself worth
    reporting: quantum kernels cannot reach the data sizes at which this study's
    quantum models finally beat their classical twins.

DIMENSIONALITY
    12 features will not fit on 8 qubits, and 12 qubits costs ~24x more per
    evaluation (12 * 2^12 against 8 * 2^8), which would push N=571 alone past
    six hours. The HQNNs solve this with a TRAINED Dense(12->8) layer. A kernel
    cannot -- it has no trainable parameters by definition -- so it gets PCA to
    8 components, fitted on the training subsample only, label-free and
    deterministic.

    That asymmetry is real and must be reported: the HQNN learns its compression,
    the QSVM is handed one. It is also unavoidable rather than chosen, and it is
    the honest structural difference between the two families under an 8-qubit
    budget.

FEATURE MAP
    qml.IQPEmbedding -- the Havlicek et al. (2019, Nature 567:209) construction,
    the canonical hard-to-simulate embedding that quantum kernel proposals are
    built on. Kernel is the fidelity |<phi(x)|phi(x')>|^2, evaluated as
    encode(x) then adjoint(encode(x')), reading the probability of |0...0>.

PROTOCOL
    * C selected on a 20% holdout carved from the TRAINING subsample, mirroring
      the HQNNs' internal early-stopping holdout. The validation split is used
      only for the reported score, never for selection.
    * Scored on VALIDATION, so the numbers sit directly beside the sweep curve
      in results/sweep_results.csv, which is also validation.
    * Checkpointed per (N, seed) to results/qsvm_results.csv; re-running resumes.
"""

import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp          # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import pennylane as qml          # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.svm import SVC                           # noqa: E402

N_QUBITS = 8
# The sizes this machine can actually compute, from MEASURED run times
# (928 us per kernel pair including overhead; support vectors ~0.37 x N):
#
#     N =    571   1.0 h/seed        N = 14,286   ~2 days/seed, 10 days for 5
#     N =  1,429   2.7 h/seed        N = 68,296   ~30 days/seed, and the kernel
#     N =  2,857   5.8 h/seed                     matrix alone is 37 GB against
#                                                 17 GB of installed RAM
#
# N >= 14,286 is therefore NOT run and NOT extrapolated. An earlier version of
# this project filled those cells with a power-law extrapolation; that was
# removed so every reported number is a measurement. Those cells are reported as
# "not computable on this hardware", which is itself a finding about quantum
# kernels at scale.
SWEEP_N = [571, 1429, 2857]
SEEDS = [0, 1, 2, 3, 4]    # 5 seeds, matching every other model in the study
C_GRID = [1.0, 10.0]          # 2 values, matching every other model's budget
PAIR_BLOCK = 20000            # pairs per jitted call (state is 256 complex/pair)
RESULTS = "results/qsvm_results.csv"

FEATS = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
         "spd_mismatch_max", "crab_mean", "prev_silence_s",
         "course_resid_mean", "mismatch_slope", "straightness",
         "mismatch_mean", "crab_valid"]
ATTACKS = ["speed_lie", "jump", "gap", "dark", "drift"]

_dev = qml.device("default.qubit", wires=N_QUBITS)


# n_repeats=1, and the state vector read directly rather than qml.probs.
# Both were chosen on measurement, not preference -- benchmarked per pair:
#     reps=1 state   501 us     reps=2 probs  1073 us
#     reps=1 probs   732 us     reps=2 state  1401 us
# The first estimate for this script assumed 60 us/pair, taken from a forward
# pass rather than a kernel evaluation; an IQP embedding plus its adjoint is
# roughly four times that depth. reps=1 is still the Havlicek construction,
# one repetition instead of two.
@qml.qnode(_dev, interface="jax", diff_method=None)
def _overlap(x1, x2):
    """|<phi(x1)|phi(x2)>|^2 -- the fidelity quantum kernel.

    Encoding x1 then UN-encoding x2 returns the system to |0...0> exactly when
    the two feature-map states coincide, so the amplitude on |0...0> IS the
    overlap. No trainable parameters anywhere -- that is what separates this
    family from the variational models in scripts/16_sweep.py.
    """
    qml.IQPEmbedding(x1, wires=range(N_QUBITS), n_repeats=1)
    qml.adjoint(qml.IQPEmbedding)(x2, wires=range(N_QUBITS), n_repeats=1)
    return qml.state()


@jax.jit
def _kernel_block(A, B):
    """Fidelity for a block of paired rows: |amplitude on |0...0>|^2."""
    return jnp.abs(_overlap(A, B)[..., 0]) ** 2


def kernel_matrix(XA, XB, symmetric=False, label=""):
    """Full kernel between two sets, evaluated in fixed-size blocks.

    Blocks are padded to one constant shape: JAX recompiles for every new input
    shape, and a ragged tail cost an 8-second recompile per distinct length
    earlier in this project.
    """
    nA, nB = len(XA), len(XB)
    if symmetric:
        ii, jj = np.triu_indices(nA, k=1)          # diagonal is 1 by definition
    else:
        ii, jj = np.repeat(np.arange(nA), nB), np.tile(np.arange(nB), nA)

    out = np.empty(len(ii), dtype=float)
    t0 = time.time()
    for s in range(0, len(ii), PAIR_BLOCK):
        e = min(s + PAIR_BLOCK, len(ii))
        a = jnp.array(XA[ii[s:e]])
        b = jnp.array(XB[jj[s:e]])
        pad = PAIR_BLOCK - (e - s)
        if pad:                                     # keep one shape
            a = jnp.concatenate([a, jnp.zeros((pad, N_QUBITS))])
            b = jnp.concatenate([b, jnp.zeros((pad, N_QUBITS))])
        out[s:e] = np.asarray(_kernel_block(a, b))[: e - s]
        if label and s % (PAIR_BLOCK * 40) == 0 and s:
            done = e / len(ii)
            print(f"      {label} {done*100:5.1f}%  "
                  f"eta {(time.time()-t0)*(1-done)/done/60:5.1f} min", flush=True)

    if symmetric:
        K = np.eye(nA)
        K[ii, jj] = out
        K[jj, ii] = out
        return K
    return out.reshape(nA, nB)


def main():
    try:                                  # same process-scoped request as the sweep
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("sleep prevention: active", flush=True)
    except Exception:
        pass

    tr = pd.read_csv("data/dataset_train.csv")
    va = pd.read_csv("data/dataset_val.csv")
    X, y = tr[FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[FEATS].to_numpy(float), va.label.to_numpy(int)
    av = va.attack.to_numpy()
    rate = y.mean()

    rows, done = [], set()
    if os.path.exists(RESULTS):
        d = pd.read_csv(RESULTS)
        rows, done = d.to_dict("records"), set(zip(d.N.astype(int), d.seed.astype(int)))
    print(f"resuming with {len(done)} runs on disk\n", flush=True)

    t_start = time.time()
    for N in SWEEP_N:
        for seed in SEEDS:
            if (N, seed) in done:
                continue
            t0 = time.time()
            # identical subsampling to scripts/16_sweep.py, so the points are comparable
            rng = np.random.default_rng(1000 + seed)
            n_pos = int(round(N * rate))
            pos = rng.choice(np.where(y == 1)[0], size=n_pos, replace=False)
            neg = rng.choice(np.where(y == 0)[0], size=N - n_pos, replace=False)
            k = rng.permutation(np.concatenate([pos, neg]))
            Xs, ys = X[k], y[k]

            # PCA 12 -> 8, fitted on THIS subsample only (no labels, no val, no test)
            pca = PCA(n_components=N_QUBITS, random_state=seed).fit(Xs)
            Zs, Zv = pca.transform(Xs), pca.transform(Xv)
            lo, hi = Zs.min(0), Zs.max(0)
            span = np.where(hi - lo == 0, 1, hi - lo)
            Zs = np.clip((Zs - lo) / span, 0, 1) * np.pi
            Zv = np.clip((Zv - lo) / span, 0, 1) * np.pi

            print(f"  N={N} seed{seed}: train kernel "
                  f"({N*(N-1)//2:,} pairs) ...", flush=True)
            Ktr = kernel_matrix(Zs, Zs, symmetric=True, label="train")

            # C on a 20% holdout carved from TRAINING -- validation stays clean
            cut = max(2, int(0.2 * N))
            hi_idx, lo_idx = np.arange(cut), np.arange(cut, N)
            best_c, best_a = C_GRID[0], -1.0
            if len(np.unique(ys[lo_idx])) == 2 and len(np.unique(ys[hi_idx])) == 2:
                for C in C_GRID:
                    m = SVC(C=C, kernel="precomputed").fit(
                        Ktr[np.ix_(lo_idx, lo_idx)], ys[lo_idx])
                    a = roc_auc_score(ys[hi_idx], m.decision_function(
                        Ktr[np.ix_(hi_idx, lo_idx)]))
                    if a > best_a:
                        best_a, best_c = a, C

            svm = SVC(C=best_c, kernel="precomputed").fit(Ktr, ys)
            sv = svm.support_
            print(f"      C={best_c}  {len(sv)} support vectors of {N}  "
                  f"-> val kernel ({len(sv)*len(Zv):,} pairs)", flush=True)

            # Only support vectors carry non-zero dual coefficients, so the
            # validation kernel is computed against those rows alone.
            Kv = kernel_matrix(Zv, Zs[sv], label="val")
            svm2 = SVC(C=best_c, kernel="precomputed").fit(Ktr[np.ix_(sv, sv)], ys[sv])
            score = svm2.decision_function(Kv)

            r = {"model": "QSVM", "N": N, "seed": seed, "C": best_c,
                 "n_sv": len(sv), "secs": round(time.time() - t0, 1),
                 "val_auc": roc_auc_score(yv, score),
                 "val_ap": average_precision_score(yv, score)}
            for a in ATTACKS:
                m = (yv == 0) | (av == a)
                r[f"auc_{a}"] = roc_auc_score(yv[m], score[m])
            rows.append(r)
            pd.DataFrame(rows).to_csv(RESULTS, index=False)
            print(f"  [{(time.time()-t_start)/60:6.1f} min] N={N} seed{seed}  "
                  f"val AUC {r['val_auc']:.4f}  drift {r['auc_drift']:.3f}  "
                  f"({r['secs']/60:.1f} min)\n", flush=True)

    d = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("QSVM vs the sweep (validation AUC)")
    print("=" * 72)
    sw = pd.read_csv("results/sweep_results.csv")
    sw = sw[sw.phase == 2]
    for N in SWEEP_N:
        q = d[d.N == N]
        if not len(q):
            continue
        line = f"  N={N:>6}  QSVM {q.val_auc.mean():.4f} ± {q.val_auc.std():.3f}"
        for m in ("Q_reup", "Q_pre", "MLP_reup", "iforest", "rf"):
            s = sw[(sw.model == m) & (sw.N == N)]
            if len(s):
                line += f"   {m} {s.auc.mean():.4f}"
        print(line)
    print(f"\n[{(time.time()-t_start)/3600:.2f} h]  wrote {RESULTS}")


if __name__ == "__main__":
    main()
