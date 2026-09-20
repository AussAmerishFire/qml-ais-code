"""
25_geometric_difference.py
==========================
Could ANY quantum kernel beat classical kernels on this AIS data? Huang et al.
(2021, Nature Communications, "Power of data in quantum machine learning",
DOI 10.1038/s41467-021-22539-9) give a test that answers this from the kernel
matrices alone, before any model is trained: the geometric difference g.

    python 25_geometric_difference.py

WHAT IT MEASURES
    With both kernels normalised to Tr K = N,

        g(K_C || K_Q) = sqrt( || sqrt(K_Q) K_C^-1 sqrt(K_Q) ||_inf ),   1 <= g <= sqrt(N)

    and the model complexity of a kernel on the ACTUAL labels,

        s_K = y' (K + lam I)^-1 K (K + lam I)^-1 y        (||w||^2 of kernel ridge)

    Prediction error scales like sqrt(s_K / N), and s_C <= g^2 * s_Q for ANY
    labels. So, following the paper's flowchart:

        g small (near 1)        the classical kernel matches the quantum kernel on
                                every possible task over these inputs. Stop here.
        g near sqrt(N)          some labelling exists where the quantum kernel
                                could win. Then the real labels decide:
        s_Q << s_C and s_Q << N potential advantage on THIS task.
        s_Q close to N          the quantum kernel memorises rather than learns.

    K_C is ill-conditioned, so the regularised form is used. For any labelling the
    quantum kernel can realise, y = sqrt(K_Q) v with s_Q = ||v||^2, classical
    kernel ridge regression with K_C gives

        complexity      s_C        <= g_gen^2 * s_Q
        training error  ||resid||^2 <= g_tra^2 * s_Q

        g_gen = sqrt( || sqrt(K_Q) sqrt(K_C) (K_C + lam I)^-2 sqrt(K_C) sqrt(K_Q) ||_inf )
        g_tra = lam * sqrt( || sqrt(K_Q) (K_C + lam I)^-2 sqrt(K_Q) ||_inf )

    BOTH terms matter. A low-rank or over-smooth classical kernel gets a small
    g_gen only because it cannot fit anything -- its g_tra is then large. So a
    classical kernel counts as "matching" the quantum kernel only if
    g_tra <= FIT_TOL (0.1: training error within 1% of the quantum complexity
    budget), and the reported g is the minimum g_gen over kernels that pass.
    Reported at four values of lam.

INPUTS -- identical to 20_qsvm.py
    Same subsample per seed (rng 1000 + seed, anomaly rate held fixed), same PCA
    12 -> 8 fitted on that subsample only, same [0, pi] scaling, same IQP
    fidelity kernel. Classical kernels receive the SAME 8 numbers, so this compares
    feature maps, not amounts of information.

CLASSICAL KERNELS
    Gaussian (RBF) at 7 bandwidths spanning 1000x around the median heuristic, and
    linear. The question is whether ANY classical kernel competes, so the reported
    g is the MINIMUM over them -- the conservative choice for an advantage claim.
    s_C is likewise the minimum over classical kernels.

OUTPUT
    geometric_difference.csv   one row per (N, seed, classical kernel, lam)
    kernels/KQ_N{N}_s{seed}.npy  the quantum kernels (regenerable; gitignored)
    Resumes per (N, seed).
"""

import importlib
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

q = importlib.import_module("20_qsvm")     # kernel_matrix, FEATS, N_QUBITS

SIZES = [571, 1429]
SEEDS = [0, 1, 2, 3, 4]
LAMS = [0.001, 0.01, 0.1, 1.0]
FIT_TOL = 0.1
GAMMA_FACTORS = [0.03, 0.1, 0.3, 1, 3, 10, 30]
OUT = "geometric_difference.csv"
KDIR = "kernels"


def keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("sleep prevention: active", flush=True)
    except Exception:
        pass


def qsvm_inputs(X, y, N, seed, rate):
    """Exactly the subsample, PCA and scaling of 20_qsvm.py."""
    rng = np.random.default_rng(1000 + seed)
    n_pos = int(round(N * rate))
    pos = rng.choice(np.where(y == 1)[0], size=n_pos, replace=False)
    neg = rng.choice(np.where(y == 0)[0], size=N - n_pos, replace=False)
    k = rng.permutation(np.concatenate([pos, neg]))
    Xs, ys = X[k], y[k]
    Zs = PCA(n_components=q.N_QUBITS, random_state=seed).fit_transform(Xs)
    lo, hi = Zs.min(0), Zs.max(0)
    span = np.where(hi - lo == 0, 1, hi - lo)
    return np.clip((Zs - lo) / span, 0, 1) * np.pi, ys


def eig(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    return np.clip(w, 0, None), V


def trace_norm(K):
    return K * (len(K) / np.trace(K))


def classical_kernels(Z):
    sq = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    g0 = 1.0 / np.median(sq[np.triu_indices(len(Z), 1)])
    ks = {f"rbf x{f:g}": np.exp(-g0 * f * sq) for f in GAMMA_FACTORS}
    ks["linear"] = trace_norm(Z @ Z.T)
    return ks


def _top_eig(sqrtQ, VC, diag):
    M = sqrtQ @ ((VC * diag) @ VC.T) @ sqrtQ
    return float(max(np.linalg.eigvalsh((M + M.T) / 2).max(), 0))


def geometric_difference(wC, VC, sqrtQ, lam):
    """(g_gen, g_tra) -- see module docstring."""
    g_gen = np.sqrt(_top_eig(sqrtQ, VC, wC / (wC + lam) ** 2))
    g_tra = lam * np.sqrt(_top_eig(sqrtQ, VC, 1.0 / (wC + lam) ** 2))
    return float(g_gen), float(g_tra)


def complexity(w, V, y, lam):
    """(s_K, mean training residual) of kernel ridge on the real labels."""
    u = V.T @ y
    return (float(np.sum(u ** 2 * w / (w + lam) ** 2)),
            float(np.sum(u ** 2 * lam ** 2 / (w + lam) ** 2)) / len(y))


def main():
    keep_awake()
    os.makedirs(KDIR, exist_ok=True)
    tr = pd.read_csv("dataset_train.csv")
    X, y = tr[q.FEATS].to_numpy(float), tr.label.to_numpy(int)
    rate = y.mean()

    rows, done = [], set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        rows, done = d.to_dict("records"), set(zip(d.N.astype(int), d.seed.astype(int)))
    print(f"resuming with {len(done)} (N, seed) cells on disk\n", flush=True)

    t_all = time.time()
    for N in SIZES:
        for seed in SEEDS:
            if (N, seed) in done:
                continue
            t0 = time.time()
            Z, ys = qsvm_inputs(X, y, N, seed, rate)
            kf = f"{KDIR}/KQ_N{N}_s{seed}.npy"
            if os.path.exists(kf):
                KQ = np.load(kf).astype(float)
            else:
                print(f"  N={N} seed{seed}: quantum kernel ({N*(N-1)//2:,} pairs)", flush=True)
                KQ = q.kernel_matrix(Z, Z, symmetric=True, label="K_Q")
                np.save(kf, KQ.astype(np.float32))
            KQ = trace_norm(KQ)
            wQ, VQ = eig(KQ)
            sqrtQ = (VQ * np.sqrt(wQ)) @ VQ.T
            yy = np.where(ys == 1, 1.0, -1.0)
            off = KQ[np.triu_indices(N, 1)]
            pr_Q = wQ.sum() ** 2 / (wQ ** 2).sum()      # effective rank

            for name, KC in classical_kernels(Z).items():
                wC, VC = eig(KC)
                for lam in LAMS:
                    g_gen, g_tra = geometric_difference(wC, VC, sqrtQ, lam)
                    s_C, r_C = complexity(wC, VC, yy, lam)
                    s_Q, r_Q = complexity(wQ, VQ, yy, lam)
                    rows.append({
                        "N": N, "seed": seed, "kernel": name, "lam": lam,
                        "g_gen": g_gen, "g_tra": g_tra, "fits": g_tra <= FIT_TOL,
                        "sqrtN": np.sqrt(N),
                        "s_C": s_C, "train_err_C": r_C,
                        "s_Q": s_Q, "train_err_Q": r_Q,
                        "effrank_C": wC.sum() ** 2 / (wC ** 2).sum(),
                        "effrank_Q": pr_Q,
                        "KQ_offdiag_mean": off.mean(), "KQ_offdiag_std": off.std(),
                        "n_pos": int(ys.sum()),
                    })
            pd.DataFrame(rows).to_csv(OUT, index=False)
            d = pd.DataFrame([r for r in rows if r["N"] == N and r["seed"] == seed])
            t = d[(d.lam == 0.01) & d.fits]
            msg = (f"min g_gen {t.g_gen.min():.2f} via {t.sort_values('g_gen').kernel.iloc[0]}"
                   if len(t) else "no classical kernel fits at lam .01")
            print(f"  [{(time.time()-t_all)/60:5.1f} min] N={N} seed{seed}  {msg}  "
                  f"(sqrt N {np.sqrt(N):.1f})  K_Q off-diag {off.mean():.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("Geometric difference, minimum over classical kernels (median over seeds)")
    print("=" * 78)
    for N in SIZES:
        s = d[d.N == N]
        for lam in LAMS:
            t = s[(s.lam == lam) & s.fits]
            if not len(t):
                print(f"  N={N:>5}  lam={lam:<6} no classical kernel fits (g_tra > {FIT_TOL})")
                continue
            gmin = t.groupby("seed").g_gen.min()
            a = s[s.lam == lam]
            print(f"  N={N:>5}  lam={lam:<6} g={gmin.median():6.2f} "
                  f"[{gmin.min():.2f}-{gmin.max():.2f}] of max {np.sqrt(N):.1f}  "
                  f"({len(gmin)}/{a.seed.nunique()} seeds)   "
                  f"s_Q={a.groupby('seed').s_Q.first().median():8.1f}  "
                  f"min s_C={t.groupby('seed').s_C.min().median():8.1f}")
    print(f"\n[{(time.time()-t_all)/60:.1f} min]  wrote {OUT}")


if __name__ == "__main__":
    main()
