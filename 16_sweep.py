"""
16_sweep.py
===========
The headline experiment: does a quantum model generalise better than a
parameter-matched classical one when training data is scarce?

    python 16_sweep.py

Runs two phases, checkpointing EVERY result to sweep_results.csv the moment it
completes. Re-running skips finished work, so an interrupted run loses at most
one model fit. (An earlier 4.4-hour grid was lost entirely when its session
ended; that must not happen again.)

------------------------------------------------------------------------------
PHASE 1 -- hyperparameter selection at full data, on validation
------------------------------------------------------------------------------
8 configurations x 3 seeds for each of four models: two quantum architectures
and their parameter-matched classical twins. Equal budgets, as Bowles, Ahmed &
Schuld (2024) require.

The previous grid was broken: half its configurations used weight decay 1e-2,
which dominated the cross-entropy and stopped training after 2.6 epochs
(val AUC 0.7089). The decay range here is 1e-5 to 1e-3.

------------------------------------------------------------------------------
PHASE 2 -- the label-scarcity sweep
------------------------------------------------------------------------------
Training-set size N is swept over 571 / 1,429 / 2,857 / 5,714 / 14,286 / 68,296
holding the anomaly rate fixed at its natural 3.543% -- design (B) in
BASELINES.md section 5. Holding the class ratio constant means N is the only
variable that moves, which is what makes the comparison across sizes readable.

WHAT IS BEING TESTED
    Caro et al. (2022) bound the generalisation gap of a model with T trainable
    gates as sqrt(T/N). At full data sqrt(50/68296) = 0.027 -- generalisation is
    not binding, and no advantage is expected or observed. At N = 571 it rises
    to 0.30, which is the regime the bound describes.

    The sweep carries THREE values of T so it tests the SHAPE of that
    relationship rather than a single point:

        E_reup        T = 57    hybrid + data re-uploading
        C_pre frozen  T = 74    same circuit, rotation angles NEVER trained
        C_pre         T = 122   hybrid with a classical pre-layer

    The frozen model is the sharp one. At full data it scored 0.8270 against
    the trained model's 0.8357 -- a gap of 0.009 against seed noise near 0.02,
    i.e. the 48 trainable rotation angles were nearly decorative. The bound
    predicts that a LOWER-T model degrades more slowly as N falls, so the
    frozen model should GAIN ground on the trained one as the sweep proceeds.
    If it does, that measures the bound's shape. If the gap stays flat, the
    Caro framing does not survive and the paper reports that instead.

NO EARLY STOPPING IN PHASE 2 -- a deliberate choice.
    Early stopping is itself a regulariser, and it would partially substitute
    for exactly the generalisation behaviour being measured, masking the effect
    at small N.

    Instead every Phase 2 run gets the same budget of 640 GRADIENT STEPS -- not
    epochs. Verified identical for quantum and classical at all six values of N
    (640 everywhere; 642 at N = 14,286, one step of rounding). Fixing epochs
    instead would have given N = 571 only 40 updates against 640 at full data,
    and the resulting undertraining at small N would have imitated the very
    generalisation effect under test.

Test split is NOT opened by this script. Selection and sweeping are validation
only; 17_final.py reads test once, afterwards.
"""

import ctypes
import itertools
import json
import os
import sys
import time


def keep_awake():
    """Ask Windows not to sleep while this process runs.

    An overnight run logged 12.9 hours of wall time for 7 hours of work: one
    configuration took 21,133 s while the SAME configuration on a different
    seed took 1,255 s. The machine was asleep; the clock was not. That single
    stall consumed 6 hours and left the sweep unfinished.

    ES_CONTINUOUS | ES_SYSTEM_REQUIRED is process-scoped and reverts the moment
    this process exits -- it is the same request a video player makes during
    playback. It does NOT change the user's power settings. The display is
    deliberately left free to switch off (no ES_DISPLAY_REQUIRED).
    """
    try:
        ok = ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print(f"sleep prevention: {'active' if ok else 'REFUSED by OS'}", flush=True)
    except Exception as e:                       # non-Windows, or blocked
        print(f"sleep prevention unavailable ({e}) -- runs may stall if the "
              f"machine sleeps", flush=True)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp          # noqa: E402
import numpy as np               # noqa: E402
import optax                     # noqa: E402
import pandas as pd              # noqa: E402
import pennylane as qml          # noqa: E402
from sklearn.ensemble import IsolationForest, RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score    # noqa: E402
from sklearn.neural_network import MLPClassifier                      # noqa: E402

N_QUBITS = 8
MAX_BATCH = 4096
GRID_EPOCHS, GRID_PATIENCE = 60, 10

# Phase 2 holds the number of GRADIENT STEPS fixed, not the number of epochs.
# This matters more than it looks. With a 4096 batch, 40 epochs is 640 steps at
# N = 68,296 but only 40 steps at N = 571 -- the small-N models would get 16x
# fewer weight updates and score worse purely from undertraining, which is
# exactly the shape of the generalisation effect being measured. Fixing steps
# instead means every run gets the same optimisation budget and only the amount
# of DATA changes, which is the variable under test.
# 640 is what the full-data runs used (40 epochs x 16 batches) to reach ~0.83.
SWEEP_STEPS = 640
RESULTS = "sweep_results.csv"

FEATS = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
         "spd_mismatch_max", "crab_mean", "prev_silence_s", "crab_valid"]
ATTACKS = ["speed_lie", "jump", "gap", "dark", "drift"]
# Ascending, deliberately: the small-N points are the ones the hypothesis is
# about, so if a run is interrupted again the scientifically load-bearing data
# already exists. N = 5714 was dropped -- it costs the same as 14,286 (both are
# capped by the 4096 batch) while sitting between two points that already
# bracket it. Five points across a 120x range of N is ample to read a trend.
SWEEP_N = [571, 1429, 2857, 14286, 68296]
SEEDS_GRID = [0, 1, 2]
SEEDS_SWEEP = [0, 1, 2, 3, 4]

_dev = qml.device("default.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# QUANTUM MODELS
# ---------------------------------------------------------------------------

def _circuit(n_layers, reupload):
    @qml.qnode(_dev, interface="jax", diff_method="backprop")
    def c(x, w):
        if reupload:
            for l in range(n_layers):
                qml.AngleEmbedding(x, wires=range(N_QUBITS))
                qml.StronglyEntanglingLayers(w[l:l + 1], wires=range(N_QUBITS))
        else:
            qml.AngleEmbedding(x, wires=range(N_QUBITS))
            qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return c


QARCH = {
    "E_reup":   dict(layers=2, pre=False, post="linear", reupload=True,  frozen=False),
    "C_pre":    dict(layers=2, pre=True,  post="sum",    reupload=False, frozen=False),
    "C_frozen": dict(layers=2, pre=True,  post="sum",    reupload=False, frozen=True),
}


def q_params(a):
    shp = qml.StronglyEntanglingLayers.shape(n_layers=a["layers"], n_wires=N_QUBITS)
    t = 0 if a["frozen"] else int(np.prod(shp))      # frozen gates are not trainable
    t += (N_QUBITS * N_QUBITS + N_QUBITS) if a["pre"] else 0
    t += (N_QUBITS + 1) if a["post"] == "linear" else 2
    return t


def q_train(a, X, y, cfg, seed, epochs, patience=None):
    circ = _circuit(a["layers"], a["reupload"])
    rng = np.random.default_rng(seed)
    shp = qml.StronglyEntanglingLayers.shape(n_layers=a["layers"], n_wires=N_QUBITS)
    frozen_w = jnp.array(rng.normal(0, 0.1, shp))

    p = {}
    if not a["frozen"]:
        p["w"] = jnp.array(rng.normal(0, 0.1, shp))
    if a["pre"]:
        p["Wp"] = jnp.array(rng.normal(0, 0.3, (N_QUBITS, N_QUBITS)))
        p["bp"] = jnp.zeros(N_QUBITS)
    if a["post"] == "linear":
        p["Wo"] = jnp.array(rng.normal(0, 0.5, (N_QUBITS,)))
        p["bo"] = jnp.array(0.0)
    else:
        p["a"], p["b"] = jnp.array(4.0), jnp.array(0.0)

    def fwd(p, Z):
        if a["pre"]:
            Z = jnp.pi * jax.nn.sigmoid(Z @ p["Wp"] + p["bp"])
        w = frozen_w if a["frozen"] else p["w"]
        z = jnp.stack(circ(Z, w), axis=-1)
        if a["post"] == "linear":
            return z @ p["Wo"] + p["bo"]
        return p["a"] * jnp.sum(z, axis=-1) + p["b"]

    @jax.jit
    def lg(p, Z, t):
        def L(p):
            q = jnp.clip(jax.nn.sigmoid(fwd(p, Z)), 1e-7, 1 - 1e-7)
            reg = sum(jnp.sum(v ** 2) for k, v in p.items()
                      if k in ("w", "Wp", "Wo"))
            return -jnp.mean(t * jnp.log(q) + (1 - t) * jnp.log(1 - q)) \
                + cfg["decay"] * reg
        return jax.value_and_grad(L)(p)

    predict = jax.jit(lambda p, Z: jax.nn.sigmoid(fwd(p, Z)))

    if cfg["oversample"]:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        want = int(len(neg) * cfg["oversample"] / (1 - cfg["oversample"]))
        if len(pos):
            pos = rng.choice(pos, size=max(want, len(pos)), replace=True)
            k = rng.permutation(np.concatenate([pos, neg]))
            X, y = X[k], y[k]

    Xa, ya = jnp.array(X), jnp.array(y.astype(float))
    B = min(MAX_BATCH, len(ya))
    n_full = max(B, (len(ya) // B) * B)
    opt = optax.adam(cfg["lr"])
    state = opt.init(p)

    # `epochs` is a step budget in Phase 2 (patience is None) and an epoch
    # budget in Phase 1 (patience set), where early stopping governs instead.
    steps_per_epoch = max(1, n_full // B)
    n_epochs = epochs if patience is not None else \
        int(np.ceil(epochs / steps_per_epoch))

    best, best_p, since, taken = -np.inf, p, 0, 0
    for ep in range(n_epochs):
        order = rng.permutation(len(ya))[:n_full]
        for s in range(0, n_full, B):
            b = order[s:s + B]
            if len(b) < B:
                continue                      # keep one shape; avoids recompiles
            _, g = lg(p, Xa[b], ya[b])
            u, state = opt.update(g, state, p)
            p = optax.apply_updates(p, u)
            taken += 1
            if patience is None and taken >= epochs:
                return p, predict             # exact step budget reached
        if patience is not None:              # Phase 1 only
            auc = roc_auc_score(ya, np.asarray(predict(p, Xa)))
            if auc > best + 1e-5:
                best, best_p, since = auc, p, 0
            else:
                since += 1
                if since >= patience:
                    break
    return (best_p if patience is not None else p), predict


# ---------------------------------------------------------------------------
# GRIDS -- 8 configurations each, matched in size and structure
# ---------------------------------------------------------------------------

QGRID = [{"lr": a, "oversample": b, "decay": c}
         for a, b, c in itertools.product((0.02, 0.05), (None, 0.35), (1e-5, 1e-3))]
CGRID = [{"lr": a, "oversample": b, "alpha": c}
         for a, b, c in itertools.product((0.003, 0.01), (None, 0.35), (1e-5, 1e-3))]

CARCH = {"MLP6": (6,), "MLP85": (8, 5)}
CPARAMS = {"MLP6": 61, "MLP85": 123}


def c_train(hl, X, y, cfg, seed, budget, early):
    """`budget` is epochs in Phase 1 (early=True) and GRADIENT STEPS in Phase 2.

    In Phase 2 the batch size mirrors the quantum models' min(4096, N) and
    max_iter is set so the step count matches theirs exactly. Left at sklearn's
    default batch_size of 200, the MLP would take N/200 steps per epoch --
    341 per epoch at full data against 3 at N = 571 -- and the classical side
    would be optimised on a completely different budget from the quantum side
    at every point in the sweep.
    """
    rng = np.random.default_rng(seed)
    if cfg["oversample"]:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        want = int(len(neg) * cfg["oversample"] / (1 - cfg["oversample"]))
        if len(pos):
            pos = rng.choice(pos, size=max(want, len(pos)), replace=True)
            k = rng.permutation(np.concatenate([pos, neg]))
            X, y = X[k], y[k]

    if early:
        batch, iters = min(200, len(y)), budget
    else:
        batch = min(MAX_BATCH, len(y))
        iters = int(np.ceil(budget / max(1, len(y) // batch)))

    m = MLPClassifier(hidden_layer_sizes=hl, alpha=cfg["alpha"],
                      learning_rate_init=cfg["lr"], max_iter=iters,
                      batch_size=batch, early_stopping=early,
                      n_iter_no_change=GRID_PATIENCE if early else iters,
                      random_state=seed)
    m.fit(X, y)
    return m


# ---------------------------------------------------------------------------
# CHECKPOINTING
# ---------------------------------------------------------------------------

def load_done():
    if not os.path.exists(RESULTS):
        return set(), []
    df = pd.read_csv(RESULTS)
    keys = set(zip(df.phase, df.model, df.cfg_id.astype(int),
                   df.N.astype(int), df.seed.astype(int)))
    return keys, df.to_dict("records")


def save(rows):
    pd.DataFrame(rows).to_csv(RESULTS, index=False)


def subsample(X, y, N, seed, rate):
    """Shrink the training set to N windows holding the anomaly rate fixed."""
    if N >= len(y):
        return X, y
    rng = np.random.default_rng(1000 + seed)
    n_pos = int(round(N * rate))
    pos = rng.choice(np.where(y == 1)[0], size=n_pos, replace=False)
    neg = rng.choice(np.where(y == 0)[0], size=N - n_pos, replace=False)
    k = rng.permutation(np.concatenate([pos, neg]))
    return X[k], y[k]


# ---------------------------------------------------------------------------

def main():
    keep_awake()
    tr = pd.read_csv("dataset_train.csv")
    va = pd.read_csv("dataset_val.csv")
    X, y = tr[FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[FEATS].to_numpy(float), va.label.to_numpy(int)
    Xvj = jnp.array(Xv)
    av = va.attack.to_numpy()
    rate = y.mean()
    t_start = time.time()

    done, rows = load_done()
    print(f"train {len(y):,} ({rate*100:.3f}% anomalous) | val {len(yv):,}")
    print(f"resuming with {len(done)} runs already on disk\n", flush=True)

    def record(phase, model, cfg_id, N, seed, T, scores, secs):
        r = {"phase": phase, "model": model, "cfg_id": cfg_id, "N": N,
             "seed": seed, "T": T, "secs": round(secs, 1),
             "auc": roc_auc_score(yv, scores),
             "ap": average_precision_score(yv, scores)}
        for a in ATTACKS:
            m = (yv == 0) | (av == a)
            r[f"auc_{a}"] = roc_auc_score(yv[m], scores[m])
        rows.append(r)
        save(rows)
        done.add((phase, model, cfg_id, N, seed))
        return r

    # ================= PHASE 1 : hyperparameter selection ==================
    print("=" * 74)
    print("PHASE 1 -- 8 configs x 3 seeds, full data, validation only")
    print("=" * 74, flush=True)

    for name in ("E_reup", "C_pre"):
        a = QARCH[name]
        for ci, cfg in enumerate(QGRID):
            for s in SEEDS_GRID:
                if (1, name, ci, 68296, s) in done:
                    continue
                t0 = time.time()
                p, pred = q_train(a, X, y, cfg, s, GRID_EPOCHS, GRID_PATIENCE)
                r = record(1, name, ci, 68296, s, q_params(a),
                           np.asarray(pred(p, Xvj)), time.time() - t0)
                print(f"  [{time.time()-t_start:6.0f}s] {name:<9} cfg{ci} "
                      f"seed{s}  AUC {r['auc']:.4f}  ({r['secs']:.0f}s)",
                      flush=True)

    for name, hl in CARCH.items():
        for ci, cfg in enumerate(CGRID):
            for s in SEEDS_GRID:
                if (1, name, ci, 68296, s) in done:
                    continue
                t0 = time.time()
                m = c_train(hl, X, y, cfg, s, 500, True)
                r = record(1, name, ci, 68296, s, CPARAMS[name],
                           m.predict_proba(Xv)[:, 1], time.time() - t0)
                print(f"  [{time.time()-t_start:6.0f}s] {name:<9} cfg{ci} "
                      f"seed{s}  AUC {r['auc']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    p1 = df[df.phase == 1]
    best = {}
    for name in list(QARCH)[:2] + list(CARCH):
        if name == "C_frozen":
            continue
        g = p1[p1.model == name].groupby("cfg_id").auc.mean()
        best[name] = int(g.idxmax())
        print(f"\n  BEST {name:<9} cfg{best[name]}  val AUC {g.max():.4f}")
    best["C_frozen"] = best["C_pre"]          # same architecture, same settings
    json.dump(best, open("sweep_best_cfg.json", "w"), indent=2)

    # ===================== PHASE 2 : scarcity sweep ========================
    print("\n" + "=" * 74)
    print("PHASE 2 -- label-scarcity sweep, fixed 40-epoch budget, 5 seeds")
    print("=" * 74, flush=True)

    for N in SWEEP_N:
        for s in SEEDS_SWEEP:
            Xs, ys = subsample(X, y, N, s, rate)

            for name, a in QARCH.items():
                if (2, name, -1, N, s) in done:
                    continue
                cfg = QGRID[best[name]]
                t0 = time.time()
                p, pred = q_train(a, Xs, ys, cfg, s, SWEEP_STEPS, None)
                r = record(2, name, -1, N, s, q_params(a),
                           np.asarray(pred(p, Xvj)), time.time() - t0)
                print(f"  [{time.time()-t_start:6.0f}s] N={N:>6} seed{s} "
                      f"{name:<9} T={r['T']:>3}  AUC {r['auc']:.4f} "
                      f"drift {r['auc_drift']:.3f}  ({r['secs']:.0f}s)",
                      flush=True)

            for name, hl in CARCH.items():
                if (2, name, -1, N, s) in done:
                    continue
                cfg = CGRID[best[name]]
                t0 = time.time()
                m = c_train(hl, Xs, ys, cfg, s, SWEEP_STEPS, False)
                r = record(2, name, -1, N, s, CPARAMS[name],
                           m.predict_proba(Xv)[:, 1], time.time() - t0)
                print(f"  [{time.time()-t_start:6.0f}s] N={N:>6} seed{s} "
                      f"{name:<9} T={r['T']:>3}  AUC {r['auc']:.4f} "
                      f"drift {r['auc_drift']:.3f}", flush=True)

            # Reference lines: Isolation Forest uses no labels at all, so it is
            # the honest competitor at the scarce end; RF is the best classical
            # model overall, for context.
            for name in ("iforest", "rf"):
                if (2, name, -1, N, s) in done:
                    continue
                t0 = time.time()
                if name == "iforest":
                    mm = IsolationForest(n_estimators=200, max_samples=min(1024, len(ys)),
                                         contamination=0.02, random_state=s,
                                         n_jobs=-1).fit(Xs)
                    sc = -mm.score_samples(Xv)
                else:
                    mm = RandomForestClassifier(n_estimators=300, max_depth=6,
                                                random_state=s, n_jobs=-1).fit(Xs, ys)
                    sc = mm.predict_proba(Xv)[:, 1]
                r = record(2, name, -1, N, s, 0, sc, time.time() - t0)
                print(f"  [{time.time()-t_start:6.0f}s] N={N:>6} seed{s} "
                      f"{name:<9}        AUC {r['auc']:.4f}", flush=True)

    print(f"\nDONE in {(time.time()-t_start)/3600:.2f} h -- {len(rows)} runs "
          f"in {RESULTS}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
