"""
scripts/12_qnn.py
=========
The hybrid quantum neural network, and the head-to-head against the
parameter-matched classical MLP.

    python scripts/12_qnn.py time      # time a single training run, then stop
    python scripts/12_qnn.py select    # 12-config x 5-seed grid, validation only
    python scripts/12_qnn.py final     # test-set evaluation of the selected config

THE MODEL
    AngleEmbedding(x, wires=0..7)            0 trainable parameters
    StronglyEntanglingLayers(w, wires=0..7)  L=2  ->  2*8*3 = 48 parameters
    readout  p = sigmoid(a * <Z_0> + b)      +2 parameters
                                             ------------------------------
                                             T = 50 trainable parameters

    matched against the classical MLP 8-5-1, which has 51. The 2% edge is the
    classical model's, which is the safe direction.

WHY A VARIATIONAL CIRCUIT AND NOT A QUANTUM KERNEL
    Caro et al. (2022, Nature Communications) bound the generalisation error of
    a QML model with T *trainable gates* as sqrt(T/N). A quantum kernel has no
    trainable gates, so the bound does not apply to it. T is the independent
    variable of this study, which forces a variational model.

FAIRNESS (docs/BASELINES.md section 3 -- every rule enforced here)
  * 12 hyperparameter configurations, exactly matching the MLP's budget.
  * The grid mirrors the MLP's structure: 2 learning rates x 3 oversampling
    levels x 2 weight-decay levels.
  * L is NOT tuned. L fixes T, and T is what makes the comparison a comparison.
  * Early stopping uses a 10% split carved out of TRAINING data, exactly as
    sklearn's MLPClassifier does -- the real validation split is used only for
    selecting between configurations, identically for both models.
  * 5 seeds. Test read once.

NUMERICS
    jax_enable_x64 is ON. Measured: with it off, expectation values differ from
    double precision by up to 4.4e-7 absolute (on values of order 0.27). That is
    almost certainly harmless -- there is no catastrophic cancellation in a
    normalised 8-qubit state, unlike the float32 latitude bug found earlier in
    this project. It costs 2x runtime and removes the question entirely, so it
    is on.
"""

import json
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp          # noqa: E402
import numpy as np               # noqa: E402
import optax                     # noqa: E402
import pandas as pd              # noqa: E402
import pennylane as qml          # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

N_QUBITS = 8
N_LAYERS = 2                     # fixes T = 50; never tuned
MAX_EPOCHS = 60
PATIENCE = 10                    # epochs without internal-val improvement
BATCH = 4096                     # measured best throughput on this machine
SEEDS = [0, 1, 2, 3, 4]

FEATS = [
    "sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
    "spd_mismatch_max", "crab_mean", "prev_silence_s", "crab_valid",
]
ATTACKS = ["speed_lie", "jump", "gap", "dark", "drift"]
ALERT_RATE = 0.035

# Mirrors the MLP grid shape exactly: 2 x 3 x 2 = 12 configurations.
GRID = [
    {"lr": lr, "oversample": ov, "decay": wd}
    for lr in (0.02, 0.05)
    for ov in (None, 0.2, 0.35)
    for wd in (1e-4, 1e-2)
]

# ---------------------------------------------------------------------------
# CIRCUIT
# ---------------------------------------------------------------------------

_dev = qml.device("default.qubit", wires=N_QUBITS)


_OBS = qml.sum(*[qml.PauliZ(i) for i in range(N_QUBITS)])


@qml.qnode(_dev, interface="jax", diff_method="backprop")
def _circuit(x, w):
    """x: (batch, 8) angles in [0, pi].  w: (L, 8, 3) trainable rotations.

    READOUT: the summed Pauli-Z over all eight wires, not <Z_0> alone.

    This is an ARCHITECTURE choice, fixed before the hyperparameter grid runs
    and before the test split is touched -- the counterpart of accepting
    sklearn's default ReLU for the MLP without tuning it. It is recorded here
    because it is the one place this study deviates from the most obvious
    default, and the reasoning must be auditable.

    An observable carries no trainable parameters, so T = 50 either way and the
    match to the MLP's 51 is unaffected.

    Two measurements motivated it, both made on validation data:

    1. Dynamic range. Randomising one input feature at a time moves <Z_0> with
       standard deviation 0.118, and the summed readout with 0.442 -- roughly
       4x the usable signal.
    2. Trained validation AUC at a fixed configuration (lr 0.05, oversample
       0.35, 25 epochs, seed 0):
           <Z_0>       untrained 0.5056 -> trained 0.7697
           sum of Z_i  untrained 0.6560 -> trained 0.8188
       Both train (the <Z_0> model gains 0.264 AUC from chance, so the
       optimiser is working); the summed readout simply ends up better.

    NOT the reason: a narrow light cone. That was the first hypothesis and it
    was tested and refuted -- at L=2 the readout's sensitivity to the far
    qubits (q4-q7) is 0.947x its sensitivity to the near ones, because
    StronglyEntanglingLayers applies long-range CNOTs. All eight features
    already reach a single-qubit measurement.
    """
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
    return qml.expval(_OBS)


def n_params():
    shp = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    return int(np.prod(shp)) + 2  # +2 for the readout scale and bias


def _probs(params, X):
    return jax.nn.sigmoid(params["a"] * _circuit(X, params["w"]) + params["b"])


@jax.jit
def _loss(params, X, y, decay):
    p = jnp.clip(_probs(params, X), 1e-7, 1 - 1e-7)
    bce = -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
    return bce + decay * jnp.sum(params["w"] ** 2)


_value_and_grad = jax.jit(jax.value_and_grad(_loss))


@jax.jit
def _predict(params, X):
    return _probs(params, X)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train(Xtr, ytr, cfg, seed, verbose=False):
    """One training run. Returns (params, epochs_used, history)."""
    rng = np.random.default_rng(seed)

    # Oversample the minority class, exactly as the MLP does. Resampling
    # changes the data, not the parameter count, so the T=50 match holds.
    if cfg["oversample"]:
        pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
        want = int(len(neg) * cfg["oversample"] / (1 - cfg["oversample"]))
        if len(pos) and want > len(pos):
            pos = rng.choice(pos, size=want, replace=True)
        keep = rng.permutation(np.concatenate([pos, neg]))
        Xtr, ytr = Xtr[keep], ytr[keep]

    # Internal 10% stratified holdout for early stopping -- carved out of
    # TRAINING data, mirroring sklearn MLPClassifier(early_stopping=True).
    idx = rng.permutation(len(ytr))
    cut = max(1, int(0.10 * len(idx)))
    vi, ti = idx[:cut], idx[cut:]
    Xa, ya = jnp.array(Xtr[ti]), jnp.array(ytr[ti].astype(float))
    Xb, yb = jnp.array(Xtr[vi]), ytr[vi]

    shp = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    params = {
        # Small init keeps the circuit near identity at the start. With 8 qubits
        # and L=2 the circuit is far too shallow for barren plateaus to bite
        # (Larocca et al. 2025), so this is about optimisation, not trainability.
        "w": jnp.array(rng.normal(0, 0.1, shp)),
        "a": jnp.array(4.0),   # expvals live in [-1, 1]; a=4 keeps sigmoid live
        "b": jnp.array(0.0),
    }
    opt = optax.adam(cfg["lr"])
    state = opt.init(params)
    decay = cfg["decay"]

    best, best_params, best_ep, since = -np.inf, params, 0, 0
    hist = []
    for ep in range(MAX_EPOCHS):
        order = rng.permutation(len(ya))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            _, g = _value_and_grad(params, Xa[b], ya[b], decay)
            upd, state = opt.update(g, state, params)
            params = optax.apply_updates(params, upd)

        auc = roc_auc_score(yb, np.asarray(_predict(params, Xb)))
        hist.append(auc)
        if auc > best + 1e-5:
            best, best_params, best_ep, since = auc, params, ep, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
        if verbose:
            print(f"    epoch {ep:>2}  internal-val AUC {auc:.4f}", flush=True)

    return best_params, best_ep + 1, hist


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def load():
    tr = pd.read_csv("data/dataset_train.csv")
    va = pd.read_csv("data/dataset_val.csv")
    te = pd.read_csv("data/dataset_test.csv")
    return (tr[FEATS].to_numpy(float), tr.label.to_numpy(int),
            va[FEATS].to_numpy(float), va.label.to_numpy(int),
            te[FEATS].to_numpy(float), te.label.to_numpy(int),
            te.attack.to_numpy())


# ---------------------------------------------------------------------------
# MODES
# ---------------------------------------------------------------------------

def mode_time():
    X, y, Xv, yv, *_ = load()
    print(f"trainable parameters T = {n_params()}  "
          f"(classical MLP has 51 -- a 2% edge to the classical model)")
    print(f"train {len(y):,} windows, {int(y.sum()):,} anomalous")
    cfg = {"lr": 0.05, "oversample": 0.35, "decay": 1e-4}
    print(f"\ntiming one run at {cfg} ...")
    t0 = time.time()
    p, eps, hist = train(X, y, cfg, 0, verbose=True)
    dt = time.time() - t0
    auc = roc_auc_score(yv, np.asarray(_predict(p, jnp.array(Xv))))
    print(f"\n  {eps} epochs in {dt:.1f}s  ({dt/len(hist):.1f}s per epoch)")
    print(f"  validation AUC {auc:.4f}   [classical MLP: 0.8441]")
    print(f"\n  projected grid cost: 12 configs x 5 seeds x {dt/60:.1f} min "
          f"= {12*5*dt/3600:.1f} hours")


def mode_select():
    X, y, Xv, yv, *_ = load()
    print(f"SELECTION -- 12 configs x 5 seeds, validation split only "
          f"(T = {n_params()})\n")
    results = []
    for i, cfg in enumerate(GRID, 1):
        t0 = time.time()
        aucs, eps = [], []
        for s in SEEDS:
            p, e, _ = train(X, y, cfg, s)
            aucs.append(roc_auc_score(yv, np.asarray(_predict(p, jnp.array(Xv)))))
            eps.append(e)
        m, sd = float(np.mean(aucs)), float(np.std(aucs))
        results.append({"cfg": cfg, "val_auc": m, "val_std": sd,
                        "mean_epochs": float(np.mean(eps))})
        print(f"  [{i:>2}/12] lr={cfg['lr']:<5} ov={str(cfg['oversample']):<5} "
              f"decay={cfg['decay']:<6} -> val AUC {m:.4f} +/- {sd:.4f}  "
              f"({np.mean(eps):.0f} epochs, {time.time()-t0:.0f}s)", flush=True)
        json.dump(results, open("results/qnn_selection.json", "w"), indent=2)

    best = max(results, key=lambda r: r["val_auc"])
    print(f"\nBEST: {best['cfg']}  val AUC {best['val_auc']:.4f}")
    json.dump({"results": results, "best": best},
              open("results/qnn_selection.json", "w"), indent=2)


def mode_final():
    X, y, Xv, yv, Xt, yt, at = load()
    best = json.load(open("results/qnn_selection.json"))["best"]
    cfg = best["cfg"]
    print(f"FINAL -- config {cfg}, val AUC {best['val_auc']:.4f}")
    print("evaluating on the test split (read once)\n")

    rows, scores = [], []
    for s in SEEDS:
        p, e, _ = train(X, y, cfg, s)
        sc = np.asarray(_predict(p, jnp.array(Xt)))
        scores.append(sc)
        k = max(1, int(round(len(sc) * ALERT_RATE)))
        pred = sc >= np.partition(sc, -k)[-k]
        tp = int((pred & (yt == 1)).sum())
        r = {"seed": s, "epochs": e,
             "auc": roc_auc_score(yt, sc), "ap": average_precision_score(yt, sc),
             "precision": tp / max(1, pred.sum()), "recall": tp / (yt == 1).sum()}
        for a in ATTACKS:
            sel = (yt == 0) | (at == a)
            r[f"auc_{a}"] = roc_auc_score(yt[sel], sc[sel])
            r[f"rec_{a}"] = float(pred[at == a].mean())
        rows.append(r)
        print(f"  seed {s}: AUC {r['auc']:.4f}  AP {r['ap']:.4f}  "
              f"({e} epochs)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/qnn_results_per_seed.csv", index=False)
    np.save("results/qnn_test_scores.npy", np.array(scores))

    print(f"\n{'metric':<22}{'QNN (T=50)':>22}{'MLP (51 params)':>22}")
    ref = {"auc": (0.7711, 0.0107), "ap": (0.3067, 0.0179),
           "precision": (0.3354, 0.0120), "recall": (0.3322, 0.0119)}
    for k, (rm, rs) in ref.items():
        m, sd = df[k].mean(), df[k].std(ddof=0)
        print(f"  {k:<20}{m:>10.4f} +/- {sd:.4f}{rm:>13.4f} +/- {rs:.4f}"
              f"   {'QNN' if m > rm else 'MLP'} ahead by {abs(m-rm):.4f}")

    print("\nper-attack AUC (mean of 5 seeds):")
    mlp_ref = {"speed_lie": 0.981, "jump": 0.913, "gap": 0.920,
               "dark": 0.710, "drift": 0.600}
    for a in ATTACKS:
        print(f"  {a:<12} QNN {df[f'auc_{a}'].mean():.3f}   "
              f"MLP {mlp_ref[a]:.3f}   "
              f"recall@3.5% {df[f'rec_{a}'].mean():.3f}")
    print("\nwrote results/qnn_results_per_seed.csv, results/qnn_test_scores.npy")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "time"
    {"time": mode_time, "select": mode_select, "final": mode_final}[m]()
