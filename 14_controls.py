"""
14_controls.py
==============
Ablation controls for C_pre, the one architecture in the screen that beat its
parameter-matched classical twin (0.8456 vs 0.8297 validation AUC).

C_pre is  Dense(8->8) -> pi*sigmoid -> AngleEmbedding -> SEL(2) -> sum<Z>
          -> sigmoid(a*e + b),  T = 72 + 48 + 2 = 122.

Before that +0.016 can be called anything, three things must be ruled out.
Each control removes exactly one ingredient and changes nothing else.

  1. DEPTH, not quantumness.
     C_pre is a two-stage model; its twin in the screen was a ONE-hidden-layer
     MLP (8->13->1). Depth buys representational power on its own. The honest
     classical comparison is a TWO-hidden-layer MLP of the same size:
     8->8->5->1 has 123 parameters against C_pre's 122.

  2. TRAINABLE GATES, not a random feature map.
     If the circuit's 48 rotation angles are frozen at their random
     initialisation and only the classical parts train, does performance hold?
     If it does, the quantum parameters are decorative -- and since Caro et
     al.'s bound is stated for *trainable* gates, the study's whole premise
     would not apply to this model.

  3. THE CIRCUIT AT ALL.
     Delete the circuit, keep the classical pre-layer and the summed readout:
     Dense(8->8) -> pi*sigmoid -> sum -> sigmoid(a*s + b), 74 parameters.
     If that alone reaches 0.845, the circuit contributes nothing.

3 seeds each. Validation only -- the test split is not opened here.
"""

import json
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp          # noqa: E402
import numpy as np               # noqa: E402
import optax                     # noqa: E402
import pandas as pd              # noqa: E402
import pennylane as qml          # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402
from sklearn.neural_network import MLPClassifier     # noqa: E402

N_QUBITS, BATCH, MAX_EPOCHS, PATIENCE = 8, 4096, 60, 10
SEEDS = [0, 1, 2]
CFG = {"lr": 0.02, "oversample": None, "decay": 1e-4}   # identical to the screen

FEATS = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
         "spd_mismatch_max", "crab_mean", "prev_silence_s",
         "course_resid_mean", "mismatch_slope", "straightness",
         "mismatch_mean", "crab_valid"]

_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_dev, interface="jax", diff_method="backprop")
def _circ(x, w):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def forward_factory(variant):
    """variant: 'full' | 'frozen' | 'nocircuit'"""
    def fwd(p, frozen_w, X):
        X = jnp.pi * jax.nn.sigmoid(X @ p["Wp"] + p["bp"])
        if variant == "nocircuit":
            s = jnp.sum(X, axis=-1)
        else:
            w = frozen_w if variant == "frozen" else p["w"]
            s = jnp.sum(jnp.stack(_circ(X, w), axis=-1), axis=-1)
        return p["a"] * s + p["b"]
    return fwd


def train_quantum(variant, X, y, seed):
    fwd = forward_factory(variant)
    rng = np.random.default_rng(seed)
    shp = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=N_QUBITS)

    frozen_w = jnp.array(rng.normal(0, 0.1, shp))
    p = {"Wp": jnp.array(rng.normal(0, 0.3, (N_QUBITS, N_QUBITS))),
         "bp": jnp.zeros(N_QUBITS),
         "a": jnp.array(4.0), "b": jnp.array(0.0)}
    if variant == "full":
        p["w"] = jnp.array(rng.normal(0, 0.1, shp))

    @jax.jit
    def loss_grad(p, Z, t):
        def L(p):
            q = jnp.clip(jax.nn.sigmoid(fwd(p, frozen_w, Z)), 1e-7, 1 - 1e-7)
            reg = sum(jnp.sum(v ** 2) for k, v in p.items() if k in ("w", "Wp"))
            return -jnp.mean(t * jnp.log(q) + (1 - t) * jnp.log(1 - q)) \
                + CFG["decay"] * reg
        return jax.value_and_grad(L)(p)

    predict = jax.jit(lambda p, Z: jax.nn.sigmoid(fwd(p, frozen_w, Z)))

    idx = rng.permutation(len(y))
    cut = max(BATCH, int(0.10 * len(idx)))
    Xa, ya = jnp.array(X[idx[cut:]]), jnp.array(y[idx[cut:]].astype(float))
    ev = idx[:cut][:(cut // BATCH) * BATCH]
    Xb, yb = jnp.array(X[ev]), y[ev]
    n_full = (len(ya) // BATCH) * BATCH

    opt = optax.adam(CFG["lr"])
    state = opt.init(p)
    best, best_p, since = -np.inf, p, 0
    for _ in range(MAX_EPOCHS):
        order = rng.permutation(len(ya))[:n_full]
        for s in range(0, n_full, BATCH):
            b = order[s:s + BATCH]
            _, g = loss_grad(p, Xa[b], ya[b])
            u, state = opt.update(g, state, p)
            p = optax.apply_updates(p, u)
        auc = roc_auc_score(yb, np.asarray(predict(p, Xb)))
        if auc > best + 1e-5:
            best, best_p, since = auc, p, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    return best_p, predict


def main():
    tr = pd.read_csv("dataset_train.csv")
    va = pd.read_csv("dataset_val.csv")
    X, y = tr[FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[FEATS].to_numpy(float), va.label.to_numpy(int)
    Xvj = jnp.array(Xv)

    print("CONTROLS FOR C_pre -- validation only, 3 seeds, identical settings\n")
    print(f"{'control':<34}{'trainable':>11}{'val AUC':>18}")
    print("-" * 64)
    rows = []

    def report(name, params, aucs, extra=""):
        m, s = float(np.mean(aucs)), float(np.std(aucs))
        rows.append({"control": name, "params": params,
                     "auc": m, "std": s, "note": extra})
        print(f"{name:<34}{params:>11}{m:>12.4f} +/-{s:.3f}", flush=True)
        json.dump(rows, open("controls.json", "w"), indent=2)

    # --- the model under test -------------------------------------------
    t0 = time.time()
    a = []
    for s in SEEDS:
        p, pred = train_quantum("full", X, y, s)
        a.append(roc_auc_score(yv, np.asarray(pred(p, Xvj))))
    report("C_pre (quantum, as screened)", 122, a)

    # --- control 2: gates frozen ----------------------------------------
    a = []
    for s in SEEDS:
        p, pred = train_quantum("frozen", X, y, s)
        a.append(roc_auc_score(yv, np.asarray(pred(p, Xvj))))
    report("  circuit FROZEN (random gates)", 74, a)

    # --- control 3: no circuit at all -----------------------------------
    a = []
    for s in SEEDS:
        p, pred = train_quantum("nocircuit", X, y, s)
        a.append(roc_auc_score(yv, np.asarray(pred(p, Xvj))))
    report("  circuit REMOVED entirely", 74, a)

    # --- control 1: depth-matched classical ------------------------------
    for hl, n in ((( 8, 5), 123), ((13,), 131), ((5,), 51)):
        a = []
        for s in SEEDS:
            m = MLPClassifier(hidden_layer_sizes=hl, alpha=1e-4,
                              learning_rate_init=0.01, max_iter=500,
                              early_stopping=True, n_iter_no_change=20,
                              random_state=s).fit(X, y)
            a.append(roc_auc_score(yv, m.predict_proba(Xv)[:, 1]))
        depth = "2 hidden layers" if len(hl) == 2 else "1 hidden layer"
        report(f"MLP {hl} — {depth}", n, a)

    print(f"\n[{time.time()-t0:.0f}s]  wrote controls.json")


if __name__ == "__main__":
    main()
