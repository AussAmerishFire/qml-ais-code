"""
13_architectures.py
===================
Architecture screen: six quantum models, each with its own parameter-matched
classical twin. Validation only -- the test split is not opened here.

WHY THIS EXISTS
    The first model built (12_qnn.py) was a *pure variational quantum
    classifier*: AngleEmbedding -> StronglyEntanglingLayers -> <sum Z> ->
    sigmoid(a*e + b). The trainable `a` and `b` are an output scaling, not a
    neural network. Calling that a "hybrid quantum neural network" would be
    wrong, and the precedent this study is measured against -- Naaman et al.
    (2025), H-FQNN for ADS-B -- is genuinely hybrid: classical layers around
    the circuit, trained jointly.

    This file builds the real hybrids and screens them honestly.

THE PARAMETER-MATCHING RULE
    Every quantum architecture has some trainable-parameter count T. Its
    classical twin is an MLP 8 -> h -> 1 with 10h + 1 parameters, and h is
    chosen as the SMALLEST h with 10h + 1 >= T. The classical model therefore
    always has at least as many parameters as the quantum one -- if a quantum
    model wins, it did not win on capacity.

    T is also the numerator of the Caro et al. (2022) bound, sqrt(T/N), so it
    is recorded exactly for every architecture, not approximated.

PROTOCOL
    Fixed hyperparameters for every architecture (lr 0.02, no oversampling,
    weight decay 1e-4) so the screen compares ARCHITECTURES, not tuning luck.
    2 seeds each. The winner then gets the full 12-config x 5-seed grid, the
    same budget the classical MLP received.

    Selection here is on validation. Test stays closed until 14_final.py.
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
from sklearn.metrics import roc_auc_score            # noqa: E402
from sklearn.neural_network import MLPClassifier     # noqa: E402

N_QUBITS = 8
BATCH = 4096
MAX_EPOCHS = 60
PATIENCE = 10
SCREEN_SEEDS = [0, 1]

FEATS = ["sog_mean", "sog_max_jump", "cog_turn_max", "dt_max",
         "spd_mismatch_max", "crab_mean", "prev_silence_s",
         "course_resid_mean", "mismatch_slope", "straightness",
         "mismatch_mean", "crab_valid"]

_dev = qml.device("default.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# CIRCUITS
# Each returns the eight single-qubit Z expectation values. Returning them
# separately (rather than pre-summed) costs nothing -- they are all diagonal in
# the computational basis -- and lets the classical readout layer weight them.
# ---------------------------------------------------------------------------

def _make_circuit(n_layers, reupload):
    @qml.qnode(_dev, interface="jax", diff_method="backprop")
    def circuit(x, w):
        if reupload:
            # Data re-uploading: the input is re-encoded before every block.
            # Same parameter count, strictly greater expressive power --
            # Schuld, Sweke & Meyer (2021) show encoding repetition widens the
            # frequency spectrum the model can represent.
            for l in range(n_layers):
                qml.AngleEmbedding(x, wires=range(N_QUBITS))
                qml.StronglyEntanglingLayers(w[l:l + 1], wires=range(N_QUBITS))
        else:
            qml.AngleEmbedding(x, wires=range(N_QUBITS))
            qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return circuit


# ---------------------------------------------------------------------------
# ARCHITECTURES
# ---------------------------------------------------------------------------

ARCHS = {
    "A_vqc": dict(
        layers=2, pre=False, post="sum", reupload=False,
        note="pure VQC, summed readout -- what 12_qnn.py built. NOT hybrid."),
    "B_post": dict(
        layers=2, pre=False, post="linear", reupload=False,
        note="hybrid: trainable classical layer over the 8 measured qubits"),
    "C_pre": dict(
        layers=2, pre=True, post="sum", reupload=False,
        note="hybrid: classical layer feeds the circuit"),
    "D_both": dict(
        layers=2, pre=True, post="linear", reupload=False,
        note="full hybrid: classical -> quantum -> classical, end to end"),
    "E_reup": dict(
        layers=2, pre=False, post="linear", reupload=True,
        note="hybrid + data re-uploading (same T as B, more expressive)"),
    "F_deep": dict(
        layers=4, pre=False, post="linear", reupload=False,
        note="deeper circuit, L=4 -- doubles the quantum parameter count"),
}


def param_count(a):
    shp = qml.StronglyEntanglingLayers.shape(n_layers=a["layers"], n_wires=N_QUBITS)
    t = int(np.prod(shp))
    t += (N_QUBITS * N_QUBITS + N_QUBITS) if a["pre"] else 0       # dense 8->8
    t += (N_QUBITS + 1) if a["post"] == "linear" else 2            # dense 8->1, or a,b
    return t


def matched_h(T):
    """Smallest hidden width whose MLP has at least T parameters."""
    h = 1
    while 10 * h + 1 < T:
        h += 1
    return h


def init_params(a, rng):
    shp = qml.StronglyEntanglingLayers.shape(n_layers=a["layers"], n_wires=N_QUBITS)
    p = {"w": jnp.array(rng.normal(0, 0.1, shp))}
    if a["pre"]:
        # Small init so the pre-layer starts near a plain pass-through of the
        # already-scaled features rather than scrambling them.
        p["Wp"] = jnp.array(rng.normal(0, 0.3, (N_QUBITS, N_QUBITS)))
        p["bp"] = jnp.zeros(N_QUBITS)
    if a["post"] == "linear":
        p["Wo"] = jnp.array(rng.normal(0, 0.5, (N_QUBITS,)))
        p["bo"] = jnp.array(0.0)
    else:
        p["a"] = jnp.array(4.0)
        p["b"] = jnp.array(0.0)
    return p


def make_forward(a):
    circ = _make_circuit(a["layers"], a["reupload"])

    def forward(p, X):
        if a["pre"]:
            # pi * sigmoid keeps the encoded angles in [0, pi] -- the same
            # domain as the raw scaled features, so the circuit sees a
            # consistent input range whether or not the pre-layer is present.
            X = jnp.pi * jax.nn.sigmoid(X @ p["Wp"] + p["bp"])
        z = jnp.stack(circ(X, p["w"]), axis=-1)        # (batch, 8)
        if a["post"] == "linear":
            return z @ p["Wo"] + p["bo"]
        return p["a"] * jnp.sum(z, axis=-1) + p["b"]

    return forward


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train(a, Xtr, ytr, cfg, seed):
    fwd = make_forward(a)

    @jax.jit
    def loss_grad(p, X, t):
        def L(p):
            q = jnp.clip(jax.nn.sigmoid(fwd(p, X)), 1e-7, 1 - 1e-7)
            reg = sum(jnp.sum(v ** 2) for k, v in p.items() if k in ("w", "Wp", "Wo"))
            return -jnp.mean(t * jnp.log(q) + (1 - t) * jnp.log(1 - q)) \
                + cfg["decay"] * reg
        return jax.value_and_grad(L)(p)

    predict = jax.jit(lambda p, X: jax.nn.sigmoid(fwd(p, X)))
    rng = np.random.default_rng(seed)

    if cfg["oversample"]:
        pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
        want = int(len(neg) * cfg["oversample"] / (1 - cfg["oversample"]))
        pos = rng.choice(pos, size=want, replace=True)
        k = rng.permutation(np.concatenate([pos, neg]))
        Xtr, ytr = Xtr[k], ytr[k]

    idx = rng.permutation(len(ytr))
    cut = max(BATCH, int(0.10 * len(idx)))
    Xa, ya = jnp.array(Xtr[idx[cut:]]), jnp.array(ytr[idx[cut:]].astype(float))
    # Fixed-size eval block: JAX recompiles for every new input shape, and a
    # ragged tail was costing an 8-second recompile per distinct length.
    ev = idx[:cut][:(cut // BATCH) * BATCH]
    Xb, yb = jnp.array(Xtr[ev]), ytr[ev]

    n_full = (len(ya) // BATCH) * BATCH          # drop the ragged tail; a fresh
    p = init_params(a, rng)                      # shuffle each epoch means no
    opt = optax.adam(cfg["lr"])                  # sample is dropped twice
    state = opt.init(p)

    best, best_p, best_ep, since = -np.inf, p, 0, 0
    for ep in range(MAX_EPOCHS):
        order = rng.permutation(len(ya))[:n_full]
        for s in range(0, n_full, BATCH):
            b = order[s:s + BATCH]
            _, g = loss_grad(p, Xa[b], ya[b])
            u, state = opt.update(g, state, p)
            p = optax.apply_updates(p, u)
        auc = roc_auc_score(yb, np.asarray(predict(p, Xb)))
        if auc > best + 1e-5:
            best, best_p, best_ep, since = auc, p, ep, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    return best_p, best_ep + 1, predict


def main():
    tr = pd.read_csv("dataset_train.csv")
    va = pd.read_csv("dataset_val.csv")
    X, y = tr[FEATS].to_numpy(float), tr.label.to_numpy(int)
    Xv, yv = va[FEATS].to_numpy(float), va.label.to_numpy(int)
    Xvj = jnp.array(Xv)

    cfg = {"lr": 0.02, "oversample": None, "decay": 1e-4}
    print("ARCHITECTURE SCREEN -- validation only, fixed hyperparameters")
    print(f"  {cfg}\n")
    print(f"{'arch':<9}{'T':>5}{'MLP h':>7}{'MLP T':>7}"
          f"{'quantum val AUC':>20}{'classical twin':>17}")
    print("-" * 72)

    out = []
    for name, a in ARCHS.items():
        T = param_count(a)
        h = matched_h(T)
        t0 = time.time()

        qa = []
        for s in SCREEN_SEEDS:
            p, eps, predict = train(a, X, y, cfg, s)
            qa.append(roc_auc_score(yv, np.asarray(predict(p, Xvj))))

        ca = []
        for s in SCREEN_SEEDS:
            m = MLPClassifier(hidden_layer_sizes=(h,), alpha=1e-4,
                              learning_rate_init=0.01, max_iter=500,
                              early_stopping=True, n_iter_no_change=20,
                              random_state=s).fit(X, y)
            ca.append(roc_auc_score(yv, m.predict_proba(Xv)[:, 1]))

        r = {"arch": name, "T": T, "mlp_h": h, "mlp_T": 10 * h + 1,
             "q_auc": float(np.mean(qa)), "q_std": float(np.std(qa)),
             "c_auc": float(np.mean(ca)), "c_std": float(np.std(ca)),
             "epochs": eps, "secs": time.time() - t0, "note": a["note"]}
        out.append(r)
        json.dump(out, open("arch_screen.json", "w"), indent=2)

        print(f"{name:<9}{T:>5}{h:>7}{10*h+1:>7}"
              f"{r['q_auc']:>13.4f} +/-{r['q_std']:.3f}"
              f"{r['c_auc']:>11.4f} +/-{r['c_std']:.3f}"
              f"   [{r['secs']:.0f}s]", flush=True)
        print(f"         {a['note']}", flush=True)

    print("\nwrote arch_screen.json")


if __name__ == "__main__":
    main()
