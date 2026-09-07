# The quantum models — architecture screen, controls, and where this leaves the study

All numbers here are **validation** AUC. The test split has not been opened for
any quantum model. Scripts: `12_qnn.py`, `13_architectures.py`, `14_controls.py`.

---

## 1. First, a correction to what was built

`12_qnn.py` builds

    AngleEmbedding -> StronglyEntanglingLayers(L=2) -> sum<Z> -> sigmoid(a*e + b)

and the write-up called it a hybrid quantum neural network. **It is not.** The
trainable `a` and `b` are an output scaling, not a classical network. This is a
pure variational quantum classifier. The precedent this study is measured
against — Naaman et al. (2025), H-FQNN for ADS-B — is genuinely hybrid:
classical layers around the circuit, trained jointly end to end.

`13_architectures.py` builds the real hybrids.

---

## 2. Architecture screen (2 seeds, fixed hyperparameters)

Fixed lr 0.02, no oversampling, decay 1e-4 for every architecture, so this
compares **architectures, not tuning luck**. Each quantum model is matched to
the smallest MLP `8 -> h -> 1` with at least as many parameters, so the
classical twin never has fewer.

| Arch | Structure | T | MLP | Quantum | Classical | Δ |
|---|---|---:|---:|---:|---:|---:|
| A_vqc | circuit only (**not hybrid**) | 50 | 51 | 0.7969 | 0.8282 | −0.031 |
| B_post | circuit → Dense(8→1) | 57 | 61 | 0.8103 | 0.8440 | −0.034 |
| **C_pre** | **Dense(8→8) → circuit** | 122 | 131 | **0.8456** | 0.8297 | **+0.016** |
| D_both | Dense → circuit → Dense | 129 | 131 | 0.8218 | 0.8297 | −0.008 |
| E_reup | re-uploading → Dense(8→1) | 57 | 61 | 0.8343 | 0.8440 | −0.010 |
| F_deep | circuit L=4 → Dense(8→1) | 105 | 111 | 0.8186 | 0.8188 | −0.000 |

What the screen showed, before controls:

- **Where the classical layer goes matters more than whether one exists.** A
  pre-layer (C, +0.049 over the pure VQC) beat a post-layer (B, +0.013).
- **Both is worse than one.** D scored 0.024 below C.
- **Data re-uploading is the best value per parameter.** E matched B's
  parameter count exactly and gained 0.024, consistent with Schuld, Sweke &
  Meyer (2021) on encoding repetition widening the representable spectrum.
- **Depth in the circuit bought nothing.** F (L=4) cost 97 minutes and scored
  below C (L=2).

---

## 3. Controls on C_pre — and the screen result does not survive

C_pre is a **two-stage** model; its screen twin was a **one**-hidden-layer MLP.
Depth buys representational power independently of parameter count, so the
comparison was not clean. Three controls, 3 seeds each, identical settings:

| Control | Trainable | Val AUC |
|---|---:|---:|
| C_pre, quantum, as screened | 122 | 0.8357 ± 0.021 |
| — circuit **frozen** (random gates, never trained) | 74 | 0.8270 ± 0.017 |
| — circuit **removed** entirely | 74 | *(control invalid — see §4)* |
| **MLP (8, 5) — 2 hidden layers, depth-matched** | **123** | **0.8389 ± 0.016** |
| MLP (13,) — 1 hidden layer | 131 | 0.8313 ± 0.014 |
| MLP (5,) — 1 hidden layer | 51 | 0.8359 ± 0.016 |

**Three findings, in order of importance.**

**(a) The advantage was depth, plus a two-seed artefact.** The depth-matched
classical MLP scores 0.8389 against C_pre's 0.8357. The quantum model is
*behind* once the classical twin is allowed the same number of stages.

**(b) The screen's 0.8456 was a false positive produced by running two seeds.**
With 2 seeds C_pre read 0.8456 ± 0.003; with 3 it reads 0.8357 ± 0.021. The
standard deviation grew sevenfold on adding a single seed — two seeds cannot
estimate a variance, and the apparent precision was an illusion. Every
architecture number in §2 carries the same weakness and none of them should be
quoted without re-running at 5 seeds.

**(c) Freezing the quantum gates costs almost nothing.** 0.8270 frozen versus
0.8357 trained — a gap of 0.009 against standard deviations near 0.02. **The 48
trainable rotation angles are close to decorative**; a random fixed circuit
serves as nearly as good a feature map. This is the most consequential result
on the page, and §5 explains why.

Also worth noting: an MLP with **51** parameters and one hidden layer (0.8359)
matches the 122-parameter quantum hybrid.

---

## 4. A control that failed, recorded rather than quietly dropped

The "circuit removed" arm returned val AUC **0.5000 ± 0.000** — exactly chance,
with zero variance across seeds. That is a bug signature, not a measurement.

Cause: the readout is `sigmoid(a * s + b)` with `a` initialised to 4. On the
circuit path `s` is a sum of eight `<Z_i>`, each in [−1, 1] and centred on zero.
With the circuit removed, `s` became a sum of eight `pi*sigmoid(.)` terms, each
in [0, pi] and centred on ~12.6 — so the readout was `sigmoid(~50) = 1.0`, the
gradient was identically zero, and nothing trained.

Re-running with the pre-activation centred gives 0.6363 ± **0.279**. The
variance across three seeds is enormous (some seeds still collapse), so this
control is unstable and **is not used as evidence in either direction**. It is
recorded because a control that silently returns chance is exactly the kind of
result that gets mistaken for a finding.

The two controls that matter — frozen gates, and the depth-matched MLP — are
unaffected by this and stand.

---

## 5. Where this leaves the study

**At full training data, no quantum architecture beats the classical baseline.**
Best quantum 0.8357; tuned classical MLP 0.8441; random forest 0.8659. This
matches Naaman et al. (parity, not advantage) and is exactly the pattern Bowles,
Ahmed & Schuld (2024) report once classical baselines are tuned honestly.

**This is the expected result, not a failed experiment.** Caro et al. bound the
generalisation gap as sqrt(T/N). With T ≈ 50–122 and N = 68,296:

    sqrt(50/68296)  = 0.027
    sqrt(122/68296) = 0.042

Generalisation is not the binding constraint at this sample size — for either
model. **The hypothesis predicts no advantage here.** The full-data comparison
is the *control condition*; the experiment that actually tests RQ2 is the
label-scarcity sweep, where N falls to 571 and sqrt(T/N) rises to 0.30.

**But finding (c) is a genuine threat to the framing.** Caro's bound is stated
for models with T *trainable* gates. If freezing those gates costs only 0.009
AUC, then the trainable quantum parameters are not what is doing the work, and
invoking a bound about them is hard to defend. Two honest responses, and the
paper should take the first:

1. **Report it.** "Trainable gates contributed 0.009 ± 0.02 AUC over frozen
   random gates at full data" is a real, citable measurement, and it is the kind
   of control the QML literature is criticised for omitting. Then test whether
   the frozen/trained gap *widens* as N shrinks — which is what the bound
   actually predicts, and which would turn this from a threat into the result.
2. Drop the Caro framing and present the study as an empirical benchmark. Weaker
   paper, but defensible.

---

## 6. What runs next

The scarcity sweep should carry **two parameter scales**, not one, so it tests
the *shape* of sqrt(T/N) rather than a single point:

| Scale | Quantum | T | Classical twin | Params |
|---|---|---:|---|---:|
| small | E_reup (hybrid + re-uploading) | 57 | MLP (6,) | 61 |
| large | C_pre (classical pre-layer) | 122 | MLP (8, 5) | 123 |

with the **frozen-gate variant of each** included as the control that decides
whether trainable quantum parameters matter, swept over
N = 571 / 1,429 / 2,857 / 5,714 / 14,286 (holding the 3.5% anomaly rate fixed —
design (B) in `BASELINES.md` §5), 5 seeds throughout.

Isolation Forest is the flat reference line across the whole sweep, since it
uses no labels at all.

---

## 7. Validation confusion matrices — `fig_q1_confusion.png`

Built by `15_qnn_confusion.py`. **Validation set, seed 0 only.** The test split
remains closed for every quantum model.

All six models flag the same 537 windows (validation's own 3.09% anomaly rate),
so only *which* windows differ.

| Panel | T | AUC | Caught (of 537) | Recall |
|---|---:|---:|---:|---:|
| A — pure VQC (not hybrid) | 50 | 0.7958 | 189 | 35.2% |
| E — hybrid + re-uploading | 57 | 0.8334 | 183 | 34.1% |
| C — hybrid, classical pre-layer | 122 | 0.8425 | 206 | 38.4% |
| **C frozen — gates never trained** | **74** | 0.8333 | **212** | **39.5%** |
| MLP (8,5) — depth-matched | 123 | 0.8191 | 202 | 37.6% |
| MLP (5,) | 51 | 0.8138 | 199 | 37.1% |

**The frozen-circuit model catches more anomalies than the trained one** — 212
against 206 — with 48 fewer trainable parameters, despite a slightly lower AUC.
AUC and operating-point recall disagree here because AUC scores the whole
ranking while recall at a fixed budget scores only the top 3.09%. Both numbers
are real; they answer different questions.

### READ THIS BEFORE QUOTING THE FIGURE

**Every quantum model appears to beat both classical twins in this figure. Do
not report that.** These are single-seed (seed 0) results, and the 3-seed means
in section 3 say the opposite:

| Model | seed 0 (fig_q1) | 3-seed mean (section 3) |
|---|---:|---:|
| C_pre | 0.8425 | 0.8357 ± 0.021 |
| MLP (8,5) | 0.8191 | **0.8389 ± 0.016** |

Seed 0 is simply a bad draw for the classical model and a good one for the
quantum model. The MLP's seed-to-seed standard deviation is 0.016, which is
larger than the entire apparent gap.

This is the *same* trap that produced the false positive in section 3(b), where
a two-seed screen reported 0.8456 ± 0.003 and three seeds turned it into
0.8357 ± 0.021. A confusion matrix is a single draw by construction, which is
why it is shown at one seed — but that makes it useless for ranking models.

**Use the figure to see the shape of each model's errors. Use section 3 for
which model is better. The sweep (`16_sweep.py`) reports 5 seeds throughout.**
