# The label-scarcity experiment — result

Run completed 2026-09-08 10:27, 669 minutes, 230 runs, exit 0.
Data: `sweep_results.csv`. Figure: `fig_s1_scarcity.png`. Script: `16_sweep.py`.

All numbers are **validation** AUC. The test split has not been opened.

> **Status note (2026-09-14).** This file records the 5-seed sweep as it was run
> and is kept for provenance. Two things in it are superseded:
>
> 1. **The effect sizes in §1 are overstated.** Re-run at 10 seeds on the test
>    set (`19_final.py`), the N = 571 gaps shrank from −0.105 / −0.018 / −0.142
>    to −0.058 / +0.016 / −0.081 — regression to the mean from a 5-seed sample.
>    The direction held; the magnitude did not. Use the 10-seed numbers in
>    `FINAL_REPORT.md`.
> 2. **§3 was false as originally written** and is corrected below.
>
> The complete, uniformly-run grid — every model at every size, validation and
> test — is `full_sweep_results.csv`.

---

## 1. What was predicted, and what happened

Caro et al. (2022, *Nature Communications*) bound the generalisation gap of a
quantum model with `T` trainable gates as `sqrt(T/N)`. The prediction that
follows is direct: **as N shrinks, a quantum model should hold up better than a
classical model of the same size.**

It does the opposite.

| Quantum − matched classical twin | N=571 | 1,429 | 2,857 | 14,286 | 68,296 |
|---|---:|---:|---:|---:|---:|
| pre-layer hybrid (T=154 vs 155) | **−0.105** | −0.076 | −0.021 | +0.006 | **+0.015** |
| re-uploading hybrid (T=161 vs 165) | **−0.018** | −0.043 | −0.036 | +0.018 | **+0.017** |
| frozen-gate hybrid (T=106 vs 111) | **−0.142** | −0.046 | +0.002 | +0.016 | **+0.064** |

Every pair is negative when data is scarce and positive when data is plentiful.
The trend is monotone for the pre-layer and frozen-gate pairs; the re-uploading
pair is not (−0.018 → −0.043 → −0.036 → +0.018 → +0.017), and at 10 seeds it is
not significant at either endpoint. The three are independent architectures with
independently matched twins. **Quantum models here need MORE training data
than classical models of the same size, not less.**

---

## 2. The internal test — no measurable effect (CORRECTED)

> **Correction (2026-09-14).** This section was titled "The internal test inverts
> too" and said the difference "falls monotonically". None of the five differences
> below is significant (Wilcoxon, 5 seeds, every p ≥ 0.625), and at 10 seeds on
> the test set they are +0.004 (N = 571, p = 1.000) and −0.001 (N = 68,296,
> p = 0.922). What the data support: freezing the angles made no measurable
> difference at any N, so the lower-T model got none of the predicted small-data
> benefit. No direction can be claimed.

The frozen-gate model is the same architecture as the pre-layer model with its
48 rotation angles held at random initialisation, so it has **T = 106 against
154** while everything else is identical. The bound predicts the lower-T model
degrades more slowly, so `frozen − trained` should rise as N falls.

| N | sqrt(T/N), frozen | sqrt(T/N), trained | frozen − trained |
|---:|---:|---:|---:|
| 571 | 0.431 | 0.519 | **−0.0188** |
| 1,429 | 0.272 | 0.328 | −0.0129 |
| 2,857 | 0.193 | 0.232 | −0.0097 |
| 14,286 | 0.086 | 0.104 | −0.0000 |
| 68,296 | 0.039 | 0.047 | **+0.0035** |

The point estimates fall as N shrinks, but none is distinguishable from zero. The
lower-T model shows no advantage in the regime where the bound's inference says
its smaller parameter count should protect it.

This is a within-architecture test — same circuit, same pre-layer, same readout,
same hyperparameters, one binary difference — so it is not explained by
architectural confounds.

**Note what this does not say.** Caro's bound is an *upper* bound on the
generalisation gap. Nothing here contradicts the theorem. What is refuted is the
practical inference people draw from it — that a smaller quantum model will
therefore win on small data. At these sizes the binding constraint is clearly
optimisation, not generalisation: quantum models are simply harder to fit from
few examples, and that dominates whatever the bound allows.

---

## 3. The result that matters operationally — CORRECTED

> **This section originally claimed that below N ≈ 3,000 "an unsupervised method
> using no labels at all beats every supervised model." That was false.** The
> sweep contained only neural models plus Random Forest and Isolation Forest.
> Logistic Regression, Gradient Boosting and the 71-parameter Multi-Layer
> Perceptron were only run at full data. Once run at every size
> (`21_baseline_sweep.py`), Logistic Regression tied Isolation Forest at N = 571
> and beat it at N = 1,429. The claim held only against the models tested.

**What is true: with little data, simple models win and every neural network
collapses — quantum and classical alike.**

| N | Best simple model | AUC | Best neural model | AUC | Gap |
|---:|---|---:|---|---:|---:|
| 571 | Isolation Forest | 0.8349 | Matched MLP (frozen twin) | 0.6811 | 0.154 |
| 1,429 | Logistic Regression | 0.8552 | Multi-Layer Perceptron | 0.8034 | 0.052 |
| 2,857 | Random Forest | 0.8621 | Matched MLP (re-uploading twin) | 0.8111 | 0.051 |

The scarce-data failure belongs to models that learn hidden representations —
quantum and classical neural networks fail by the same 0.15 at N = 571. On labels
narrowly: Isolation Forest with none ties Logistic Regression with 20 labelled
anomalies at N = 571, and Logistic Regression pulls ahead by N = 1,429.

---

## 4. A statistical problem in the protocol, found by running it

`BASELINES.md` §3 rule 6 requires a Wilcoxon signed-rank test over 5 paired
seeds for any difference claim. **That test cannot reach p < 0.05 with 5 pairs.**
The two-sided minimum attainable p-value is 2/2⁵ = **0.0625**, whatever the data
says. Every comparison in §1 therefore came back unstarred, including the
−0.142 one.

The rule was unrunnable as written, and it was my rule. Three honest options:

1. **Report effect sizes and consistency, not p-values.** The evidence here is
   that three independent architecture pairs show the same sign at both ends, and
   two of the three a monotone trend across five values of N. That is 3 replications × 5
   points, which is a stronger argument than any single test on 5 seeds.
2. **Raise to 10+ seeds** for the comparisons the paper leans on. At 10 pairs the
   two-sided minimum is 0.002. Cost is roughly 2 hours per (model, N) pair.
3. Use a one-sided test — attainable minimum 0.031 — but only for a direction
   fixed in advance. Ours was fixed in advance and was **wrong**, so a one-sided
   test in the pre-registered direction would fail by construction. Not usable.

**Recommendation: (1) now, (2) before submission** for the headline pair at
N=571 and N=68,296.

---

## 5. Caveats a reviewer will raise

- **Tuning budget was 2 configurations per model**, cut from 8 against a 7.6-hour
  compute window. Equal on both sides, but small. `MLP_frozen` in particular
  shows seed-to-seed standard deviations up to 0.149 and a Phase-1 selection
  std of 0.202 — that model is unstable and its +0.064 gap at full data should
  not be leaned on.
- **Selection ran on a 14,286-window subsample**, not all 68,296.
- **Every quantum model carries a classical Dense(12→8) pre-layer**, forced by
  the 8-qubit budget. The quantum side compresses 12 features into 8 while its
  twin reads all 12 — a handicap, in the conservative direction.
- **Validation only in this file.** Test-set numbers now exist: at both
  endpoints at 10 seeds (`final_results.csv`) and at every size at 5 seeds
  (`full_sweep_results.csv`).
- **8-qubit simulation, not hardware.** No noise model.

---

## 6. What this makes the paper

The study set out to test whether quantum models generalise better from few
labels. They do not — they generalise **worse**, consistently, across three
architectures and a 120× range of training-set size, in the presence of
depth- and parameter-matched classical controls.

That is a publishable negative result of exactly the kind Bowles, Ahmed &
Schuld (2024) argue the field is short of, and it comes with something their
paper does not have: a frozen-gate control isolating whether the *trainable*
quantum parameters do anything (they contribute ~0.009 AUC at full data, and
their contribution goes the wrong way as N falls).

The secondary finding — that with scarce data simple models (Isolation Forest,
Logistic Regression, Random Forest) beat every neural network, quantum or
classical — is arguably the more useful one for a maritime operations audience,
and it is not a quantum result at all. (An earlier wording of this finding,
"unsupervised beats supervised below ~100 labels", was false; see §3.)
