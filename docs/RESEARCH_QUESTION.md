# The research question — final

Agreed 2026-09-30. This supersedes the original question, which asked whether
quantum models generalise *better* from few labels. They do not; the measured
result is the opposite, and the question was rewritten to match the evidence
rather than the expectation.

---

## Primary research question

> **Under what conditions do quantum machine learning models outperform classical
> models in detecting anomalous vessel behaviour in AIS data?**

The question is deliberately conditional. A yes/no question ("is quantum
better?") would have to be answered "no", and would throw away the finding that
the answer flips sign depending on how much data there is and what the quantum
model is compared against.

**The one-sentence answer:** at equal model size, depth and training budget,
quantum models win once training data passes a few thousand windows and lose
below it; against the best classical models they never win; and the advantage
that does exist is bounded, small, and does not transfer to industry models.

---

## Four sub-questions, and what answers each

### SQ1 — Against a classical model of the same size, does the advantage depend on training-set size?

**Hypothesis (from Caro et al. 2022).** A quantum model with `T` trainable gates
has generalisation error bounded by `sqrt(T/N)`, so a quantum model should hold
up *better* than a same-size classical model as `N` shrinks.

**Answer: it depends on size, and in the direction opposite to that prediction.**

| Training windows | Quantum − matched classical twin | Significant wins / losses |
|---:|---|---:|
| 571 | −0.058, −0.081, +0.016 | 0 wins, 6 losses |
| 68,296 | +0.020, +0.040, +0.009 | 19 wins, 0 losses |

Two of three architectures are significant at both ends (p = 0.027/0.010 and
0.020/0.002). The third, data re-uploading, is significant at neither. Measured
crossover: **2,443 windows** (frozen circuit), **7,229** (classical pre-layer),
**9,432** (data re-uploading).

*Evidence:* `COMPLETE_RESULTS.md` §5–6, `FINAL_REPORT.md` §2, figures 05, 10, 12.
*Control:* freezing the 48 quantum rotation angles changes nothing significantly
at any size — so the trainable quantum parameters are not what produces the win.

### SQ2 — Do quantum models beat the best classical models?

**Answer: no, at no training-set size.**

Gradient Boosting reaches test AUC 0.8316 and Random Forest 0.8225, against the
best quantum model's 0.8074. The gap narrows from −0.119 at 571 windows to
−0.020 at 14,286, then stops narrowing. Against Logistic Regression and the
71-parameter Multi-Layer Perceptron the quantum models are level, not ahead: the
margins are smaller than their own seed-to-seed spread.

So the quantum advantage in SQ1 exists **only** under an equal-size, equal-depth,
equal-training-budget constraint — a constraint nobody deploying a detector would
impose, but the real constraint on edge hardware.

*Evidence:* `COMPLETE_RESULTS.md` §1 and §3, figure 13.

### SQ3 — Do quantum kernels behave differently from variational quantum circuits?

**Answer: yes, and this is the sharpest mechanistic finding.**

With 571 training windows the Quantum Support Vector Machine scores 0.758 and
beats all seven neural networks in the study, quantum and classical, which sit at
0.54–0.68. It is the only quantum model that does not collapse when data is
scarce. But it barely improves with more data (+0.005 from 571 to 1,429, against
+0.117 to +0.128 for the three variational networks), and its cost grows with the square of the
training set, so it cannot reach the sizes where variational models finally win.

The kernel trains a convex problem; the variational models run gradient descent
on circuit parameters. **The scarce-data deficit therefore belongs to variational
optimisation, not to quantum feature maps as such.**

*Evidence:* `COMPLETE_RESULTS.md` §8, figures 14, 09, 04.

### SQ4 — Is there a measurable ceiling on quantum advantage for this data?

**Answer: yes, and it is low.**

The geometric difference of Huang et al. (2021) is **2.9** at 571 windows and
**3.8** at 1,429, against maxima of 23.9 and 37.8 — about a tenth of the possible
gap, and shrinking as a share as data grows. A classical Gaussian kernel can
learn anything this quantum kernel can, at no more than 8–15× the complexity.

The one point in quantum's favour: on the real anomaly labels at 571 windows, the
quantum kernel describes the data more simply than any classical kernel, on 5 of
5 seeds. That edge is gone by 1,429.

Separately, quantum features added to Random Forest and Gradient Boosting made
both **worse** on test, and were never better than the best classical control
built from the same inputs.

*Evidence:* `RESULTS_FOLLOWUP.md` §1–2.

---

## Where quantum did win, stated precisely

These are the claimable positives, all against size-matched classical twins at
full data, 10 paired seeds:

- **Transponder switch-off ("going dark") detection: +0.073 AUC** (p = 0.002) —
  the attack most associated with sanctions evasion and illegal fishing.
- **PR-AUC +0.086** (p = 0.004) — the metric that matters at 3.5% anomaly rate.
- **Speed falsification 0.998** — the highest score of any model in the study,
  ahead of Gradient Boosting's 0.994, though every model is near ceiling here.
- **Quantum kernels do not collapse with scarce data**, unlike every neural
  network tested.

---

## What this question does NOT claim

Stating these in the paper protects every claim above.

1. **No speed-up.** Nothing here measures runtime advantage, and none is expected
   for classical data (Tang 2019; Huang, Kueng & Preskill 2021).
2. **No hardware result.** All quantum models are 8-qubit noiseless simulations.
   Real hardware noise would only lower these numbers.
3. **No claim beyond 8 qubits or one feature map.** Every quantum model
   compresses 12 features into 8 qubits through a classical layer while its
   classical twin reads all 12 — a handicap in the conservative direction.
4. **No parameter-efficiency claim.** A 71-parameter classical network matches
   the 106–161-parameter quantum models.
5. **No generalisation beyond this data.** One day of US coastal AIS, with
   injected rather than observed anomalies.

---

## Title options

1. *Quantum models need more data, not less: a controlled test of generalisation-bound predictions in maritime AIS anomaly detection*
2. *Under what conditions does quantum machine learning help? Evidence from AIS maritime anomaly detection*
3. *Size-matched but not state of the art: quantum machine learning for vessel-trajectory anomaly detection*

Option 1 leads with the surprising result and is the strongest if the inversion
is the headline. Option 2 matches the research question exactly. Option 3 is the
most honest summary of the practical finding.

---

## How the paper maps onto this

| Section | Answers | Main evidence |
|---|---|---|
| Introduction | Motivates the primary question | Caro et al.; Schuld & Killoran (2022) on why "under what conditions" is the better goal |
| Related work | How AIS ML is done in practice | `CITATIONS_ROUND3.md` |
| Method | Dataset, attacks, matched twins, controls | `METHODS.md`, figures 01–03 |
| Results 1 | SQ1 | figures 05, 10, 12 |
| Results 2 | SQ2 | figure 13, `COMPLETE_RESULTS.md` |
| Results 3 | SQ3 | figures 14, 09, 04 |
| Results 4 | SQ4 | `RESULTS_FOLLOWUP.md` |
| Discussion | Where quantum could win | `QUANTUM_ADVANTAGE.md` Part 2 |
| Limitations | The five non-claims above | `FINAL_REPORT.md` §6 |
