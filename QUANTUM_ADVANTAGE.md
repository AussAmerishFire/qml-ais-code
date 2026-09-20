# Where quantum did better — and where it could

Two parts, kept strictly apart. **Part 1 is measured and claimable.** Part 2 is
future work supported by the literature, and must be written as possibility, not
result. Mixing them is how QML papers end up overclaiming.

Every number in Part 1 comes from a systematic scan of *every* Hybrid Quantum
Neural Network versus its matched classical twin, across every metric and attack
type, at both endpoints — 54 comparisons in all — not from picking favourites.
Test set, 10 paired seeds, Wilcoxon signed-rank, p < 0.05.

---

# Part 1 — Where quantum measurably won

## The headline: a clean split by data size

| Training data | Significant quantum wins | Significant quantum losses |
|---|---:|---:|
| **Full (N = 68,296)** | **19** | **0** |
| Scarce (N = 571) | 0 | 6 |

**With enough data, a Hybrid Quantum Neural Network beats a classical network of
the same size and depth on essentially every metric, and never significantly
loses.** With scarce data the reverse holds. That symmetry is the finding.

## 1. Transponder switch-off detection — the strongest single result

All three Hybrid Quantum Neural Networks beat their twins, all significantly:

| Comparison | Quantum | Matched classical | Δ | p |
|---|---:|---:|---:|---:|
| Classical pre-layer HQNN vs matched MLP | 0.819 | 0.746 | **+0.073** | 0.002 |
| Frozen circuit HQNN vs matched MLP | 0.814 | 0.741 | **+0.073** | 0.010 |
| Data re-uploading HQNN vs matched MLP | 0.812 | 0.769 | **+0.044** | 0.004 |

Transponder switch-off ("going dark") is the attack where a vessel stops
broadcasting and reappears. It is also the attack most associated with real
sanctions evasion and illegal fishing, which makes this the most operationally
meaningful quantum result in the study. All three HQNNs also beat Logistic
Regression (0.780) on it. Against the separate 71-parameter Multi-Layer Perceptron
(0.818) only the pre-layer HQNN is ahead, by +0.002 — a tie within seed noise.
**Both tree ensembles are far ahead on this attack (0.961)**, so this is a win
over matched networks, not over the best classical detector.

*(Corrected 2026-09-16: this previously said the HQNNs beat the Multi-Layer
Perceptron baseline. Two of the three score below it.)*

## 2. PR-AUC — the metric that matters at 3.5% imbalance

| Comparison | Δ PR-AUC | p |
|---|---:|---:|
| Frozen circuit HQNN vs matched MLP | **+0.086** | 0.004 |
| Classical pre-layer HQNN vs matched MLP | **+0.039** | 0.006 |
| Data re-uploading HQNN vs matched MLP | **+0.026** | 0.027 |

All three significant. PR-AUC weights the top of the ranking — the alerts an
analyst actually opens — so this is a stronger claim than a ROC-AUC win.

## 3. Operating-point precision and recall

At the 3.5% alert budget, the pre-layer and frozen HQNNs catch significantly more
anomalies *and* raise significantly fewer false alarms than their twins (recall
+0.012 and +0.050; precision +0.012 and +0.050; p ≤ 0.037).

## 4. Reporting gaps and speed falsification

All three HQNNs significantly beat their twins on both. The margins on speed
falsification are small (+0.002 to +0.054), but the data re-uploading HQNN
scores **0.998 — the highest of any model in the study**, including Gradient
Boosting (0.994).

## 5. Against the other classical models

At full data all three HQNNs clearly beat Isolation Forest (0.7568) and the
Single-Threshold Rule (0.7113). Against the 71-parameter Multi-Layer Perceptron
(0.8041) and Logistic Regression (0.8016) they are **level, not ahead**: the
10-seed HQNN means (0.8067–0.8074) sit 0.003–0.006 above them, less than the
HQNNs' own seed standard deviation (0.010–0.014), and in the 5-seed full sweep
only the frozen-circuit HQNN (0.8056) stays above both. No test across these
model families was run, so no ranking between them should be claimed.

**No parameter-efficiency claim is possible.** That Multi-Layer Perceptron is a
12-5-1 network with 71 parameters — fewer than any HQNN (106–161) — and it
matches them. Quantum does not do more per parameter here.

*(Corrected 2026-09-16: this previously said all three HQNNs beat the
Multi-Layer Perceptron and Logistic Regression and ranked third to fifth.)*

**One lead worth reporting as such:** on gradual drift — the hardest attack —
all three HQNNs (0.650–0.651, 10 seeds) score above both tree ensembles
(Gradient Boosting 0.637, Random Forest 0.612). Logistic Regression is best on
drift (0.664), and the margins are inside seed noise, so this is a direction for
future work, not a result.

## 6. Stability — for one model only

The frozen circuit HQNN is **6.1× more stable across seeds** than its twin (test
AUC standard deviation 0.0096 against 0.0589). **This does not generalise:** the
pre-layer HQNN ties its twin (0.0112 vs 0.0115) and the re-uploading HQNN is
*less* stable (0.0141 vs 0.0059). Claim it for the frozen model only.

## 7. The quantum kernel at small N

The Quantum Support Vector Machine beats every matched Multi-Layer Perceptron at
N = 571 (validation AUC 0.752 against 0.642–0.677, a margin of 0.075–0.111), and
every variational quantum model by 0.13–0.21. It is the only quantum model that
does not collapse when data is scarce. *Three seeds — effect size only, no
significance test is possible.*

---

## The limits these wins must be reported with

**Every Part 1 win is under a parameter-and-depth constraint.** No quantum model
beats Gradient Boosting (0.8316) or Random Forest (0.8225) at any data size. The
honest statement is: *at equal model size, quantum is better with enough data.*

That constraint is not artificial in every setting. It is the real constraint
when a model must be small — on-vessel or buoy-mounted edge hardware, low-power
satellite payloads — or if quantum hardware eventually becomes cheaper per
effective parameter. That is the legitimate framing for these results. "Quantum
beats classical" without the qualifier is not.

**And the scarce-data losses are equally real.** Six significant losses at
N = 571, including −0.260 on reporting gaps for the frozen HQNN.

---

# Part 2 — Other ways quantum could be better

Future work. Each item follows from something this study measured, and each
cited paper was verified against OpenAlex.

## 1. Test whether advantage is possible *before* building a model

**Huang, Broughton, Mohseni et al. (2021).** *Power of data in quantum machine
learning.* **Nature Communications 12.** DOI: 10.1038/s41467-021-22539-9 · 734 cites

> Introduces the **geometric difference** between a classical and a quantum kernel
> — a computable quantity that bounds whether *any* quantum kernel can outperform a
> classical one on a given dataset. Small geometric difference means no quantum
> kernel can win, whatever its design.

**Now measured** (`25_geometric_difference.py`, `RESULTS_FOLLOWUP.md` §2, 2026-09-19).
The geometric difference is 2.9 at N = 571 and 3.8 at N = 1,429, against maximum
possible values of 23.9 and 37.8. So a classical Gaussian kernel is within a
factor of about 3–4 of this quantum kernel on every possible task. On the real
anomaly labels the quantum kernel is simpler than any classical kernel at N = 571
(5 of 5 seeds), but not at N = 1,429. The room for a quantum-kernel advantage on
this data is small, and it narrows as data grows.

## 2. Fix the optimiser, because the optimiser is the bottleneck

This study's central mechanistic finding is that the scarce-data deficit is
**optimisation, not representation**: the quantum kernel, which trains a convex
SVM, does not collapse, while variational models, which run non-convex gradient
descent on circuit parameters, do.

**Stokes, Izaac, Killoran & Carleo (2020).** *Quantum Natural Gradient.*
**Quantum 4, 269.** DOI: 10.22331/q-2020-05-25-269 · 466 cites

> An optimiser that accounts for the geometry of the quantum state space rather
> than treating circuit parameters as flat Euclidean coordinates. The direct
> next experiment: rerun the scarcity sweep with quantum natural gradient instead
> of Adam and see whether the N = 571 deficit closes.

## 3. Data re-uploading as the architecture direction

**Pérez-Salinas, Cervera-Lierta, Gil-Fuster & Latorre (2020).** *Data re-uploading
for a universal quantum classifier.* **Quantum 4, 226.**
DOI: 10.22331/q-2020-02-06-226 · 628 cites

> Proves that re-encoding the input between circuit layers makes even a single
> qubit a universal classifier. In this study the re-uploading HQNN was the only
> variational model never significantly beaten by its twin at N = 571, and the
> best quantum model there. Deeper re-uploading is the natural extension.

## 4. A hybrid system that switches by data regime

Not a citation — a design that follows directly from Tables 2 and 5. The quantum
kernel is flat and strong below ~1,400 windows; variational HQNNs are
data-hungry but pull ahead above it. A deployed detector could use a quantum
kernel for newly monitored regions with few confirmed anomalies, and hand over to
a variational model as labelled incidents accumulate.

## 5. More qubits removes this study's own handicap

Every HQNN here compresses 12 features into 8 qubits through a classical layer,
while its twin reads all 12 directly — a structural disadvantage forced by the
simulation budget. At 12+ qubits that compression disappears. The quantum wins in
Part 1 were obtained *despite* it, so they are plausibly an underestimate.

## 6. Where advantage is actually proven — and why AIS is not there yet

Three results worth citing so a reader understands the gap between theory and
this application:

**Liu, Arunachalam & Temme (2021).** *A rigorous and robust quantum speed-up in
supervised machine learning.* **Nature Physics 17.**
DOI: 10.1038/s41567-021-01287-z · 540 cites
> A proven exponential quantum advantage for classification — but on a problem
> built around the discrete logarithm, which has no counterpart in vessel
> trajectories.

**Huang, Broughton, Cotler et al. (2022).** *Quantum advantage in learning from
experiments.* **Science 376.** DOI: 10.1126/science.abn7293 · 542 cites
> Demonstrated advantage when the *data itself is quantum* — produced by quantum
> sensors or quantum systems. AIS is classical radio data.

**Huang, Kueng & Preskill (2021).** *Information-theoretic bounds on quantum
advantage in machine learning.* **Physical Review Letters 126, 190505.**
DOI: 10.1103/PhysRevLett.126.190505 · 267 cites
> Shows that for classical data with average-case error, quantum models can offer
> at most a polynomial advantage in sample complexity — which frames honestly why
> large advantages should not be expected on a task like this one.

**The implication for your discussion:** proven quantum advantage lives in
structured or natively quantum problems. Classical tabular sensor data sits
outside that regime, which is exactly why this study found parity-to-modest
gains under constraint rather than a breakthrough.

## 7. Unsupervised quantum detection

At N = 571 the unsupervised Isolation Forest tied the best supervised model
(Logistic Regression, 0.8349 vs 0.8348) while every neural network, quantum or
classical, trailed by ~0.15. A quantum autoencoder trained on normal traffic only
— **Tscharke et al. (2025)**, already in `CITATIONS.md` — would compete in that
regime without any labelled anomalies, and would test whether a *quantum*
unsupervised model avoids the collapse that supervised quantum networks showed.

*(Corrected 2026-09-14: this previously said Isolation Forest was "the strongest
model below ~100 labelled anomalies". It tied Logistic Regression rather than
beating it.)*

---

## 8. Data with the structure a quantum kernel is built for

**Glick, Gujarati, Córcoles et al. (2024).** *Covariant quantum kernels for data
with group structure.* **Nature Physics.** DOI: 10.1038/s41567-023-02340-9 ·
63 cites (verified OpenAlex 2026-09-17)

> Quantum kernels designed around a symmetry group in the data. Vessel tracks
> have natural symmetries — rotating every course, or shifting every position,
> should not change whether a track is anomalous. A kernel built around those
> symmetries is the most AIS-specific quantum design this study did not test.

## 9. Maritime data that is itself quantum

Huang et al. (2022, *Science*, Part 2 §6) prove advantage when the data comes
from a quantum system. AIS is classical radio, so it sits outside that regime —
but quantum sensors (magnetometers, atomic clocks, cold-atom inertial
navigation) are being developed for GPS-independent navigation and vessel
detection. A spoofing detector that checks reported positions against
**quantum sensor readings, processed before they are measured**, is the setting
where theory most strongly supports a quantum learning advantage. This is the
future-implications link between this study and a provable advantage.

## 10. Why speed-ups on classical data should not be claimed

**Tang (2019).** *A quantum-inspired classical algorithm for recommendation
systems.* **STOC 2019.** DOI: 10.1145/3313276.3316310 · 232 cites (verified)

> Showed a claimed exponential quantum speed-up for a classical-data problem
> could be matched classically. Together with Huang, Kueng & Preskill (2021),
> this is why the paper should not predict large speed-ups for AIS.

## 11. Framing the research question honestly

**Schuld & Killoran (2022).** *Is quantum advantage the right goal for quantum
machine learning?* **PRX Quantum 3, 030101.** DOI: 10.1103/PRXQuantum.3.030101 ·
212 cites (verified)

> Argues that "beat classical" is too narrow a goal and that understanding
> *when and why* quantum models behave differently is the more useful question.
> Direct support for an RQ of the form "under what conditions…".

**Gil-Fuster, Eisert & Bravo-Prieto (2024).** *Understanding quantum machine
learning also requires rethinking generalization.* **Nature Communications 15.**
DOI: 10.1038/s41467-024-45882-z · 75 cites (verified)

> Shows that uniform generalisation bounds such as √(T/N) do not tell the whole
> story for quantum models — relevant context for this study's finding that the
> bound's small-data inference did not hold.

---

## Suggested one-paragraph future-implications framing

> Under a parameter-and-depth constraint, hybrid quantum neural networks
> significantly outperformed matched classical networks on 19 of 19 significant
> comparisons at full training data — most strongly on transponder switch-off
> detection (+0.07 AUC) and PR-AUC — while losing significantly on 6 of 6 at
> scarce data. A quantum kernel, by contrast, did not degrade at small sample
> sizes, indicating that the scarce-data deficit arises from variational
> optimisation rather than from quantum representation. This points to three
> directions: geometry-aware optimisers such as quantum natural gradient;
> data-regime-switching hybrid systems; and computing the geometric difference of
> Huang et al. (2021) to establish, before model construction, whether any quantum
> kernel can outperform classical methods on maritime surveillance data.
