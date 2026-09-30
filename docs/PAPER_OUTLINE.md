# Paper structure — section by section

Every section lists what it must do, which figure/table carries it, where the
numbers live, and the traps to avoid. Word budgets assume a ~6,000-word paper;
scale them if your venue differs. The ratio between sections matters more than
the totals.

**Write the prose yourself.** This file gives structure, evidence and numbers,
not sentences to copy. The one drafted passage (the abstract skeleton) is there
to show what the abstract must contain — rewrite it in your own voice.

**Write in this order, not top to bottom:** Method → Results → Discussion →
Introduction → Conclusion → Abstract. The introduction is easiest to write once
you know exactly what you found, and the abstract is a summary of a finished
paper, not a plan for one.

---

## Title

Three options in `RESEARCH_QUESTION.md`. Recommended:

> *Quantum models need more data, not less: a controlled test of
> generalisation-bound predictions in maritime AIS anomaly detection*

---

## Abstract — 200–250 words

Must contain, in this order: the problem, the gap, what you did, the headline
number, the qualifier, the conclusion. Structured skeleton:

> **Background.** Automatic Identification System (AIS) data is the primary
> source for maritime surveillance, but vessels can falsify or withhold it.
> **Gap.** Quantum machine learning is argued to generalise from fewer examples
> than classical models of the same size, a claim following from a generalisation
> bound of `sqrt(T/N)` in the number of trainable gates, but it is rarely tested
> against matched classical controls.
> **Method.** 114,969 ten-report windows from 8.2 million AIS reports, five
> injected attack types, vessel-disjoint splits. Three hybrid quantum neural
> networks against depth- and parameter-matched classical twins, a quantum
> kernel, and six classical baselines, across five training-set sizes from 571 to
> 68,296 windows; 1,050 training runs; paired Wilcoxon tests over 10 seeds.
> **Results.** The gap inverts with training-set size, opposite to the bound's
> popular inference: [numbers]. No quantum model beats gradient boosting at any
> size [numbers]. The quantum kernel does not collapse when data is scarce,
> locating the deficit in variational optimisation rather than in quantum feature
> maps. A geometric-difference analysis bounds any quantum-kernel advantage at
> [number] against a possible [number].
> **Conclusion.** Under a size constraint and with sufficient data, quantum
> models outperform matched classical networks; against unconstrained classical
> models they do not.

---

## 1. Introduction — ~800 words

1. **Why AIS anomaly detection matters.** Vessels disable transponders or falsify
   positions to hide sanctions evasion, illegal fishing and unsafe operation.
   Paolo et al. (2024, *Nature*) found most industrial fishing vessels are not
   publicly tracked — a concrete, citable motivation.
2. **Why machine learning, and what industry uses.** Rules, then tree ensembles
   on engineered features, then deep learning on raw tracks (`CITATIONS_ROUND3.md`).
3. **Why quantum is worth testing.** The bound of Caro et al. (2022) and the
   inference drawn from it: fewer trainable gates should mean better
   generalisation from few labels. Labelled maritime anomalies are genuinely
   scarce, so this is exactly the regime that matters operationally.
4. **The gap.** QML studies frequently lack matched classical controls; Bowles,
   Ahmed & Schuld (2024) show the field needs negative results with real baselines.
5. **This paper.** State the research question and the four sub-questions verbatim
   from `RESEARCH_QUESTION.md`.
6. **Contributions** — a five-item list:
   - a controlled comparison at five training-set sizes with depth- and
     parameter-matched classical twins, 1,050 runs;
   - evidence that the quantum-classical gap **inverts** with data size, opposite
     to the bound's popular inference, replicated across two architectures;
   - a frozen-circuit ablation isolating what trainable quantum gates contribute;
   - separation of quantum kernels from variational circuits, locating the
     scarce-data deficit in optimisation;
   - a measured ceiling on quantum-kernel advantage for this data, plus a negative
     result for quantum features inside the models industry actually uses.

**Trap:** do not promise a quantum advantage in the introduction. State the
question, not the hoped-for answer.

---

## 2. Related work — ~700 words

Four short paragraphs, each ending with what it leaves unresolved:

| Paragraph | Sources |
|---|---|
| Maritime anomaly detection: rules, statistical models, deep learning | Riveiro et al. 2018; Pallotta et al. 2013; Millefiori et al. 2016; Kroodsma et al. 2018; Nguyen et al. 2021 (GeoTrackNet); Paolo et al. 2024 |
| Why tree ensembles remain the practical baseline for tabular features | Grinsztajn et al. 2022; Shwartz-Ziv & Armon 2022 |
| Quantum ML on classical data: kernels, variational circuits, data re-uploading | Havlíček et al. 2019; Schuld & Killoran 2022; Pérez-Salinas et al. 2020; Huang et al. 2021 |
| Generalisation bounds and the benchmarking critique | Caro et al. 2022; Gil-Fuster et al. 2024; Bowles, Ahmed & Schuld 2024 |

All references are verified in `CITATIONS*.md` — use those entries, do not
re-type DOIs from memory.

---

## 3. Data and problem setup — ~800 words · Figures 1–3, Table 1

- Source, date, scale: MarineCadastre, 2023-01-01, 8,156,509 reports → 114,969
  windows (**Figure 1**, `figures/01_data_pipeline.png`).
- Windowing: 10 consecutive reports per vessel; 12 engineered features. Full
  definitions go in an appendix, not here.
- **Vessel-disjoint splits** — state the zero-overlap check explicitly. This is
  the single most important methodological detail in the paper: without it, the
  same vessel appears in train and test and every number is inflated.
- The five injected attacks, with severities derived from measured properties of
  clean data (**Figure 2**, `figures/02_attack_composition.png`). Gradual drift is
  the largest class at 37.7% of test anomalies and the hardest to detect.
- Class imbalance: 3.47% anomalous. Say here that this is why PR-AUC and an alert
  budget are reported alongside AUC.
- **Figure 3** (`figures/03_feature_separability.png`) shows what each feature can
  do alone — it justifies the feature set and previews why drift is hard.
- **Why injected rather than observed anomalies**: labelled real spoofing at this
  scale does not exist publicly. Say it here, and again in Limitations.

---

## 4. Models and experimental design — ~1,200 words · Table 2

The section that decides whether a reviewer believes the rest.

**4.1 Quantum models.** Three 8-qubit hybrid architectures: classical pre-layer
(T = 154), data re-uploading (T = 161), frozen circuit (T = 106, control). State
the forced `Dense(12→8)` pre-layer and that it is a handicap in the conservative
direction — the quantum side sees 8 compressed features, the classical twin sees
all 12.

**4.2 Matched classical twins.** Matched on both depth and parameter count, with
the classical side always holding at least as many parameters (155, 165, 111).
This is what makes SQ1 a controlled comparison rather than a demo.

**4.3 The quantum kernel.** IQP fidelity kernel (Havlíček et al. 2019), PCA
12→8, SVM on a precomputed kernel. Note the O(N²) cost, which is why it stops at
1,429 windows — itself a finding, not just a limitation.

**4.4 Classical baselines.** Gradient Boosting, Random Forest, Logistic
Regression, a 12-5-1 Multi-Layer Perceptron (71 parameters), Isolation Forest
(unsupervised), and a single-threshold rule (zero parameters). The rule and
Isolation Forest matter: they show what is achievable with no learning and no
labels at all.

**4.5 Training protocol.** Fixed **gradient-step budget** (640 steps), not fixed
epochs — explain why, since at a fixed epoch count a model trained on 571 windows
gets 120× fewer updates than one trained on 68,296, which confounds any
scarce-data comparison. Identical subsamples across models at each seed. Two
hyperparameter configurations per model, equal on both sides.

**4.6 Evaluation and statistics.** ROC-AUC, PR-AUC, precision and recall at a
3.5% alert budget, plus per-attack AUC. Paired Wilcoxon signed-rank over 10
seeds; note that 5 paired seeds cannot reach p < 0.05 (floor 0.0625) and that
this is why the headline comparisons use 10.

**4.7 Reproducibility.** 8-qubit noiseless simulation, PennyLane + JAX, pinned
versions; determinism verified at 0.00e+00 across 60 matched runs; code and every
result CSV public.

---

## 5. Results — ~1,500 words, one subsection per sub-question

Lead each subsection with the answer in one sentence, then the evidence.

**5.1 (SQ1) The gap inverts with training-set size** — **Figures 4–6**
(`05_inversion`, `10_endpoint_seeds`, `12_significance_map`), Table 3.
Report Δ and p at both endpoints for all three pairs, the 19/0 versus 0/6 split,
and the measured crossover range. Then the frozen-circuit control: freezing the
48 quantum angles changes nothing significantly at any size, so the win is not
coming from the trainable quantum parameters. Note the re-uploading exception —
significant at neither end — and what it suggests about encoding repetition.

**5.2 (SQ2) No quantum model beats the best classical model** — **Figure 7**
(`13_model_ranking`), Table 4 from `COMPLETE_RESULTS.md` §1.
Gradient Boosting 0.8316, Random Forest 0.8225, best quantum 0.8074. Show the gap
narrowing then stalling. State plainly that against Logistic Regression and the
71-parameter MLP the quantum models are level, not ahead.

**5.3 (SQ3) Quantum kernels behave differently from variational circuits** —
**Figures 8–9** (`14_quantum_kernel`, `04_learning_curves`).
The kernel beats all seven neural networks at 571 windows, then barely improves.
This is the paper's mechanistic claim: the scarce-data deficit belongs to
variational optimisation, not to quantum feature maps.

**5.4 (SQ4) A measured ceiling, and a negative result for quantum features** —
from `RESULTS_FOLLOWUP.md`; needs two new figures (see "Still to make").
Geometric difference 2.9 of 23.9 and 3.8 of 37.8. Quantum features make Random
Forest and Gradient Boosting worse and behave like classical random features.
**Report the validation/test reversal here** — it is a genuine methodological
contribution, not an embarrassment.

**5.5 Detection by attack type** — **Figure 10** (`11_attack_detection`).
Transponder switch-off +0.073 over the matched twin (p = 0.002); speed
falsification 0.998, the study's highest single score; drift unsolved at
0.61–0.66 for every model including a zero-parameter rule.

---

## 6. Discussion — ~900 words

1. **Why the inversion happens.** Optimisation, not generalisation: the kernel,
   which solves a convex problem, does not collapse; the variational models,
   which run gradient descent over circuit parameters, do. Caro's theorem is an
   *upper* bound — nothing here contradicts it; what fails is the inference drawn
   from it. Cite Gil-Fuster et al. (2024).
2. **What the size constraint means in practice.** Equal-size comparisons are the
   real constraint on satellite AIS receivers, buoys and on-vessel hardware. Say
   where it is legitimate and where it is artificial.
3. **The ceiling.** The geometric difference says the room for a quantum-kernel
   advantage on this data is roughly a tenth of what is theoretically possible,
   and narrows as data grows.
4. **Where advantage is actually proven, and why AIS is not there.** Liu et al.
   2021 (structured data), Huang et al. 2022 (quantum-native data), Huang, Kueng
   & Preskill 2021 (polynomial limits for classical data).
5. **Operational reading.** Transponder switch-off is the attack most tied to
   sanctions evasion and illegal fishing, so the strongest quantum result lands on
   the most operationally meaningful attack — under the size constraint, and still
   far behind the tree ensembles on that attack.

---

## 7. Limitations — ~500 words

The five non-claims from `RESEARCH_QUESTION.md`, plus:

- tuning budget of two configurations per model, equal on both sides but small;
- `MLP_frozen` instability (seed standard deviation 0.059);
- the test set was read twice (baselines and endpoint runs) — no selection ever
  occurred on it, but say so;
- the quantum kernel was not run at 2,857 windows and is reported as not run;
- injected anomalies; one day of US coastal AIS; 8-qubit noiseless simulation.

Write these as declarative sentences. A limitation stated plainly reads as
competence; a limitation buried reads as a flaw a reviewer found for you.

---

## 8. Future work — ~400 words

From `QUANTUM_ADVANTAGE.md` Part 2, in descending order of how cheap and
informative each is: quantum natural gradient to test the optimisation
explanation; 12+ qubits to remove the compression handicap; covariant kernels
built on the symmetries of vessel tracks (Glick et al. 2024); quantum-native
sensor data as the setting where advantage is actually proven; a regime-switching
hybrid detector.

---

## 9. Conclusion — ~200 words

Answer the research question in its conditional form. No new numbers, no new
citations, no hedging. Two or three sentences on what the result means for anyone
considering quantum models for maritime surveillance.

---

## Appendices

- **A. Feature definitions** — all 12, from `METHODS.md`.
- **B. Attack injection parameters** — from `INJECTION_AND_TIMELINE.md`.
- **C. Complete results tables** — `COMPLETE_RESULTS.md`, including the
  completeness audit.
- **D. Reproducibility** — repository link, pinned versions, run times, the
  determinism check.
- **E. Corrections log** — every claim corrected during the study, with the
  original wording. Unusual, and it is the appendix that will most convince a
  careful reader that the rest is trustworthy.

---

## Figures and tables — final list

| Paper | File | Section |
|---|---|---|
| Figure 1 | `01_data_pipeline.png` | 3 |
| Figure 2 | `02_attack_composition.png` | 3 |
| Figure 3 | `03_feature_separability.png` | 3 |
| Figure 4 | `05_inversion.png` | 5.1 |
| Figure 5 | `10_endpoint_seeds.png` | 5.1 |
| Figure 6 | `12_significance_map.png` | 5.1 |
| Figure 7 | `13_model_ranking.png` | 5.2 |
| Figure 8 | `14_quantum_kernel.png` | 5.3 |
| Figure 9 | `04_learning_curves.png` | 5.3 |
| Figure 10 | `11_attack_detection.png` | 5.5 |
| Appendix | `06`, `07`, `08`, `09` | C |

| Table | Contents | Source |
|---|---|---|
| 1 | Dataset: splits, windows, vessels, anomaly rate | `COMPLETE_RESULTS.md` / Figure 1 |
| 2 | Model inventory: family, parameters, depth, constraint | `METHODS.md` |
| 3 | Quantum minus matched twin at both endpoints, with p | `COMPLETE_RESULTS.md` §6 |
| 4 | Test AUC, every model at every size | `COMPLETE_RESULTS.md` §1 |
| 5 | Per-attack AUC at full data | `COMPLETE_RESULTS.md` §7 |

## Still to make

1. Figures for §5.4 — the quantum-features result and the geometric-difference
   ceiling currently exist only as tables.
2. Test-set versions of Figures 4 and 9 (currently validation). The data exists
   in `results/full_sweep_results.csv`; a reviewer will ask why the headline
   figures are not on the held-out set.
