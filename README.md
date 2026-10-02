# Quantum machine learning for AIS maritime anomaly detection

**Under what conditions do quantum machine learning models outperform classical
models in detecting anomalous vessel behaviour in AIS data?**

That is the study's research question, with four sub-questions and the evidence
answering each set out in [`docs/RESEARCH_QUESTION.md`](docs/RESEARCH_QUESTION.md).

The study started from Caro et al. (2022, *Nature Communications*), who bound the
generalisation error of a quantum model with `T` trainable gates by `sqrt(T/N)`.
The inference people draw is that quantum models should hold up better than
classical models when training data is scarce. **This study tested that directly,
against depth- and parameter-matched classical twins, and found the opposite.**

**Status: data collection complete.** 1,222 recorded model fits across 13
model families, five training-set sizes and two follow-up experiments. Every
number in every document is computed from the result CSVs at build time. Nothing
is extrapolated.

<sub>Model fits by source: scarcity sweep and tuning grid 230 · baseline sweep 100 · 10-seed endpoints 120 · full sweep 240 (60 more reused from the endpoints, not re-run) · quantum kernel 10 · quantum-features experiment 500 · full-data baselines 22.</sub>

---

## What was found

**1. The quantum-versus-classical gap inverts with data size.** Hybrid Quantum
Neural Networks (HQNNs) are *worse* than matched classical networks when data is
scarce and *better* when it is plentiful — the opposite of the bound's popular
inference. At full data: 19 significant quantum wins, 0 losses. At 571 training
windows: 0 wins, 6 losses. Measured crossover: 2,400–9,400 windows.

**2. No quantum model beats the best classical model at any size.** Gradient
Boosting (test AUC 0.8316) and Random Forest (0.8225) lead everywhere; the best
quantum model reaches 0.8074. The quantum advantage in finding 1 exists only
under an equal-size, equal-depth, equal-training-budget constraint.

**3. Where quantum did win.** Transponder switch-off detection (+0.073 over the
matched twin, p = 0.002), PR-AUC (+0.086, p = 0.004), and speed falsification,
where the data re-uploading HQNN scores 0.998 — the highest of any model in the
study. The Quantum Support Vector Machine is the only quantum model that does not
collapse with scarce data, beating all seven neural networks at N = 571.

**4. Two follow-ups closed off two hypotheses.** Quantum features do not improve
Random Forest or Gradient Boosting: they lower test AUC and are never better than
the best classical random-feature control built from the same inputs. The geometric difference of Huang et al. (2021) puts a
measured ceiling on any quantum-kernel advantage here: 2.9 at N = 571 against a
maximum possible 23.9.

Full numbers: **[`docs/COMPLETE_RESULTS.md`](docs/COMPLETE_RESULTS.md)** (every model at
every size) and **[`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md)** (the argument).

---

## Repository layout

```
README.md            you are here
requirements.txt     exact version pins (they matter — see Reproducing)

docs/                21 documents, indexed in docs/README.md
                       FINAL_REPORT.md      the findings
                       COMPLETE_RESULTS.md  every number, auto-generated
                       QUANTUM_ADVANTAGE.md where quantum won / could win
                       RESULTS_FOLLOWUP.md  the two follow-up experiments
                       METHODS.md, DECISIONS.md, CITATIONS*.md, ...

scripts/             every script, numbered in run order (PIPELINE.py first)

data/                dataset_train/val/test.csv   the built dataset
                     windows_*.csv               pipeline intermediates
                     old8_*.csv                  superseded 8-feature version
                     raw/                        raw AIS downloads (not committed)

results/             every measured run, as CSV or JSON — the experimental record
                     kernels/                    saved quantum kernel matrices

figures/             21 figures, explained in figures/README.md
                     earlier/                    superseded exploration figures

logs/                run logs, exactly as written during each run

paper/               the paper in LaTeX (IEEE two-column): main.tex, refs.bib,
                     generated tables, and the 15 figures it uses
```

**Run everything from the repository root**, so that relative paths resolve:

```bash
.\.venv\Scripts\python.exe scripts/24_figures.py
```

---

## Start here

| Read this | For |
|---|---|
| **[`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md)** | The findings, the paired tests, the limitations. Start here. |
| **[`docs/COMPLETE_RESULTS.md`](docs/COMPLETE_RESULTS.md)** | Every model × every size, test and validation, with a completeness audit |
| **[`docs/QUANTUM_ADVANTAGE.md`](docs/QUANTUM_ADVANTAGE.md)** | Where quantum measurably won, kept strictly apart from where it *could* win |
| **[`docs/RESULTS_FOLLOWUP.md`](docs/RESULTS_FOLLOWUP.md)** | Quantum features in Random Forest; the geometric-difference ceiling |
| **[`paper/`](paper/)** | The paper itself: LaTeX source, ready for Overleaf |
| **[`figures/`](figures/)** | 21 figures, with a README explaining each |
| **[`docs/THE_DOCUMENT.md`](docs/THE_DOCUMENT.md)** | Narrative walkthrough from raw data onward, with a glossary |
| **[`docs/METHODS.md`](docs/METHODS.md)**, [`docs/DECISIONS.md`](docs/DECISIONS.md), [`docs/SENSITIVITY.md`](docs/SENSITIVITY.md) | Methods prose, every judgement call, threshold sweeps |
| **[`docs/CITATIONS.md`](docs/CITATIONS.md)**, [`docs/CITATIONS_ROUND2.md`](docs/CITATIONS_ROUND2.md), [`docs/CITATIONS_ROUND3.md`](docs/CITATIONS_ROUND3.md) | References, each verified against Crossref/OpenAlex rather than recalled |
| [`docs/BASELINES.md`](docs/BASELINES.md), [`docs/QNN_RESULTS.md`](docs/QNN_RESULTS.md), [`docs/RESULTS_SWEEP.md`](docs/RESULTS_SWEEP.md) | Earlier stage reports, kept for provenance |

Two files are superseded and marked at the top: `docs/RESULTS_TABLES.md` and
`docs/WHERE_QUANTUM_WINS.md`. They are kept only so the record is complete.

---

## The code, in run order

| Script | Does |
|---|---|
| `scripts/PIPELINE.py` | 8.2M raw AIS reports → 114,969 windows, 12 features, 5 injected attacks, vessel-disjoint splits |
| `scripts/07_validate_dataset.py` | 42 property assertions on the built dataset |
| `scripts/09_independent_check.py` | 22 checks recomputing features by a separate code path |
| `scripts/08_sensitivity_sweep.py` | 9 pipeline rebuilds sweeping the three main thresholds |
| `scripts/10_classical_baselines.py`, `scripts/11_baseline_viz.py` | Classical baselines on the full data, and their figures |
| `scripts/16_sweep.py` | The scarcity sweep: quantum models and matched twins across five sizes |
| `scripts/19_final.py` | Endpoint runs at 10 seeds for the paired significance tests |
| `scripts/20_qsvm.py` | Quantum Support Vector Machine (IQP fidelity kernel) |
| `scripts/21_baseline_sweep.py`, `scripts/22_full_sweep.py` | Every model at every size, validation and test |
| `scripts/23_build_tables.py` | Builds `docs/COMPLETE_RESULTS.md` from the CSVs |
| `scripts/24_figures.py` | Writes all 14 figures |
| `scripts/25_geometric_difference.py` | Huang et al. (2021) test: can any quantum kernel win here? |
| `scripts/26_quantum_features_rf.py` | Quantum features inside Random Forest / Gradient Boosting |
| `scripts/27_paper_tables.py` | Writes every table in the paper as LaTeX, from the result CSVs |
| `scripts/28_paper_figures.py` | Writes figures 15–21: method overview, circuits, attack examples, test-set curves, follow-ups |

Result files (`*_results.csv`, `*_selection.json`) are committed: they are the
experimental record, not build artefacts.

---

## Reproducing

```bash
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**The version pins matter.** PennyLane 0.45.1 calls `jax.core.is_concrete`,
removed in JAX 0.11 — any newer JAX makes every quantum model raise
`AttributeError` at the first circuit evaluation. See `requirements.txt`.

**The raw AIS data is not in this repository.** `data/raw/AIS_2023_01_01.csv` is 836 MB and
the zone downloads are larger still, all over GitHub's 100 MB per-file limit.
Download from [MarineCadastre](https://marinecadastre.gov/ais/) (US federal data,
public domain), date 2023-01-01, then run `scripts/PIPELINE.py` to regenerate everything
downstream. The built datasets (`dataset_train/val/test.csv`) *are* committed, so
every experiment can be rerun without the raw file.

Runs are checkpointed per result row and resume if interrupted; the long ones
request Windows sleep prevention while they work.

---

## How the numbers are kept honest

- Vessel-disjoint splits: zero MMSI overlap between train, validation and test.
- Classical twins matched on both depth and parameter count, with the classical
  side always holding at least as many parameters.
- A frozen-circuit control isolating what the trainable quantum gates contribute.
- Paired Wilcoxon signed-rank tests at 10 seeds wherever a claim depends on one.
- Determinism verified: two independent pipelines agreed to 0.00e+00 on 60
  matched runs.
- Corrections are recorded in place rather than quietly edited. Several claims in
  earlier drafts were wrong and are marked as corrected where they appeared,
  including one in the published summary page.

**Known gap:** the Quantum Support Vector Machine was never run at N = 2,857
(about 30 hours of compute). It is reported as not run, never estimated.
