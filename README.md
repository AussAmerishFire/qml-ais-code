# Quantum machine learning for AIS maritime anomaly detection

Do quantum machine-learning models detect anomalous vessel behaviour in AIS data
better than classical models — and if so, under exactly what conditions?

The study started from Caro et al. (2022, *Nature Communications*), who bound the
generalisation error of a quantum model with `T` trainable gates by `sqrt(T/N)`.
The inference people draw is that quantum models should hold up better than
classical models when training data is scarce. **This study tested that directly,
against depth- and parameter-matched classical twins, and found the opposite.**

**Status: data collection complete.** 1,050 measured training runs across 13
model families, five training-set sizes and two follow-up experiments. Every
number in every document is computed from the result CSVs at build time. Nothing
is extrapolated.

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
Random Forest or Gradient Boosting — they behave like classical random features
and lower test AUC. The geometric difference of Huang et al. (2021) puts a
measured ceiling on any quantum-kernel advantage here: 2.9 at N = 571 against a
maximum possible 23.9.

Full numbers: **[`COMPLETE_RESULTS.md`](COMPLETE_RESULTS.md)** (every model at
every size) and **[`FINAL_REPORT.md`](FINAL_REPORT.md)** (the argument).

---

## Start here

| Read this | For |
|---|---|
| **[`FINAL_REPORT.md`](FINAL_REPORT.md)** | The findings, the paired tests, the limitations. Start here. |
| **[`COMPLETE_RESULTS.md`](COMPLETE_RESULTS.md)** | Every model × every size, test and validation, with a completeness audit |
| **[`QUANTUM_ADVANTAGE.md`](QUANTUM_ADVANTAGE.md)** | Where quantum measurably won, kept strictly apart from where it *could* win |
| **[`RESULTS_FOLLOWUP.md`](RESULTS_FOLLOWUP.md)** | Quantum features in Random Forest; the geometric-difference ceiling |
| **[`figures/`](figures/)** | 14 figures, with a README explaining each |
| **[`THE_DOCUMENT.md`](THE_DOCUMENT.md)** | Narrative walkthrough from raw data onward, with a glossary |
| **[`METHODS.md`](METHODS.md)**, [`DECISIONS.md`](DECISIONS.md), [`SENSITIVITY.md`](SENSITIVITY.md) | Methods prose, every judgement call, threshold sweeps |
| **[`CITATIONS.md`](CITATIONS.md)**, [`CITATIONS_ROUND2.md`](CITATIONS_ROUND2.md), [`CITATIONS_ROUND3.md`](CITATIONS_ROUND3.md) | References, each verified against Crossref/OpenAlex rather than recalled |
| [`BASELINES.md`](BASELINES.md), [`QNN_RESULTS.md`](QNN_RESULTS.md), [`RESULTS_SWEEP.md`](RESULTS_SWEEP.md) | Earlier stage reports, kept for provenance |

Two files are superseded and marked at the top: `RESULTS_TABLES.md` and
`WHERE_QUANTUM_WINS.md`. They are kept only so the record is complete.

---

## The code, in run order

| Script | Does |
|---|---|
| `PIPELINE.py` | 8.2M raw AIS reports → 114,969 windows, 12 features, 5 injected attacks, vessel-disjoint splits |
| `07_validate_dataset.py` | 42 property assertions on the built dataset |
| `09_independent_check.py` | 22 checks recomputing features by a separate code path |
| `08_sensitivity_sweep.py` | 9 pipeline rebuilds sweeping the three main thresholds |
| `10_classical_baselines.py`, `11_baseline_viz.py` | Classical baselines on the full data, and their figures |
| `16_sweep.py` | The scarcity sweep: quantum models and matched twins across five sizes |
| `19_final.py` | Endpoint runs at 10 seeds for the paired significance tests |
| `20_qsvm.py` | Quantum Support Vector Machine (IQP fidelity kernel) |
| `21_baseline_sweep.py`, `22_full_sweep.py` | Every model at every size, validation and test |
| `23_build_tables.py` | Builds `COMPLETE_RESULTS.md` from the CSVs |
| `24_figures.py` | Writes all 14 figures |
| `25_geometric_difference.py` | Huang et al. (2021) test: can any quantum kernel win here? |
| `26_quantum_features_rf.py` | Quantum features inside Random Forest / Gradient Boosting |

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

**The raw AIS data is not in this repository.** `AIS_2023_01_01.csv` is 836 MB and
the zone downloads are larger still, all over GitHub's 100 MB per-file limit.
Download from [MarineCadastre](https://marinecadastre.gov/ais/) (US federal data,
public domain), date 2023-01-01, then run `PIPELINE.py` to regenerate everything
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
