# Quantum machine learning for AIS maritime anomaly detection

Does a hybrid quantum neural network detect vessel-trajectory anomalies better
than a parameter-matched classical model **when labelled anomalies are scarce**?

The question comes from Caro et al. (2022, *Nature Communications*), who bound
the generalisation error of a quantum model with `T` trainable gates as
`sqrt(T/N)`. If that bound bites, a quantum model should degrade more slowly
than a classical one of the same size as the training set `N` shrinks.

**Status: experiments in progress. No result is claimed yet.** Current state,
with numbers, is in `QNN_RESULTS.md`.

---

## Start here

| Read this | For |
|---|---|
| **`THE_DOCUMENT.md`** | Everything, from raw data to now, with a glossary. Start here if you are new. |
| **`BASELINES.md`** | The five classical baselines, why each exists, and the full test-set results |
| **`QNN_RESULTS.md`** | Quantum architectures, the ablation controls, and where the study stands |
| **`FIGURES_BASELINE.md`** | What each of the eight baseline figures shows and which belong in the paper |
| **`CITATIONS.md`** + **`CITATIONS_ROUND2.md`** | Verified references. Metadata checked against Crossref/OpenAlex, not recalled. |
| `METHODS.md`, `DECISIONS.md`, `SENSITIVITY.md` | Methods prose, every judgement call, threshold sweeps |

---

## Reproducing

```bash
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**The version pins matter.** PennyLane 0.45.1 calls `jax.core.is_concrete`,
removed in JAX 0.11 — install any newer JAX and every quantum model raises
`AttributeError` at the first circuit evaluation. See `requirements.txt`.

The raw AIS file (837 MB) is **not** in the repository. Download
`AIS_2023_01_01.csv` from [MarineCadastre](https://marinecadastre.gov/ais/)
(US federal data, public domain) into the repository root, then:

```bash
.\.venv\Scripts\python.exe PIPELINE.py            # raw -> dataset_{train,val,test}.csv
.\.venv\Scripts\python.exe 07_validate_dataset.py # 42 property assertions
.\.venv\Scripts\python.exe 09_independent_check.py# 20 checks, independent code path
```

The committed `dataset_*.csv` files are the exact data behind every reported
number, so the experiments below run without the raw download.

---

## Pipeline

| Script | Does |
|---|---|
| `PIPELINE.py` | **The consolidated pipeline.** Raw AIS to modelling-ready splits, every constant derived inline with its justification. |
| `07_validate_dataset.py` | 42 property assertions — leakage, label correctness, physical sanity |
| `09_independent_check.py` | 20 checks that **recompute features with different code** (explicit loops, spherical law of cosines instead of haversine). This is what found the float32 coordinate bug. |
| `08_sensitivity_sweep.py` | 9 full rebuilds sweeping the three main thresholds |
| `10_classical_baselines.py` | Five classical baselines, tuned on validation, test read once |
| `11_baseline_viz.py` | The eight baseline figures |
| `13_architectures.py` | Six quantum architectures, each with a parameter-matched classical twin |
| `14_controls.py` | Ablations: frozen gates, circuit removed, depth-matched classical |
| `15_qnn_confusion.py` | Validation confusion matrices |
| `16_sweep.py` | **The headline experiment** — hyperparameter selection then the label-scarcity sweep |
| `17_feature_probe.py` | Tests whether four new trend/direction features break the ~0.85 ceiling |

Numbering is chronological. `PIPELINE.py` supersedes `01_`–`05_`, which are kept
because `DECISIONS.md` refers to them.

---

## Long runs, and not losing them

`16_sweep.py` takes many hours. Two protections:

**Checkpointing.** Every completed run is appended to `sweep_results.csv`
immediately. On restart the script reads that file and skips finished work, so
a crash, reboot or power cut costs at most one model fit.

```powershell
.\resume.ps1     # safe any time; resumes, refuses to start a second copy
```

**Sleep prevention.** The script calls `SetThreadExecutionState` on start. This
is process-scoped and reverts automatically on exit — it does not change your
power settings, and the display still sleeps.

This exists because of a measurement: one configuration took 21,133 s while the
*same* configuration on a different seed took 1,255 s. Compute does not vary 17×
on a seed — the machine was asleep and the wall clock was not. That single stall
consumed 6 of one night's 13 hours.

---

## Ground rules the experiments follow

These are enforced in code, not just intended. Full statement in
`BASELINES.md` §3.

1. **Hyperparameters selected on validation only.** Test is read once, at the end.
2. **Equal tuning budgets.** Quantum and classical get the same number of
   configurations — the failure mode Bowles, Ahmed & Schuld (2024) identified.
3. **Parameter matching.** Each quantum model's classical twin is the smallest
   MLP with *at least* as many trainable parameters, so a quantum win is never a
   capacity win.
4. **Never accuracy.** At a 3.5% anomaly rate a model that flags nothing scores
   96.5%. ROC-AUC and PR-AUC only.
5. **5 seeds, and paired tests for any difference claim.** A two-seed screen
   already produced one false positive in this project (`QNN_RESULTS.md` §3b).
6. **Equal gradient-step budgets across the sweep**, not equal epochs. Fixed
   epochs would have given N=571 forty updates against 640 at full data, and the
   resulting undertraining would have imitated the generalisation effect under
   test.

---

## Data

NOAA Office for Coastal Management, *AIS Vessel Traffic Data*,
[MarineCadastre.gov](https://marinecadastre.gov/ais/), file `AIS_2023_01_01.csv`,
accessed 2026-08-27. US federal data, public domain — the processed subset may
be redistributed with the paper.

8,156,509 raw reports → 1,256,434 rows from 2,441 vessels → 114,969 windows,
3.47% anomalous, split vessel-wise with zero MMSI overlap between splits.

Anomalies are **injected**, not observed: five attack types (`speed_lie`,
`drift`, `jump`, `gap`, `dark`) with severities derived from measured properties
of the clean data. `INJECTION_AND_TIMELINE.md` has the full derivation.

---

## Honest state of the work

- No quantum architecture beats a properly-tuned classical baseline at full
  training data. Best quantum 0.8357, tuned MLP 0.8441, random forest 0.8659.
  This is **expected** — at N = 68,296, `sqrt(T/N)` ≈ 0.03, so generalisation is
  not the binding constraint for either model.
- Freezing the quantum gates costs ~0.009 AUC against seed noise near 0.02. The
  trainable rotation angles are close to decorative at full data. Whether that
  changes as N shrinks is what `16_sweep.py` is measuring.
- `drift` — 37.7% of test anomalies — is detected at ~0.60 by every model,
  including a zero-parameter threshold rule. `17_feature_probe.py` tests a
  diagnosis: the bearing actually travelled is computed at `PIPELINE.py:232` and
  then discarded, so nothing measures reported course against real course.
