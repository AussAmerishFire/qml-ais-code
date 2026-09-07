# References

All metadata verified against Semantic Scholar and Crossref on 2026-08-27 —
authors, years, venues and DOIs are checked, not recalled.

**Cite the journal version, not the arXiv preprint**, wherever one exists.
Several of these are in Nature, Nature Communications, PRX Quantum and Physical
Review A; citing the preprint instead looks careless and understates the source.

---

## TIER 1 — load-bearing. Your argument depends on these.

### The hypothesis

**Caro et al. (2022)** — *Generalization in quantum machine learning from few
training data.* **Nature Communications 13.**
DOI: [10.1038/s41467-022-32550-3](https://doi.org/10.1038/s41467-022-32550-3) · 587 citations

> Proves the generalization error of a QML model with T trainable gates scales
> as √(T/N). **This is the entire basis of RQ2.** Note the wording: it is stated
> for *variationally optimizing a parameterized quantum circuit* with *trainable
> gates* — which is why the study uses a variational QNN and not a quantum
> kernel, whose circuit has no trainable parameters.

```bibtex
@article{caro2022generalization,
  title   = {Generalization in quantum machine learning from few training data},
  author  = {Caro, Matthias C. and Huang, Hsin-Yuan and Cerezo, M. and Sharma, Kunal
             and Sornborger, Andrew and Cincio, Lukasz and Coles, Patrick J.},
  journal = {Nature Communications}, volume = {13}, year = {2022},
  doi     = {10.1038/s41467-022-32550-3}
}
```

### The direct precedent

**Naaman et al. (2025)** — *Quantum Enhanced Anomaly Detection for ADS-B Data
Using Hybrid Deep Learning.* **AIAA/IEEE 44th Digital Avionics Systems
Conference (DASC), 1–9.**
DOI: [10.1109/DASC66011.2025.11257365](https://doi.org/10.1109/DASC66011.2025.11257365)

> The same study for aircraft surveillance data. Establishes the method
> (hybrid QNN, injected anomalies, public dataset) and reports **parity, not
> advantage** — H-FQNN 90.17–94.05% vs classical FNN 91.50–93.37%. Your maritime
> result is directly comparable. **Note: this is a peer-reviewed conference
> paper, not merely a preprint.**

```bibtex
@inproceedings{naaman2025quantum,
  title     = {Quantum Enhanced Anomaly Detection for {ADS-B} Data Using Hybrid Deep Learning},
  author    = {Naaman, Rani and Magalh{\~a}es, F. and Ouattara, Jean-Yves and others},
  booktitle = {2025 AIAA DATC/IEEE 44th Digital Avionics Systems Conference (DASC)},
  pages     = {1--9}, year = {2025},
  doi       = {10.1109/DASC66011.2025.11257365}
}
```

### Honest framing of quantum advantage

**Schuld & Killoran (2022)** — *Is Quantum Advantage the Right Goal for Quantum
Machine Learning?* **PRX Quantum 3, 030101.**
DOI: [10.1103/PRXQuantum.3.030101](https://doi.org/10.1103/PRXQuantum.3.030101) · 229 citations

> Cite this when framing your contribution. It argues the field's obsession with
> "beating classical" is the wrong lens. Protects you from the reviewer who asks
> why you aren't claiming speedup.

### Benchmarking rigour

**Bowles, Ahmed & Schuld (2024)** — *Better than classical? The subtle art of
benchmarking quantum machine learning models.* arXiv:2403.07059 · 197 citations

> Found that QML models do not beat properly-tuned classical baselines. **Cite
> when justifying why your classical baseline is tuned on a validation set** —
> this is the paper that shows what goes wrong otherwise.

### Encoding

**Schuld, Sweke & Meyer (2021)** — *Effect of data encoding on the expressive
power of variational quantum-machine-learning models.* **Physical Review A 103,
032430.**
DOI: [10.1103/PhysRevA.103.032430](https://doi.org/10.1103/PhysRevA.103.032430) · 895 citations

> Why the log-transform-then-scale step matters. Encoding determines what
> functions the model can represent at all.

---

## TIER 2 — supporting

**Corli et al. (2025)** — *Quantum machine learning algorithms for anomaly
detection: A review.* **Future Generation Computer Systems 166, 107632.**
DOI: [10.1016/j.future.2024.107632](https://doi.org/10.1016/j.future.2024.107632) · 56 citations
> Your related-work section for quantum anomaly detection.

**Tscharke et al. (2025)** — *Quantum Autoencoder for Multivariate Time Series
Anomaly Detection.* **IEEE QCE 2025, 2470–2481.**
DOI: [10.1109/QCE65121.2025.00268](https://doi.org/10.1109/QCE65121.2025.00268)
> The closest methodological alternative — unsupervised, reconstruction-based.
> Cite when explaining why you chose supervised classification instead.

**Simeone (2022)** — *An Introduction to Quantum Machine Learning for
Engineers.* **Foundations and Trends in Signal Processing.** arXiv:2205.09510
> Cite for standard definitions of variational circuits and angle encoding.

**Park et al. (2026)** — *AIS-LLM: A Unified Framework for Maritime Trajectory
Prediction, Anomaly Detection, and Collision Risk Assessment.* **IEEE Access 14,
106874–106891.**
DOI: [10.1109/ACCESS.2026.3705215](https://doi.org/10.1109/ACCESS.2026.3705215)
> State of the art on the *classical* side of your exact problem. Essential for
> positioning — shows what maritime anomaly detection achieves without quantum.

---

## TIER 3 — context and caution

**Havlíček et al. (2019)** — *Supervised learning with quantum-enhanced feature
spaces.* **Nature 567, 209–212.**
DOI: [10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2) · 2,863 citations
> The quantum kernel paper. Cite when justifying why you chose a variational
> model instead — kernels have no trainable gates, so Caro's bound doesn't apply.

**Larocca et al. (2025)** — *Barren plateaus in variational quantum computing.*
**Nature Reviews Physics 7, 174–189.**
DOI: [10.1038/s42254-025-00813-9](https://doi.org/10.1038/s42254-025-00813-9) · 585 citations
> Trainability limits. Relevant because your circuit is small (8 qubits), where
> barren plateaus are not yet a practical problem — worth saying explicitly.

**Cerezo et al. (2025)** — *Does provable absence of barren plateaus imply
classical simulability?* **Nature Communications 16.**
DOI: [10.1038/s41467-025-63099-6](https://doi.org/10.1038/s41467-025-63099-6)
> The sharpest current critique of variational QML. Cite in limitations — it
> argues that circuits which are trainable are often classically simulable.

**Kerenidis & Prakash (2017)** — *Quantum Recommendation Systems.* **ITCS 2017,
49:1–49:21.**
DOI: [10.4230/LIPIcs.ITCS.2017.49](https://doi.org/10.4230/LIPIcs.ITCS.2017.49) · 455 citations

**Tang (2019)** — *A quantum-inspired classical algorithm for recommendation
systems.* **STOC 2019.**
DOI: [10.1145/3313276.3316310](https://doi.org/10.1145/3313276.3316310) · 493 citations

> Cite these two **together**. Kerenidis & Prakash claimed exponential speedup;
> Tang dequantized it. The canonical cautionary tale about QML advantage claims,
> and a graceful way to show you understand the field's history.

---

## DATA

**NOAA Office for Coastal Management** — *Automatic Identification System (AIS)
Vessel Traffic Data.* MarineCadastre.gov.
File: `AIS_2023_01_01.csv`. Accessed 2026-08-27.
https://marinecadastre.gov/ais/

> US federal data, public domain — the processed subset may be redistributed
> with the paper. State this in a Data Availability section.

```bibtex
@misc{noaa_ais_2023,
  title  = {Automatic Identification System ({AIS}) Vessel Traffic Data},
  author = {{NOAA Office for Coastal Management}},
  year   = {2023}, howpublished = {MarineCadastre.gov},
  url    = {https://marinecadastre.gov/ais/}, note = {Accessed 2026-08-27}
}
```

---

## SOFTWARE — cite these; reviewers increasingly expect it

**PennyLane** — Bergholm et al. (2018), *PennyLane: Automatic differentiation of
hybrid quantum-classical computations.* arXiv:1811.04968. **Version 0.45.1.**

**scikit-learn** — Pedregosa et al. (2011), *Scikit-learn: Machine Learning in
Python.* JMLR 12, 2825–2830. **Version 1.9.0.**

**NumPy** — Harris et al. (2020), *Array programming with NumPy.* **Nature 585,
357–362.** DOI: 10.1038/s41586-020-2649-2. **Version 2.5.2.**

**pandas** — McKinney (2010), *Data Structures for Statistical Computing in
Python.* SciPy 2010. **Version 3.0.5.**

**Matplotlib** — Hunter (2007), *Matplotlib: A 2D Graphics Environment.*
CiSE 9(3), 90–95. **Version 3.11.1.**

Python 3.13.

---

## INDUSTRY SOURCES — use carefully

The anomaly taxonomy (position falsification, implausible route, transponder
switch-off, false data broadcast) comes from **Windward**, **MarineTraffic** and
**SkyFi** knowledge-base articles.

**These are commercial vendor publications, not peer-reviewed.** Cite them for
the *taxonomy* — which attack types occur in practice — and say so explicitly.
Do **not** cite them for magnitudes; they give none, which is precisely why
every severity bound in this study is derived from measured properties of the
data instead.

- Windward, *The Dark Side of AIS: Staying Ahead in a Sea of Spoofing.*
- MarineTraffic, *Understanding AIS & GNSS Spoofing.*
- SkyFi, *AIS Spoofing Explained: How to Identify Dark Ships.*

---

## WHAT YOU CANNOT CITE

**The Govia Systems ruleset.** The thresholds (STS ≤100 m for ≥45 min; going
dark >30 min; route deviation >1.5 nm) were team-decided, not derived from
research. They can be described as prior work by the author, but they are **not
evidence** for any threshold in this paper.

Interesting corroboration only: the Govia team independently chose >30 minutes
for "going dark", and this study independently measured that **0.000% of
legitimate Class A intervals reach 1800 s**. Two groups landing on the same
scale is suggestive — it is not a citation.

---

## Citation hygiene

1. **Journal over preprint.** Caro et al. is Nature Communications, not
   arXiv:2111.05292.
2. **Check the year.** Several of these were preprinted years before publication
   — Caro is 2021 on arXiv, 2022 in Nature Communications.
3. **Naaman et al. is peer-reviewed** (DASC 2025). Don't cite it as a preprint.
4. **Cite software versions.** They are recorded above and in `METHODS.md`.
5. **Kerenidis & Tang travel together.** Citing one without the other misleads.
