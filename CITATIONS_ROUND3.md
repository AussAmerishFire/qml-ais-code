# Citations, round 3 — how AIS machine learning is done in practice

Every entry verified against OpenAlex on 2026-09-18 (title, authors, venue, DOI).
Commercial companies (Windward, Kpler/MarineTraffic, Spire Maritime, S&P Global,
Lloyd's List Intelligence) do not publish their model internals. Do **not**
write "company X uses model Y" in the paper. Cite the published methods below
instead.

## Rules and behaviour indicators (the explainable first layer)

- **Welch, Clavelle, White et al. (2022).** *Hot spots of unseen fishing vessels.*
  Science Advances. DOI 10.1126/sciadv.abq2109 · 147 cites.
  Detecting AIS disabling ("going dark") events at global scale.
- **Riveiro, Pallotta & Vespe (2018).** *Maritime anomaly detection: A review.*
  WIREs Data Mining and Knowledge Discovery. DOI 10.1002/widm.1266 · 163 cites.
  Taxonomy of approaches: rule-based, statistical, machine learning.

## Statistical models of normal traffic

- **Pallotta, Vespe & Bryan (2013).** *Vessel pattern knowledge discovery from AIS
  data: a framework for anomaly detection and route prediction (TREAD).* Entropy.
  DOI 10.3390/e15062218 · 690 cites. Unsupervised route clustering.
- **Millefiori, Braca, Bryan et al. (2016).** *Modeling vessel kinematics using a
  stochastic mean-reverting process for long-term prediction.* IEEE Trans.
  Aerospace and Electronic Systems. DOI 10.1109/TAES.2016.150596 · 118 cites.

## Tabular machine learning on engineered features (this study's setting)

- **Grinsztajn, Oyallon & Varoquaux (2022).** *Why do tree-based models still
  outperform deep learning on typical tabular data?* NeurIPS 2022 Datasets and
  Benchmarks. DOI 10.52202/068431-0037 · 585 cites.
  Explains why Gradient Boosting and Random Forest top this study's rankings.
- **Shwartz-Ziv & Armon (2022).** *Tabular data: Deep learning is not all you
  need.* Information Fusion. DOI 10.1016/j.inffus.2021.11.011 · 115 cites.

## Deep learning on raw tracks and imagery

- **Kroodsma, Mayorga, Hochberg et al. (2018).** *Tracking the global footprint of
  fisheries.* Science. DOI 10.1126/science.aao5646 · 1,099 cites.
  Convolutional neural networks on AIS tracks (Global Fishing Watch).
- **Nguyen, Vadaine, Hajduch et al. (2021).** *GeoTrackNet — a maritime anomaly
  detector using probabilistic neural network representation of AIS tracks and a
  contrario detection.* IEEE Trans. Intelligent Transportation Systems.
  DOI 10.1109/TITS.2021.3055614 · 129 cites. Variational recurrent network.
- **Nguyen & Fablet (2021).** *TrAISformer — a generative transformer for AIS
  trajectory prediction.* arXiv. (No DOI; 27 cites.)
- **Paolo, Kroodsma, Raynor et al. (2024).** *Satellite mapping reveals extensive
  industrial activity at sea.* Nature. DOI 10.1038/s41586-023-06825-8 · 252 cites.
  Deep learning on satellite imagery matched against AIS: most industrial
  fishing vessels are not publicly tracked.

## Quantum counterpart for sequence models (future work)

- **Chen, Yoo & Fang (2022).** *Quantum long short-term memory.* ICASSP 2022.
  DOI 10.1109/ICASSP43922.2022.9747369 · 236 cites.

## Quantum decision trees and forests

- **Srikumar, Hill & Hollenberg (2024).** *A kernel-based quantum random forest for
  improved classification.* Quantum Machine Intelligence.
  DOI 10.1007/s42484-023-00131-2 · 27 cites. The main quantum random forest:
  each split is made by a quantum-kernel classifier instead of a threshold.
- **Lu & Braunstein (2014; online 2013).** *Quantum decision tree classifier.*
  Quantum Information Processing. DOI 10.1007/s11128-013-0687-5 · 123 cites.
- **Heese, Bickert & Niederle et al. (2022).** *Representation of binary
  classification trees with binary features by quantum circuits.* Quantum.
  DOI 10.22331/q-2022-03-30-676 · 23 cites.
- **Khadiev, Mannapov & Safina (2019).** *The quantum version of classification
  decision tree constructing algorithm C5.0.* arXiv.
  DOI 10.48550/arXiv.1907.06840 · 18 cites. Quantum speed-up for *building* trees.
- **Kumar, Yalovetzky, Li et al. (2023).** *Des-q: a quantum algorithm to construct
  and efficiently retrain decision trees.* arXiv. Only the data/code record was
  verified (Zenodo DOI 10.5281/zenodo.8428525); check the arXiv entry before citing.
