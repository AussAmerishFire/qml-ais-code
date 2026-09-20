# Round 2 papers — short, applied, close to what you are actually doing

Found via arXiv full-text search (OpenAlex keyword relevance was too noisy for
these queries — it kept returning QUANTUM ESPRESSO and Majorana modes).
All free to read. Metadata below is from the arXiv record; **check for a
journal version before citing**, per the rule in `CITATIONS.md`.

---

## The two that matter most

### 1. The closest thing to your paper that exists

**Tanner, Davies, Elahi et al. (2025)** — *Maritime object classification with
SAR imagery using quantum kernel methods.* arXiv:2512.11367
<http://arxiv.org/abs/2512.11367>

> Quantum machine learning applied to a maritime surveillance problem — illegal,
> unreported and unregulated (IUU) fishing detection. **This is the nearest
> published neighbour to your study, and it does not close your gap:** they
> classify *SAR satellite imagery*, you classify *AIS trajectory behaviour*.
> Different sensor, different data type, different failure modes.
>
> It also uses **quantum kernels**, which have no trainable gates — so the Caro
> bound does not apply to their model and does to yours. That contrast is worth
> a sentence in your related-work section; it justifies your variational choice
> against a real maritime comparator rather than a hypothetical one.
>
> **Cite this in the introduction.** A reviewer who finds it after you didn't
> will assume you missed it.

### 2. The methodological template

**Fellner, Kreplin, Tovey et al. (2025)** — *Quantum vs. classical: A
comprehensive benchmark study for predicting time series with variational
quantum machine learning.* arXiv:2504.12416
<http://arxiv.org/abs/2504.12416>

> A systematic variational-QML-versus-classical benchmark on sequential data.
> Same question as yours, same model family, different task (forecasting rather
> than anomaly classification). Read this one for how they structure the
> comparison and report it — then cite it as the precedent your protocol
> follows. Pairs naturally with Bowles, Ahmed & Schuld (2024).

---

## Supporting

**de Lima, Farias, Ricardo et al. (2025)** — *Assessing the Advantages and
Limitations of Quantum Neural Networks in Regression Tasks.* arXiv:2509.00854
> An honest-assessment paper on where QNNs do and do not help. Useful in the
> discussion when you report a null or parity result — it establishes that
> "we looked carefully and found no advantage" is a normal contribution here,
> not a failed experiment.

**Liu, Chen, Chen et al. (2025)** — *Quantum-Enhanced Parameter-Efficient
Learning for Typhoon Trajectory Forecasting.* arXiv:2505.09395
> Quantum models on **trajectory** data, and explicitly framed around
> *parameter efficiency* — the same argument as your T-matched comparison.
> The closest methodological analogue outside maritime.

**Fan, Shi, Guggemos et al. (2023)** — *Hybrid Quantum-Classical Convolutional
Neural Network Model for Image Classification.* **IEEE Transactions on Neural
Networks and Learning Systems.** DOI: 10.1109/TNNLS.2023.3312170 · 120 citations
> Peer-reviewed hybrid quantum-classical architecture with a classical baseline.
> Cite when justifying the hybrid structure (classical layer + circuit) that
> your architecture screen settled on.

**Cerezo, Arrasmith, Babbush et al. (2021)** — *Variational quantum algorithms.*
**Nature Reviews Physics.** DOI: 10.1038/s42254-021-00348-9 · 3,232 citations
> The standard reference for what a variational circuit *is*. Cite once in
> methods so you don't have to define it from scratch.

**Biamonte, Wittek, Pancotti et al. (2017)** — *Quantum machine learning.*
**Nature 549.** DOI: 10.1038/nature23474 · 4,727 citations
> The canonical field-level review. One introduction citation.

---

## What the search confirms about your gap

Searching arXiv for quantum machine learning together with maritime, vessel or
ship returns **exactly one relevant paper** (Tanner et al., SAR imagery). There
is still nothing on QML for AIS trajectory anomaly detection.

The gap is real — but it is now one paper narrower than when this project
started, and that paper must be cited and distinguished.
