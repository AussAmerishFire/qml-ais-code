# Project Plan — plain English

You are writing the paper, so every step below says three things:
**what it means**, **why it's needed**, and **what it gives you to write about**.

---

## Where we are now

| Done | What it produced |
|---|---|
| Downloaded 1 day of US AIS data | 8.16M position reports, 15,932 vessels |
| Filtered out junk and moored vessels | 1.26M reports, 2,441 vessels still moving |
| Built "windows" — 10 consecutive reports each | 122,287 windows |
| Engineered 13 features, cut to 6 by redundancy | the 6 that go into the quantum circuit |
| 8 baseline figures | what normal maritime behaviour looks like |
| First attempt at injecting fake anomalies | works, but has a labelling bug |

---

## STEP 1 — Fix the labels ⚠️ BLOCKING

**What it means.** Right now, when we corrupt part of a vessel's voyage, we mark
*the entire voyage* as suspicious. A voyage becomes ~20 windows, but the fake
anomaly only touches one or two of them. So about 85–90% of the windows we've
labelled "anomalous" are actually completely normal.

**Analogy.** Like labelling an entire 8-hour flight as "hijacked" because two
minutes of it were. The model would be trained on mostly-wrong answers.

**Why it's needed.** Every accuracy number we produce would be meaningless
until this is fixed. It's not a small error.

**What it gives the paper.** Nothing directly — it just makes everything after
it trustworthy.

---

## STEP 2 — Decide about the "went silent" feature

**What it means.** We built a feature measuring how long a vessel was quiet
before a voyage started, to catch "going dark". But we inject our fake silences
*inside* voyages, never at the start — so the feature never sees anything. It
currently shows 4629 for normal windows and 4470 for anomalous. No difference.

**Two options.**
- **Drop it** — five features instead of six. Simple, honest.
- **Add a fifth attack type** that deletes a whole chunk of voyage, creating a
  genuine multi-hour disappearance. More work, but "going dark" is the anomaly
  your own Govia system ranks as Critical, and right now we cannot detect it.

**Recommendation:** add the attack. It is the most operationally important
anomaly type in maritime security, and a reviewer familiar with the domain will
ask why it's missing.

**What it gives the paper.** Either a cleaner 5-feature story, or a 5th attack
type that makes your taxonomy match real maritime practice.

---

## STEP 3 — Squash the lopsided features

**What it means.** Most of our features have thousands of tiny values and a few
enormous ones. Taking the logarithm compresses the big ones and spreads the
small ones out.

**Why it's needed.** This is specific to quantum encoding. Each feature becomes
a rotation angle between 0° and 180°. If 99% of values crowd into the first 2°,
almost every window produces nearly the same angle — and that qubit ends up
carrying almost no information. Log-transforming spreads them across the full
range so the qubit is actually used.

**What it gives the paper.** A methods sentence, and it prevents a result where
the quantum model underperforms for a boring technical reason rather than a
real one.

---

## STEP 4 — Handle the missing heading data

**What it means.** 19% of windows have no valid heading, because the AIS
standard sends "511" when the sensor isn't reporting. So `crab_mean`
(the difference between where a ship points and where it actually goes) is
missing for a fifth of the data.

**Two options.**
- Fill missing values with the median, and add a flag column saying "this was
  missing" — the flag itself might be informative, since a vessel that stops
  reporting heading mid-voyage is doing *something*.
- Drop the feature entirely.

**Recommendation:** fill and flag. Throwing away a fifth of your data, or a
whole feature, is a bigger cost than carrying one extra column.

**What it gives the paper.** A limitations sentence, honestly stated.

---

## STEP 5 — Convert everything to angles

**What it means.** Rescale all features so they run from 0 to π (0° to 180°),
which is what the quantum circuit accepts as input.

**Why it's needed.** Purely mechanical. The circuit cannot take a speed in knots;
it takes a rotation angle.

**Important:** the scaling must be computed on the *training* data only, then
applied to the test data. Computing it on everything leaks information about
the test set into training — a subtle and very common mistake.

**What it gives the paper.** One methods sentence, plus you avoid a leakage
error a careful reviewer would catch.

---

## STEP 6 — Split train/test BY VESSEL ⚠️ IMPORTANT

**What it means.** When dividing data into "train the model on this" and "test
it on that", all windows from a given ship must go entirely to one side.

**Why it's needed.** If the same ship appears in both, the model can memorise
*that specific vessel's habits* rather than learning what suspicious behaviour
looks like in general. Your accuracy would look great and mean nothing.

**Analogy.** Like testing a student on the exact questions they revised. High
score, no evidence of understanding.

**What it gives the paper.** This is one of the things your abandoned audit was
going to count other papers for getting wrong. Getting it right — and saying so
explicitly — puts your methodology above much of the field.

---

## STEP 7 — Check visually that injection worked

**What it means.** Regenerate three figures (box plots, PCA, t-SNE) with
anomalies coloured red.

**Why it's needed.** A one-minute check that answers "are these anomalies even
detectable?" before spending hours training models. If red points sit invisibly
inside the normal blob, the anomalies are too subtle and we adjust *now*.

**What it gives the paper.** A strong before/after figure pair, and evidence you
validated the setup rather than trusting it.

---

## STEP 8 — Build the classical model first

**What it means.** Train ordinary models — logistic regression, random forest, a
small neural network — on the same features.

**Why it's needed.** You cannot claim quantum is better without knowing what
classical achieves. And the classical model must be **properly tuned**, or the
comparison is rigged. This is the single most common way QML papers manufacture
a fake advantage.

**What it gives the paper.** Your baseline table, and the credibility that comes
from having tuned the thing you are trying to beat.

---

## STEP 9 — Build the quantum model

**What it means.** The hybrid quantum neural network: a small classical layer,
then a 6-qubit circuit, then an output layer. Matched in parameter count to the
classical network so the comparison is fair.

**What it gives the paper.** Your method section and architecture diagram.

---

## STEP 10 — The main experiment (this IS the paper)

**What it means.** Train both models repeatedly while shrinking the number of
labelled anomalies they get to learn from: 500, then 200, 100, 50, 20. Five
random seeds at each level so you can report error bars.

**Why it's needed.** This tests your actual research question. The theory
(Caro et al.) says quantum models should hold up better when labels are scarce.
This is where you find out.

**Three possible outcomes, all publishable:**
- Quantum degrades more slowly → theory confirmed in a real domain
- Both degrade identically → theory doesn't transfer, honestly reported
- Quantum degrades faster → also a real finding

**What it gives the paper.** Your headline figure — two curves, performance
against label count — and your entire results section.

---

## STEP 11 — Robustness checks

**What it means.** Re-run the pipeline with different values for the three
thresholds I picked arbitrarily (0.5 kn "moving", 30-min track split, 10-report
windows) and show the conclusions don't change.

**Why it's needed.** Those numbers are currently my judgment with nothing behind
them. A reviewer will ask. "We tested 0.3, 0.5 and 1.0 and the result held" is a
complete answer.

**What it gives the paper.** An appendix table that closes off the most obvious
line of attack.

---

## Rough timing

| Steps | Work |
|---|---|
| 1–6 (pipeline fixes) | one pass, then re-run |
| 7 (visual check) | minutes |
| 8 (classical baseline) | fast — these models train in seconds |
| 9 (quantum model) | build and test on a small subset first |
| 10 (main experiment) | **50 training runs — likely an overnight job** |
| 11 (robustness) | re-runs of 1–10 at different settings |

Step 10 is the expensive one. We should time a single quantum training run
before committing to the full matrix, so we size it against real numbers rather
than guesses.
