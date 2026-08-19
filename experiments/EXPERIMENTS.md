# ESOREX Experiments, Architectural Reference

This document defines the experiments and their measurement structure, the enzyme
systems, the two evaluation regimes, and the metrics each produces. Each experiment
produces a **measurement dictionary** used to evaluate the ESOREX specificity model.

The current specificity model is the **energetic constraint system**: rates are converted
to activation energies (`E = −RT ln k`) and the free-energy decomposition `X w = E` is solved
exactly (`esorex/energetic_specificity.py`, `esorex/constraint_system.py`). Both experiment
types map onto it directly, for LOO, each substrate is one equation; for regioselectivity, the
reactive site is a low-energy positive and the bypassed sites are high-energy inferred negatives.
Predicted energies convert back to rates via `k = e^(−E/RT)`, and each prediction carries a
determined/extrapolation verdict.

> The multi-model comparison harness (`run_all_datasets.py`, `pipeline.py`, and the
> per-variant `results/`) has been **retired**, ESOREX now ships a single specificity
> model, so the "rank several variants" framing no longer applies. The experiment
> descriptions below are kept as the record of the systems and their measurement
> structure, and their curated data remains under `data/`.
>
> Of the dataset loaders under `experiments/model_selection/datasets/`, the FucTIII
> regioselectivity loader is wired to the energetic model (`build_energetic()` /
> `build_test_energetic()`, consumed by `experiments/demonstrations/generate_fuctiii.py`).
> The TyrB and FucTIII demonstrations both run on the energetic model today
> (`experiments/demonstrations/`); the Benchmark IDs named below are descriptions of the
> measurement structure, not currently-runnable loaders (except FucTIII regioselectivity).

---

## Two experiment types

### LOO, Substrate activity prediction

We have a panel of substrates with measured rates (kcat/KM, kf/KD, or relative
activity). We train a model on the panel and ask how well it can predict activity on
held-out substrates. The standard evaluation is **leave-one-out (LOO)**: for each
substrate, train on all others, predict the held-out one.

Each substrate has its own reaction SMILES, so its features are derived from the full
mechanistic context of that reaction. The training set contains multiple positives
(substrates with measurable activity), differing in rate.

This type answers: *given that we know an enzyme accepts these substrates at these
rates, can the model infer what makes a substrate good or bad?*

### Regioselectivity, Site-preference inference from inferred negatives

We have exactly one confirmed reactive site on one natural substrate (occasionally
two). All other chemically feasible sites on that same substrate molecule are treated
as **inferred negatives**, the enzyme did not react there despite chemical
feasibility. This tiny set (1–2 positives + a handful of inferred negatives per
substrate) is the complete training signal. No reaction SMILES exist for the negatives;
they are hypothetical sites.

The model is then applied to a set of **test substrates**, molecules not in the
training data. For each test substrate, the model predicts an energy for every
chemically feasible site. We evaluate whether those predictions agree with what the
enzyme actually does on those test substrates.

This type answers: *can a model trained on one confirmed site correctly predict the
reactive site on other substrates it has never seen?*

---

## The two extreme scenarios this suite covers

The transaminase experiments represent one extreme: **many positives, no inferred
negatives**. Every substrate in the panel is an amino acid with a single reactive
amine, there are no alternative sites to use as negatives. The model must learn
purely from rate variation across substrates.

The FucTIII regioselectivity experiment represents the opposite extreme: **one (or two)
positives, many inferred negatives**. The natural acceptor has multiple chemically feasible
OH sites; only one is productive. The model learns entirely from site contrast on a single
molecule.

Both experiment types can exist for the same enzyme system and must be treated as
separate experiments with separate measurement dictionaries.

---

## Measurement dictionary

Every experiment produces a subset of the following measurements. Not all
measurements apply to every experiment type.

| Key | Applies to | Definition |
|-----|-----------|------------|
| `pearson_r` | LOO; Regioselectivity when substrate-level activity varies | Pearson r on log-scale between predicted energy and measured activity. Range [−1, 1]; 1 = perfect positive correlation. |
| `spearman_r` | Both types | Rank correlation between predicted and measured values across all test substrates. Non-parametric; more robust than Pearson for small n. |
| `top_site_accuracy` | Regioselectivity | For each test substrate, is the site with highest predicted energy the site the enzyme is known to act on? Fraction of test substrates where this holds. Range [0, 1]. |
| `rmse_log` | LOO | Root mean squared error in log space across all test substrates. |
| `n_test` | Both types | Number of test substrates contributing to the measurement. Essential context for interpreting all other numbers. |

---

## Enzyme systems and their experiments

---

### Transaminases (Onuffer & Kirsch, *Protein Sci.* 1995)

Two enzyme variants, wild-type eTATase and engineered eAATase, tested on the
same panel of 22–23 amino-acid substrates with measured kf/KD rates. Amino acids have
a single reactive amine group, so there are no alternative sites to use as inferred
negatives. **LOO only.**

#### Experiment T1: eTATase LOO

Wild-type tyrosine aminotransferase. Best substrates are aromatic amino acids
(Phe, Tyr, Trp, 4-MePhe, etc.). Rates normalized to [0, 1].

- **Type**: LOO
- **n_test**: 23
- **Measurements**: `pearson_r`, `spearman_r`, `rmse_log`
- **Benchmark ID**: `transaminases_etatase`

#### Experiment T2: eAATase LOO

Engineered aspartate aminotransferase. Best substrates are aliphatic amino acids
with carboxylate side chains (Asp, Glu), the opposite selectivity from eTATase,
despite sharing the same substrate panel. The structural features that distinguish
the best substrates are sparse relative to the many aromatic-ring features in the
panel, making this a harder and more discriminating test than T1.

- **Type**: LOO
- **n_test**: 22
- **Measurements**: `pearson_r`, `spearman_r`, `rmse_log`
- **Benchmark ID**: `transaminases_eaatase`

---

### FucTIII (Tsai et al., *ACS Catalysis* 2019)

Helicobacter pylori α1-3/4-fucosyltransferase. Transfers fucose from GDP-Fuc to
the 3-OH of GlcNAc in lacto-series glycan acceptors. Acceptor substrates are
oligosaccharides with multiple OH sites. Both experiment types apply.

#### Experiment F1: FucTIII LOO

Six substrates with measured kcat/KM. LOO substrate activity prediction. Features
are derived from each substrate's full reaction context. Note that 3 of 6 substrates
have nearly identical high activities; `spearman_r` is the primary metric.

- **Type**: LOO
- **n_test**: 6
- **Measurements**: `spearman_r`, `rmse_log`
- **Benchmark ID**: `fuctiii_loo`

#### Experiment F2: FucTIII regioselectivity

Training signal: GlcNAc 3-OH in LacNAc and GlcNAc 3-OH in LNB are the two
confirmed positives. All other OH sites on LacNAc and LNB are inferred negatives
(12 total). Training set: 2 positives + 12 inferred negatives.

Test substrates: the 21-compound panel from the paper, for which kcat/KM is measured
and, for synthetic compounds, whether fucosylation was achieved. Evaluation compares
the highest-predicted site per compound against the known reactive site, and predicted
max-site energies against measured kcat/KM.

- **Type**: Regioselectivity
- **Training set**: 2 positives + 12 inferred negatives
- **n_test**: up to 21 (those with parseable SMILES and a known reactive site)
- **Measurements**: `top_site_accuracy`, `spearman_r`
- **Benchmark ID**: `fuctiii_regioselectivity`
