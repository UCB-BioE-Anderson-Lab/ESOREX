# ESOREX

**ESOREX** (Enzymatic Specificity Operator Reaction EXclusion) models enzyme substrate
specificity as a **free-energy problem**. [EVODEX](https://github.com/UCB-BioE-Anderson-Lab/EVODEX)
provides a deterministic mechanistic feasibility filter; within the feasible set, ESOREX converts
measured rates to activation energies (`E = −RT ln k`), decomposes that energy into contributions
from physically-grounded chemical features, and learns those contributions as a **hard constraint**
that reproduces the training rates exactly. Given a small set of labeled reactions for one enzyme,
it predicts a rate for any candidate compound, and, crucially, reports whether that prediction is
**determined** by the data or is an **extrapolation** beyond them.

> **Beta release.** In the cases tested so far, ESOREX reproduces the training rates exactly, ranks
> held-out substrates well, and flags the predictions it cannot support rather than guessing
> confidently. Predictions are intended for experimental prioritization, not as confirmed substrate
> annotations. The API and data formats are still stabilizing. Feedback welcome via GitHub Issues.

**Documentation:** start at the [ESOREX docs home](docs/index.md) for a guided path through the
theory, representation, model, and demonstrations. This README covers installation and the
repository itself.

---

## Quickstart

Run ESOREX on the TyrB dataset in your browser, no install required:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UCB-BioE-Anderson-Lab/ESOREX/blob/main/notebooks/tyrb_quickstart.ipynb)

The [`notebooks/tyrb_quickstart.ipynb`](notebooks/tyrb_quickstart.ipynb) notebook trains the
energetic model on the 9 natural amino-acid substrates of *E. coli* TyrB and predicts 14 held-out
unnatural analogs, reporting for each whether the prediction is **determined** by the data or an
**extrapolation** beyond it. It runs in about a minute on a free Colab CPU.

For a local install instead, see [Installation](#installation) below.

---

## Worked examples

Two end-to-end vignettes walk through the model on real enzymes, the data, the representation, what the model learns about the active site, and how it does:

→ [Demonstrations](docs/demonstrations.md), [FucTIII](docs/demonstrations/fuctiii_regioselectivity.md) (regioselectivity: picking the right hydroxyl) and [TyrB](docs/demonstrations/tyrb_extrapolation.md) (substrate specificity: predicting novel analogs, and what happens when a surprising one is fed back in).

---

## Pipeline

Given atom-mapped training reactions and measured activity values, ESOREX:

1. **Builds a mechanistic tree** from the training reactions to identify the reactive center
   and derive the EVODEX operator for that reaction type.
2. **Reduces each substrate to its carbon skeleton** and describes its chemistry as physical
   deltas hung on the carbons (proton count, oxidation state, charge, lone pairs, π character),
   then **aligns every training substrate's skeleton into one shared frame** (the alignment that
   minimizes electron edit distance) so a delta is addressed by where it sits on that common
   scaffold.
3. **Trains the energetic model**, converts rates to activation energies (`E = −RT ln k`) and
   solves the free-energy decomposition `X w = E` as a hard constraint, reproducing every training
   rate exactly and keeping the full space of solutions the data leave underdetermined.
4. **Screens candidates** through the EVODEX mechanistic pre-filter, rejecting compounds that
   lack the electronic prerequisites for the reaction.
5. **Predicts a rate** for each passing compound, with a provenance: whether the prediction is
   determined by the training data or an extrapolation beyond it.

**Inputs:** pre-mapped reaction SMILES with measured rates (kcat/KM, kf/KD, or relative rates).

**Output:** a predicted rate per candidate, each tagged determined or extrapolation, plus a novelty
score measuring how far outside the training support the prediction reaches.

A typical use case: 1 to 20 measured substrates for one enzyme, screened against thousands
of metabolites. At EVODEX D-level this narrows a few thousand candidate metabolites to hundreds
before specificity scoring, and the specificity model further narrows these to tens or fewer.
With as few as one substrate, the regioselectivity path can generate predictions by inferring
negatives from unproductive reaction sites on the same molecule.

---

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/UCB-BioE-Anderson-Lab/ESOREX
cd ESOREX
pip install -r requirements.txt
pip install -e .
```

Key dependencies:

| Package | Version | Role |
|---------|---------|------|
| [`evodex`](https://github.com/UCB-BioE-Anderson-Lab/EVODEX) | ==2.1.0 | Chemical abstraction operators |
| `CGRtools` | ==4.0.41 | Reaction graph manipulation |
| `rdkit` | latest | Molecule handling and fingerprints |
| `scikit-learn` | latest | Numerical utilities |
| `numpy` | <2 | Numerical arrays |
| `rxnmapper` | latest | One option for atom-mapping reaction SMILES |

## Running tests

```bash
python -m pytest tests/
```

---

## Inputs and assumptions

### Required inputs

- **Pre-mapped reaction SMILES** for each training substrate. Atom-map numbers must identify
  the reactive center. Atom mapping is a prerequisite, not handled by ESOREX; RXNMapper and
  the EVODEX operator library are two options, but this is treated as a separate upstream step.
- **Measured rates** for each training substrate, raw kinetic rates (kcat/KM, kf/KD, or
  relative rates). ESOREX converts them to activation energies internally (`E = −RT ln k`);
  see [the model](docs/model.md#from-fragments-to-features).
- **One model per enzyme/reaction type.** ESOREX trains a separate model for each reaction.

### Not handled internally

- Atom mapping from scratch
- Tautomer canonicalization
- Protonation-state assignment
- Salt stripping and resonance normalization
- Multi-step reaction cascades

ESOREX assumes inputs have been standardized before use. A tautomer or protonation-state
mismatch between training SMILES and query SMILES will appear as a spurious physical difference
in the feature set, producing incorrect predictions.

---

## Terminology

| Term | Definition |
|------|-----------|
| **Mechanistic tree** | Representation of the reaction center and bond changes, derived from the mapped reaction and the EVODEX operator. Defines which atoms are "the chemistry." |
| **Reactive center** | Atoms directly involved in bond changes (broken or formed bonds). Identified by atom-map numbers in the mapped reaction SMILES. The narrowest definition of "what reacts." |
| **Reactive domain** | The set of substrate atoms that match the EVODEX operator at the abstraction level in use. At B-level this equals the reactive center; at C/D/E levels it expands outward (sigma shell, pi shell, extended sigma shell). The reactive domain is what the EVODEX pre-filter tests for, and it is removed before building the passenger domain. |
| **Passenger domain** | The molecular fragment(s) left after removing the reactive domain. These are the parts of the substrate the enzyme accommodates but that do not participate in, or immediately surround, the bond changes. The passenger domain shrinks as the EVODEX abstraction level increases. |
| **Carbon skeleton** | The saturated carbon-only graph of a substrate, its defined energetic zero state. The substrate's chemistry is described as physical deltas hung on it, and every training substrate's skeleton is aligned into one shared frame so positions are commensurable across molecules. |
| **EVODEX operator** | Mechanistic pre-filter: tests whether a candidate compound contains the electronic prerequisites for the reaction to occur at all, at a specified abstraction level. |
| **Constraint system** | The linear system `X w = E` (substrate features × free-energy weights = activation energies), solved exactly rather than fit; the object that stores rank, consistency, null space, and cooperative terms. |
| **Specificity model** | The ESOREX energetic model: an additive free-energy decomposition `E = E₀ + Σ wᵢxᵢ`, reproduced exactly on training and predicting new rates with an explicit determined/extrapolation verdict. |
| **Determined prediction** | A prediction invariant over every exact-fitting weight vector (`q·N = 0`), pinned by the data even when individual weights are not. Everything else is flagged as extrapolation. |

---

## How it works

ESOREX separates the problem of enzyme specificity into two distinct stages:

1. **Mechanistic feasibility**, does the compound have the right electronic configuration for the reaction to occur at all? This is handled by [EVODEX](https://github.com/UCB-BioE-Anderson-Lab/EVODEX), which represents the reactive center and its electronic neighborhood (sigma and pi shells) at five abstraction levels. ESOREX uses EVODEX operators as a pre-filter, rejecting compounds that lack the required electronic configuration before scoring begins.

2. **Substrate specificity**, among compounds that pass the mechanistic filter, which does the enzyme actually accept, and how fast? This is learned from the training data. ESOREX converts each measured rate to an activation energy (`E = −RT ln k`) and decomposes it as a baseline plus feature contributions, `E = E₀ + Σ wᵢxᵢ`. The weights are solved as a **hard constraint**, the model must reproduce every training rate exactly, and the whole space of solutions the data leave underdetermined is kept, so a new prediction can be reported as **determined** or flagged as **extrapolation**.

Each substrate is reduced to its **carbon skeleton** and described by physical deltas hung on the carbons, proton count, oxidation state, formal charge, lone pairs, π character. Every training substrate's skeleton is then **aligned into one shared frame** (the alignment that minimizes electron edit distance, grown from the most active substrate outward), so a delta is addressed by where its carbon sits on that common scaffold and the same position means the same thing across molecules. No functional groups are named; recognition motifs emerge as combinations of these primitives, and an identifiability collapse keeps only the distinctions the training measurements can support.

→ [Theory: EVODEX and the mechanistic pre-filter](docs/theory.md)
→ [Representation: the carbon-only ensemble](docs/representation.md)
→ [Model: energetic specificity](docs/model.md)

---

## Demonstrations

Worked examples illustrating what ESOREX does, mechanistic trees, specificity models, and how predictions extrapolate to novel compounds.

→ [Demonstrations and vignettes](docs/demonstrations.md)

---

## Known limitations

- **Additivity with entangled features.** The model is additive in free energy (with cooperative terms added only where the data force them). When two determinants co-occur in every training substrate, in TyrB, large side chains are always aromatic, their weights cannot be separated, and a test substrate that presents one determinant without the other is exactly where the additive model mis-credits the combination. The model flags such predictions as extrapolation rather than trusting them.

- **Representation resolution.** Two substrates the featurizer cannot distinguish cannot be assigned different energies by any model. Exact reproduction of the training set therefore depends on the representation resolving every distinct substrate; where it does not, ESOREX reports the limit rather than papering over it.

- **Stereochemistry:** configuration (R/S, E/Z) is retained in the substrate↔reference mapping but is not yet a feature; which stereochemical distinctions become features, and at what resolution, is a deliberate open decision. As with any feature, a configuration difference matters only when the training set contains substrates that differ in exactly that way.

- **Input standardization:** tautomers, salts, resonance, and protonation state must be standardized before use; a mismatch between training and query forms appears as a spurious physical difference and produces incorrect predictions.

- **Sparse regions extrapolate poorly**, but visibly. A region the training barely covers (few aliphatic substrates, one ring position) yields coarse or undetermined predictions, and the model marks them as such. The regioselectivity path enriches sparse data by inferring negatives from unproductive sites on the same molecule.

- **One model per reaction type.** ESOREX does not handle multi-step cascades or multi-enzyme predictions within a single model.

---

## Repository structure

```
esorex/                               # Core package (pip-installable)
    # --- mechanistic layer (shared by all models) ---
    mechanistic_tree.py               # MechanisticTree dataclass
    generate_mechanistic_tree.py      # Build the mechanistic tree from a bag of reactions
    visualize_mechanistic_tree.py     # Render mechanistic tree as SVG
    label_reaction.py                 # Atom-map a reaction against an EVODEX operator
    label_substrate.py                # Match EVODEX operators to candidate molecules
    reaction_preparation.py           # Normalize a mapped reaction SMILES
    # --- energetic specificity model ---
    carbon_tree.py                    # Carbon-only ensemble: per-passenger skeleton + deltas,
                                      #   electron-edit-distance alignment into one shared frame
    carbon_featurize.py               # Delta features from the aligned ensemble, fit/transform
    hydrogenated_reference.py         # Carbon reference + lossless reconstruction check
    reference_collapse.py             # Identifiability collapse to the supported representation
    constraint_system.py              # X w = E: rank, null space, cooperative terms, determinism
    energetic_specificity.py          # End-to-end train/predict with provenance
    free_energy.py                    # Rate ↔ energy helpers

experiments/
    demonstrations/                   # Worked-example generators (TyrB, FucTIII) on the energetic model
    model_selection/
        datasets/                     # FucTIII regioselectivity loader (energetic model)
    transaminases/                    # Output directory for the TyrB demonstration report

data/
    transaminases/                    # TyrB (ONUFFER, lee-peng) reaction and activity data
    fucosyltransferases/              # FucTIII reaction data

assets/
    demonstrations/                   # TyrB and FucTIII figures used in the docs
    readme/                           # Figures referenced by the docs and README

tests/
    unit_tests/                       # Unit tests for core modules
    integration_tests/                # End-to-end tests for the mechanistic tree pipeline
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Citation

If you use ESOREX in your research, please cite:

> Anderson, J.C. *ESOREX: Enzymatic Specificity Operator Reaction EXclusion.*
> Beta release v0.1.0-beta. https://github.com/UCB-BioE-Anderson-Lab/ESOREX (2026).
