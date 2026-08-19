# ESOREX

**ESOREX** (Enzymatic Specificity Operator Reaction EXclusion) models enzyme substrate
specificity as **free energy**. Given a small set of measured reactions for one enzyme, it
learns what the active site rewards, exactly interpolates the training measurements, and, for any
candidate that passes the mechanistic filter, assigns an energetic prediction tagged as either
**determined** by the data or an **extrapolation** beyond them.

## How it works, in one screen

ESOREX separates specificity into two stages:

```
candidate compound
      │
      ▼
1. MECHANISTIC FEASIBILITY   does the reaction chemistry even apply here?   (EVODEX pre-filter)
      │  passes
      ▼
2. ENERGETIC SPECIFICITY     how fast, and how sure are we?
      │
      └─▶  E = E₀ + Σ wᵢxᵢ  (+ cooperativity)      predicted rate + provenance
```

The first stage rejects compounds that do not match the enzyme's learned reaction pattern. The
second maps each comparable measured rate onto a free-energy scale (`E = −RT ln k`, so *differences*
in `E` are activation-free-energy differences, [see the model page](model.md#from-fragments-to-features))
and decomposes it into contributions from chemical features, following Jencks's account of binding
energy. The weights are solved as a **hard constraint** that exactly interpolates the training
measurements, and the model reports whether each new prediction is pinned down by the data or is a
point estimate in extrapolation.

## A guided path

Read the three concept pages in order; each builds on the one before.

1. **[Theory: the mechanistic pre-filter](theory.md)**: how ESOREX decides a reaction is
   feasible, and how it splits a substrate into the reactive core and the *passenger* that
   specificity acts on. (Specificity-focused readers can skim this and move on.)
2. **[Representation: the carbon-only ensemble](representation.md)**: how a substrate becomes a
   set of physical features (proton count, oxidation state, charge, lone pairs, π character), as
   deltas hung on a carbon skeleton and addressed at three positional scales (atom → subdomain →
   passenger), with the rigid-body subdomain layer giving the length-independent middle abstraction
   and a most-active-first alignment fixing the exact atom coordinate across molecules.
3. **[Model: energetic specificity](model.md)**: the free-energy decomposition, the constraint
   system that reproduces the data exactly, the identifiability collapse, and how predictions
   carry provenance.

Then see it work:

- **[Demonstrations](demonstrations.md)**: worked examples on real enzymes, with commentary on
  where the model succeeds and where it fails.

Reference:

- **[Experiments](../experiments/EXPERIMENTS.md)**: the enzyme systems, evaluation regimes, and
  metrics.
- **[Repository README](../README.md)**: installation, inputs, and repository layout.

---

*ESOREX docs · **Home** · [Theory](theory.md) · [Representation](representation.md) · [Model](model.md) · [Demonstrations](demonstrations.md)*
