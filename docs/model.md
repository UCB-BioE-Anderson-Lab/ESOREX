# Model: energetic specificity

ESOREX explains an enzyme's substrate specificity by decomposing activation free energy into a
sum of contributions from chemical features. The decomposition, and the discipline about what it
can and cannot claim, follow Jencks's analysis of how binding energy is attributed to the parts of
a molecule.¹

## The attribution problem

Consider a molecule A–B that binds a protein strongly, while its isolated parts A and B each bind
weakly. How much of the binding energy belongs to A, and how much to B? Naive subtraction is
incoherent. A binds weakly on its own, so one attributes the energy to B; but B also binds weakly,
so the same argument attributes it to A. The contribution assigned to a group would depend only on
which comparison one happened to make.

Jencks resolves this by separating the **observed** binding free energy of a fragment from its
**intrinsic binding energy**: the favorable interaction a group provides when it is properly
presented to its site, without charging that group for the translational, rotational, and
conformational costs of forming the complex. Observed fragment binding underestimates the intrinsic
contribution, because those costs are paid once for a connected molecule but over again for separate
fragments. He writes the observed binding of A–B as

```
ΔG°_AB = ΔG^i_A + ΔG^i_B + ΔG^s
```

the intrinsic contributions of A and B, less a **connection Gibbs energy** `ΔG^s` that collects the
cost of localizing and organizing the molecule. To a first approximation the intrinsic
contributions are additive; the connection term is paid once, not per group.

## From fragments to features

ESOREX applies the same reasoning to a finer decomposition. A feature is a chemical identity at an
address relative to the reaction center,

```
f_i = (c_i, r_i)
```

where `c_i` is the electronic character (a lone pair, a π system, an oxidized center) and `r_i` is
its position (see [representation](representation.md)). Each feature carries an **effective
coefficient** `w_i = w(c_i, r_i)`: the net energetic increment associated with that chemical identity
at that address across the dataset. It plays the role Jencks's intrinsic binding energy plays in the
additive approximation, but it is an effective, dataset-relative quantity, not a microscopic
interaction energy (see [Effective coefficients](#effective-coefficients-not-microscopic-energies)).

The observable is a rate, and ESOREX maps **comparable** rate measurements onto a free-energy scale,
`E_S = −RT ln k_S`. Through transition-state theory a first-order rate constant satisfies
`ΔG^‡ = RT ln(k_B T/h) − RT ln k`, so `−RT ln k` equals the activation free energy only up to the
common transition-state-theory prefactor; that additive constant, like the arbitrary energy zero and
the common uncatalyzed term, is absorbed into `E₀`, and only *differences* in `E` between substrates
are physical. This requires the measurements to be commensurable, the same observable (`k_cat`,
`k_cat/K_M`, or a consistent relative activity) under comparable conditions; mixing kinds of rate
breaks the free-energy reading. With an indicator `x_i` for the presence of a feature, the
first-order model is Jencks's additive approximation extended from two fragments to a feature set:

```
E_S = E₀ + Σ_i w_i x_{S,i}
```

`E₀` is the reference energy: everything that does not vary with the encoded features across the
dataset. A common organizational cost of the kind Jencks folds into `ΔG^s` contributes to it, but
`E₀` should **not** be identified with `ΔG^s`. It is whatever is invariant within the dataset,
which also includes the common uncatalyzed term and the chosen energy zero.

## Departures from additivity

Additivity can fail: the energetic consequence of one feature can depend on which others are present.
Jencks accounts for this with an interaction, or "coupling," term `ΔG_12`. In the feature model, when
features `i` and `j` together differ from the sum of their individual effects, a pairwise coefficient
`w_{ij}` carries the difference, and it contributes only when both are present:

```
E_S = E₀ + Σ_i w_i x_{S,i} + Σ_{i<j} w_{ij} x_{S,i} x_{S,j}
      (reference energy) + (individual feature effects) + (pairwise nonadditivity)
```

`w_{ij}` equals the double-mutant-cycle quantity `E_{ij} − E_i − E_j + E₀` **only** when those four
states form the clean factorial comparison with all other features held constant. In a constraint
system with many correlated features a fitted pair coefficient need not equal an experimentally
isolated double-mutant cycle; it is whatever nonadditivity the whole system attributes to `i` and `j`
jointly. So `w_{ij}` is a cooperativity term in the thermodynamic sense only: a nonzero value says the
joint effect differs from the additive prediction, favorable or antagonistic, without asserting a
molecular mechanism or a measured coupling between features `i` and `j`.

## Effective coefficients, not microscopic energies

A weight is an **effective** free-energy coefficient, the net consequence of a feature relative to
the reference, not the physical interaction energy of a chemical group. Adding a feature can change
direct interactions, solvation, conformation, and entropy at once; a total energy identifies only
their sum, `w_i = ΔE_i`, never the separate parts, and the same holds for `w_{ij}`. This is Jencks's
warning about thermodynamic parameters: in aqueous protein systems, compensating enthalpy and entropy
changes make the microscopic origin unrecoverable from total energies. ESOREX therefore decomposes
energy by what **changes in the representation**, not by microscopic mechanism. That is a deliberately
modest claim, and an exact one.

## The constraint system, not a fit

Every training substrate supplies one equation. Stacked together they form a linear system

```
X w = E
```

with one row per substrate and one column per feature (plus the intercept). ESOREX treats this as a
**constraint to satisfy exactly**, never as a least-squares objective to approximate. That is why
over-granular features are a signal to change the representation rather than a reason to regularize.

Exactness here means one specific thing: the model **interpolates the supplied numbers exactly**. It
does *not* mean it has recovered the true underlying kinetic relationship. The supplied rates are
experimental measurements and carry noise; ESOREX treats each as exact, so if two substrates differ
only by measurement error the model will still reproduce that difference (introducing a cooperative
term if it must, per the next section). Exact interpolation is thus a property of the fit to the data
as given, not a claim of physical fidelity; it is a defensible stance only because the same framing
makes the *unsupported* directions explicit (below) rather than smoothing them away. Replicate
uncertainties are not currently propagated into the fit or the provenance.

The constraint system (`esorex/constraint_system.py`) computes three things about `X w = E`:

- **rank**: how many independent parameter combinations the substrates actually distinguish;
- **consistency**: whether an exact solution exists at all (is `E` in the column space of `X`);
- **the null space** `N`: the directions in weight space that leave every training prediction
  unchanged.

Rather than choosing one weight vector, it keeps the **whole solution space**, `w = w_p + N z`: a
particular solution `w_p` (the **minimum-norm least-squares solution**, `np.linalg.lstsq`, exact when
the system is consistent) plus arbitrary movement along the null space. Two different situations
both appear as a large null space, and the model keeps them distinct: the parameters can be
underdetermined while a *prediction* is still pinned down (see [Prediction](#prediction-and-provenance)).

## Reaching exactness with cooperative terms

When the first-order model cannot reproduce the training energies (no weight assignment satisfies every
equation), ESOREX adds cooperative terms `w_{ij}` from the previous section until the system is exactly
consistent (`restore_exactness`): each iteration finds one remaining conflict and adds one product
column that distinguishes it, repeating until none remain. "Minimal" here means **greedy** in exactly
this sense, one pair per unresolved conflict, generated from the observed conflicts rather than by
enumerating all products. It is *not* a globally minimum-cardinality set, and a different conflict
ordering can yield a different set. Because several alternative interaction sets can restore exactness,
a reported `w_{ij}` is one bookkeeping choice for the nonadditivity, not evidence of a specific
coupling between `i` and `j`, the microscopic-interpretation warning above applies with full force.

When a conflict has **no** distinguishing feature pair, because the conflicting substrates have
*identical* feature vectors but different energies, the algorithm refuses rather than inventing a term.
That is not a model failure but a **featurization** limit: no interaction of these features can separate
two substrates the featurizer cannot tell apart, and the refusal is the signal to sharpen the
representation.

## The identifiability collapse

The raw representation deliberately over-describes each substrate, emitting every physical feature at
several positional and chemical abstraction levels. This is a candidate *lattice*, not a final feature
set, and fed directly to the constraint system it would be massive double-counting: the same physical
effect appears as many perfectly correlated columns.

The **collapse** (`esorex/reference_collapse.py`) selects the experimentally-supported representation:

1. partition the candidate features by their **presence pattern across the training substrates**,
   *within* a physical channel (a size feature is never merged with an electronic one merely because
   they coincide on a small training set);
2. features in one partition are experimentally indistinguishable (no experiment here separates their
   weights), so they become **one** energetic variable;
3. drop partitions present in every substrate or none: they discriminate nothing and are absorbed by
   `E₀`;
4. **name each surviving variable by its most abstract member** (passenger over subdomain over atom).
   The label is only a name: a query activates the variable iff it shares *any* member of the
   partition, so behaviour depends on the member set, not the representative.

Two separate things happen here, and they are not the same criterion, so both must be stated.
**Which distinctions survive** is set by the presence patterns (steps 1–3): a partition is the finest
grouping the training data can actually tell apart, so nothing finer than the data support is kept and
nothing coarser is forced. **How each surviving distinction is named** is set by the abstraction rule
(step 4): among the experimentally indistinguishable members of a partition, the most abstract is
chosen, because it is the most transferable reading of an equivalence the data cannot refine. So
"keeps the finest distinction the data pin down" describes the *partitioning*, and "the most abstract
representation that still reproduces the rates exactly" describes the *labeling*; together they are
the guiding principle, **keep every distinction the data resolve, at the most abstract level
consistent with resolving it**.

## Prediction and provenance

To predict a new substrate `Q`, ESOREX builds its feature vector `q` (through the same collapse) and
forms `E_Q = q · w`. Because `w` is known only up to the null space, the prediction is a single
committed number **only when it is invariant over the whole solution space**:

```
q · N = 0   ⟹   every exact-fitting model agrees on E_Q   (the prediction is "determined")
```

This is the central rule: **parameter underdetermination is not prediction underdetermination.** If
training establishes only `w_A + w_B = −3` and a query needs exactly `A + B`, its contribution is
determined (`−3`) even though `w_A` and `w_B` individually are not.

**What number is reported when the prediction is *not* determined.** When `q · N ≠ 0` the
exact-fitting weightings disagree on `E_Q`, so no unique value exists, and what ESOREX displays is a
**point estimate chosen by convention, not a determined value**. The convention is fixed and explicit:

```
E_Q = q · w_p ,   w_p = the minimum-norm particular solution
```

Because that `w_p` lies in the row space (orthogonal to `N`), `q · w_p` keeps only the component of
the query the training data support and sets every unsupported (null-space) direction to `ΔE = 0`.
The displayed number is therefore *the supported part of the prediction with the unresolved part
zeroed*, and it must never be read as a value the data determine. Two shapes of "not determined" are
distinguished, both special cases of this convention:

- **Novel increment.** A query feature with no training precedent at any abstraction level; its
  contribution is `ΔE = 0`. This is conservative bookkeeping, not a claim the physical effect is zero.
- **Unresolved combination.** The query needs to split a known combination the data leave
  underdetermined; the split is not invented, so the supported aggregate is used and the rest zeroed.

The amount zeroed is quantified by a **novelty** score, defined as the fraction of the query vector
lying outside the training row space,

```
novelty = ‖q · N‖ / ‖q‖        (0 = fully determined; → 1 = mostly unprecedented)
```

Every prediction (`esorex/energetic_specificity.py`) therefore carries a provenance: whether it is
determined, the displayed minimum-norm point estimate, and the novelty score. On the TyrB held-out
set novelty correlated with error (the accurate predictions were the low-novelty ones), but that is
an observation on a **single** system with a fixed novelty definition, not a calibrated cross-enzyme
validation; treat it as a qualitative honesty signal, not a guaranteed error bar.

## What the constraint framing buys

- **Exact training reproduction.** Every known rate is reproduced to numerical precision, a fact the
  model never trades away.
- **Honest extrapolation.** Predictions carry a determined/undetermined verdict and a novelty score, so
  the boundary between constrained knowledge and extrapolation is explicit.
- **Graceful augmentation.** Adding a measurement, even a surprising one that contradicts the model's
  prior extrapolation, adds a constraint. The system absorbs it exactly and the existing predictions,
  each anchored by its own measurement, barely move. New data makes the model more complete without
  eroding what it already knew. The [TyrB demonstration](demonstrations/tyrb_extrapolation.md) shows this
  directly.

## Where it is weaker

- **Additivity with entangled features.** When two determinants co-occur in every training substrate,
  their weights cannot be separated. A test substrate that presents one without the other is exactly
  where the additive model over-credits the parts, and the prediction is flagged undetermined. This is
  the honest place for a cooperative term, once the data to fit one exist. The
  [TyrB demonstration](demonstrations/tyrb_extrapolation.md) shows a worked case.
- **Sparse regions.** A region the training barely covers (few aliphatic substrates, one ring position)
  yields coarse or undetermined predictions, correctly flagged rather than confidently wrong.
- **Representation resolution.** Two substrates the featurizer cannot distinguish cannot be given
  different energies by any model. Exactness then depends on the [representation](representation.md), not
  on the learner.

---

¹ Jencks, W. P. "On the attribution and additivity of binding energies." *Proc. Natl. Acad. Sci. USA*
**78**, 4046–4050 (1981). The paper distinguishes the *intrinsic* binding energy of a group from the
*observed* binding of a fragment, shows that intrinsic contributions are additive to first order while
a connection term `ΔG^s` is paid once, introduces the interaction term `ΔG_12` for the failures of
additivity, and cautions that such coefficients are effective thermodynamic quantities, not uniquely
resolvable into microscopic energies. ESOREX is that framework applied to an addressed, abstracted
feature representation.

---

← [Representation: the carbon-only ensemble](representation.md)   ·   **Read next:** [Demonstrations](demonstrations.md) →

*ESOREX docs · [Home](index.md) · [Theory](theory.md) · [Representation](representation.md) · **Model** · [Demonstrations](demonstrations.md)*
