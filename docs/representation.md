# Representation: the carbon-only ensemble

ESOREX describes a substrate in two commitments. First, its chemistry is a set of **deltas** (a
heteroatom, a bond that is not single, an oxidized carbon, a charge) hung on a **carbon skeleton**,
described with **elemental and electronic descriptors** (atomic number, oxidation state, formal
charge, valency, orbital occupancy, π character) rather than a predefined functional-group vocabulary.
The choice of a carbon reference, the skeleton/delta split, the subdomains, and the alignment are all
modeling choices; what is avoided is a named-group vocabulary, not modeling. Second, each delta is
**addressed positionally** on a three-scale decomposition, **atom → subdomain → passenger**, so the
model can attribute an effect at whatever resolution the data actually support. The decomposition is
**reconstructable**: the constitutional and electronic structure rebuilds from skeleton-plus-deltas
exactly, up to the stereochemistry that is deliberately excluded (see [Stereochemistry](#stereochemistry)),
so nothing but that excluded information is lost before the model sees it.

## The carbon skeleton

For each substrate `S`, ESOREX keeps the **saturated carbon-only graph**: the carbon atoms and their
C–C connectivity, with every heteroatom deleted, every remaining bond made single, and carbon valence
filled with hydrogen (`esorex/carbon_tree.py`; the reconstruction check lives in
`esorex/hydrogenated_reference.py`).

This is not ordinary hydrogenation. Heteroatoms are **deleted**, not reduced in place, so a ring
closed by a heteroatom opens:

| substrate | carbon skeleton |
|---|---|
| benzene | cyclohexane |
| pyridine | **pentane** (the nitrogen is gone; the ring opens) |
| phenol | cyclohexane |
| anisole | cyclohexane **+ methane** (the methyl detaches with its ether oxygen) |
| acetone | propane |

These are **reference graphs**, not chemical transformations. "pyridine → pentane" means the carbon
reference *for* pyridine is the five-carbon graph, a representational zero-point against which
pyridine's chemistry is measured, not a reduction pyridine could undergo; the nitrogen and the
aromatic π return in full as the addressed deltas.

The skeleton is a real graph (possibly fragmented, which is exactly what the carbon-only definition
implies), paired with a mapping back to `S`. Everything erased returns as an **addressed delta**: a
heteroatom (with the carbon loci it connected), a π bond, an aromatic system, an oxidation-state
change, a charge. The pair carries no less than `S`,

```
S = carbon skeleton + { addressed deltas }
```

and the decomposition is verified lossless (`reconstruct(...) == S`, up to stereochemistry) before
any feature is abstracted from it. A hydroxyl or a guanidinium is therefore an *emergent pattern* of
deltas, never an input to the featurizer.

## The positional structure: passenger, subdomain, atom

The reactive core (the [mechanistic layer](theory.md)) is removed first; what remains is the
**passenger domain**, the fragment the enzyme must accommodate but that does not react. The passenger
is addressed at three positional scales, and every delta is emitted at all three so the model can
work at whichever resolution the data support:

- **passenger**, the whole fragment;
- **subdomain**, a rigid body or a flexible linker within it;
- **atom**, a single carbon locus.

![Four-panel diagram showing the hierarchy of structural decomposition used in ESOREX. From left to right: (1) Reaction Centers, the alpha-carbon, amino group, and carboxylate of an amino acid substrate are highlighted; the aromatic ring and methoxy group below are not highlighted. (2) Passenger Domains, the reaction centers are removed; the remaining phenyl ring, oxygen, and methyl group are highlighted as the passenger domain. (3) Subdomains, the passenger domain is split at the sigma/pi boundary into a rigid subdomain (the aromatic ring plus the directly bonded oxygen) and a flexible subdomain (the methyl group); the root atom connecting back to the reactive center is shown as a distinct node. (4) Atoms, individual atoms are shown as labeled circles for fine-grained addressing.](../assets/readme/subdomains.png)

*Figure: the passenger (what remains after the reactive center is removed) is decomposed positionally into passenger, subdomain, and atom scales relative to the reaction center. In the current carbon-only representation the subdomain is the rigid carbon body (a ring or π system) or a flexible sp³ linker, and heteroatoms such as the ring oxygen shown here hang on it as deltas.*

The middle scale carries most of the weight, and it is a σ/π **rigid-body decomposition**. The
passenger's carbons partition deterministically at σ/π boundaries into **rigid subdomains**, rings
and conjugated or aromatic π systems, and **flexible linkers**, the sp³ carbon runs. Overlapping
rings and π systems merge into one rigid body; the sp³ remainder forms the linkers (`carbon_tree.py`:
`subdomains`). "Rigid vs. flexible" here is an **algorithmic classification** from the σ/π graph, not
a literal claim about conformational rigidity, a saturated ring is not a free linker and many sp³
chains are conformationally constrained; it is used as a structural proxy for how much a region's
shape is pinned by its bonding.

A subdomain is addressed by its **hop count from the reaction core, in units of subdomains**, so the
length *inside* a subdomain is abstracted away. A three-methylene and a four-methylene linker get the
**same subdomain-level address**, a flexible hydrophobic linker at the same hop. They are not the
same represented structure, they still differ at the atom scale, but they are the same *category* at
this scale, and the carbon count that separates them is exactly what the subdomain address ignores.
That is the reason the middle scale exists: given a 3- and a 4-methylene substrate that both react,
the model learns "this enzyme wants a hydrophobic linker here," not "a carbon must sit at depth
four." No length threshold is ever invented, the structural unit itself *is* the abstraction. Aromaticity, ring membership, π character, and whether the
subdomain carries any heteroatom (hydrophobic vs. decorated) are its descriptors.

The atom scale is finer, an individual carbon, for the cases where an effect genuinely turns on one
position rather than the whole subdomain (which oxygen on a ring, not just "an oxygen on the ring").
Its coordinate is pinned by the cross-substrate alignment below; the subdomain scale needs no
alignment, because a "flexible linker at hop 0" is already the same address in every substrate.

## Heteroatoms as deltas: substituents and bridges

Within a passenger, the carbons are the skeleton and the heteroatoms are deltas, grouped into
connected clusters and classified by how many carbons each cluster touches:

- a cluster bonded to **one** carbon is a **substituent** on that carbon (a hydroxyl oxygen, a
  carbonyl oxygen, a terminal amine);
- a cluster bonded to **two or more** carbons is a **bridge** that keeps the carbon skeleton
  connected across the heteroatom (an ether oxygen linking two carbons; arginine's Nε linking its
  chain to the guanidinium carbon).

Bridges are why a carbon cut off from the rest of the skeleton by a heteroatom is never lost: it
stays part of the passenger, reached across the bridge. Each carbon also carries its own
carbon-framework deltas, the non-single bond orders on it (a C=C, the sp² carbon of a carbonyl) and
its oxidation state. Every one of these deltas is addressed at all three scales, atom, subdomain, and
passenger.

## Aligning the atom coordinate across substrates

The subdomain scale is commensurable across molecules by construction (hop plus kind). The finer
**atom** scale needs more: to say "this exact carbon of one molecule is the same position as that
carbon of another," the carbon skeletons are aligned into **one shared frame**.

The frame is grown **most-active-first**: the most active substrate seeds it, and each next
substrate is added by the ordering that **minimizes electron edit distance** to the frame.
Concretely, the two carbon trees are matched node-to-node, and the cost of a matching is the number
of differing **delta fields** between paired carbons (oxidation state, π/bond orders, ring closure,
and the heteroatom substituent and bridge clusters, so the metric sees the electronic deltas, not the
bare graph) plus a fixed penalty for each carbon left unmatched, weighted by the size of the subtree
it drags along. The alignment searches sibling permutations at each branch point for the minimum-cost
matching (`carbon_tree.py`: `_match`, `_cost`); carbons the frame does not yet carry are folded in as
new positions. Permuting sibling branches is the graph analogue of rotating about the connecting
bonds, though not every permutation is a physically realizable rotation. When two orderings tie, the
first found under the molecule's canonical seed order is taken, so the result is **deterministic
though not guaranteed unique**, and a symmetric center can be assigned either of its equivalent
alignments.

This is what lets the model say "the same carbon." Phenylalanine and tyrosine align onto the
identical ring, so tyrosine's para hydroxyl becomes a delta at exactly the coordinate where
phenylalanine has a plain carbon; the model attributes a weight to that position and transfers it
between them. A fixed canonical (CIP) order does not give this, some datasets produce inconsistent
orders, and CIP order is not a claim about physical correspondence; the edit-distance alignment is
what makes the atom addresses commensurable.

So position is carried at two resolutions that meet in the middle: the **subdomain** by canonical
hops (length-independent), and the **atom** by the aligned frame (exact), with the whole
**passenger** as the coarsest.

**One caveat about the atom scale.** Because the frame is seeded most-active-first, the atom
coordinates depend on the activity ranking: a change in measured rates large enough to reorder two
training substrates can relabel atom positions and so change the emitted atom-level features. The
subdomain and passenger scales do not depend on the ranking, and the aligned coordinates are stable
for well-separated substrates, but the representation is not formally invariant to such reorderings.
Multiple passenger fragments (a substrate with more than one reactive center, or a fragment the core
splits off) are each their own passenger slot with independent subdomain and atom coordinates, and
features are namespaced by slot.

## Physical primitives

A delta is described by graph-computable descriptors, with no functional-group lookup. Higher-order
notions (oxidation state, formal lone pairs) are **derived from the counts**, not stored as a second
vocabulary:

| Primitive | Meaning |
|---|---|
| **atomic number** (Z) | which nucleus sits here; reference = 6 (carbon). Seeding a threshold at 6 makes "heavier than carbon" (`Z > 6`) an available abstraction; within the C/N/O/S/P/halogen domain ESOREX targets that reads as "any heteroatom" (strictly `Z > 6` is any element heavier than carbon, so it excludes boron, `Z = 5`), and it lets nitrogen and oxygen share a "substituent present" feature |
| **formal charge** | localized charge at the atom |
| **oxidation state** | the electronegativity-assigned oxidation number, computed per bond (Pauling); a carbonyl carbon reads as oxidized relative to an alkane carbon, so "a carbon bore an oxygen" survives even after the oxygen is stripped to the skeleton |
| **valency** | sum of bond orders |
| **formal lone-pair count** | `(valence electrons − Σ bond orders − formal charge) / 2` from the given valence/resonance representation (neutral O = 2, N = 1, halogen = 3; quaternary ammonium N⁺ = 0; carboxylate O⁻ = 3). This is a **formal** count from the drawn structure, not chemical availability, an amide nitrogen's lone pair is delocalized and largely unavailable yet still counts here |
| **attached protons** | number of bonded hydrogens (an X–H donor is `Z > 6` with attached H) |
| **π character** | a carbon's non-single bond orders, at the atom level; aromaticity and ring membership are properties of the **subdomain** (a rigid π body), not of an isolated atom |

Each numeric primitive is expanded into `> threshold` features pooled across the training set, with
the saturated-carbon reference value seeded as a threshold (`Z > 6`, `lone_pairs > 0`,
`oxidation_state > 0`, …) so that "deviates from the carbon reference" is always an available level.
This is the **chemical abstraction axis**: an exact value is the most specific description, and
coarser thresholds are the more abstract ones. "Electron-withdrawing," "hydrogen-bond donor," and the
other organic notions can *in principle* be **represented** by combinations of these primitives
rather than supplied as named inputs, that is the design intent, not a demonstrated equivalence for
every such notion.

## From primitives to features

The two axes together, positional (atom → subdomain → passenger) and chemical (exact value up to
"deviation present"), generate a **candidate lattice**: every delta described at every supported
level. This is deliberate over-description; the
[identifiability collapse](model.md#the-identifiability-collapse) reduces it to the most specific
representation the training measurements actually determine, and the resulting binary features feed
the constraint system (`esorex/carbon_featurize.py`). The collapse keeps the finest distinction the
data pin down and backs off atom → subdomain → passenger otherwise.

A purely aliphatic side chain is the clearest case. It has no heteroatom or electronic deltas, so at
the subdomain scale it is simply one flexible hydrophobic linker, exactly the right abstraction when
chain length does not matter. Leucine, norleucine, and norvaline share that linker subdomain and
separate only at the atom scale. The mechanism is explicit, and worth stating because features are
not only deltas: alongside the deltas, the **presence of a carbon at each aligned coordinate** is
itself an emitted feature ("a carbon exists at position P"), so the skeleton's *extent* is
represented, not just its decorations. A longer chain has carbon-present features at deeper
coordinates that a shorter chain lacks; the model can therefore tell "no carbon here" from "a plain
carbon here." The collapse keeps those atom-level distinctions only when the data support them, never
through an invented carbon-count threshold.

## Stereochemistry

Stereochemistry is retained in the substrate-to-skeleton mapping but is **not** yet part of the
feature set or the reconstruction round-trip: the current model resolves constitution and electronic
structure, and which stereochemical distinctions become features, at which positional resolution, is
a deliberate later decision rather than an accidental omission. As with any feature, a configuration
difference acquires predictive weight only when the training set contains substrates that differ in
exactly that way. See [Known limitations](../README.md#known-limitations).

---

**Read next:** [Model: energetic specificity](model.md) →   ·   ← [Theory: the mechanistic pre-filter](theory.md)

*ESOREX docs · [Home](index.md) · [Theory](theory.md) · **Representation** · [Model](model.md) · [Demonstrations](demonstrations.md)*
