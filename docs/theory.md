# Theory: EVODEX and the mechanistic pre-filter

**In short:** before ESOREX asks how fast a substrate reacts, it asks whether the reaction
chemistry applies at all. This mechanistic layer, built on EVODEX operators, is the feasibility
gate. It also defines the reactive core and, by exclusion, the *passenger* domain that the
[representation](representation.md) and [model](model.md) act on. A reader focused on the
specificity model can skim this page and continue to the representation.

## EVODEX abstraction levels

Chemical bonds arise from electrons occupying orbitals with definite angular symmetry. The key distinction EVODEX uses is between **sigma bonds**, symmetric around the internuclear axis, and **pi bonds**, which have a nodal plane containing that axis. σ and π connectivity provide a useful **structural abstraction** of the local electronic environment: as a first approximation σ frameworks rotate freely while π systems are conformationally locked, and whether conjugated π extends into a reaction center's neighborhood shifts the orbital interactions available. This is an abstraction, not a physical law, σ rotation can itself be hindered, conjugation imposes partial barriers, and orbital symmetry is only part of what sets reactivity, but it is a compact structural proxy for the local electronic context, which is what EVODEX encodes.

[EVODEX](https://github.com/UCB-BioE-Anderson-Lab/EVODEX) encodes this environment at five abstraction levels, expanding outward from the reactive center:

| Level | What it encodes |
|-------|----------------|
| **A** | Reaction topology only: bond changes and atom-map connectivity, no element identity |
| **B** | Reactive-center atoms with element types and bonding changes |
| **C** | B plus the sigma shell: atoms one sigma bond away from the reactive center |
| **D** | C plus the pi shell: expansion across conjugated and aromatic connectivity |
| **E** | D plus the extended sigma shell: sigma neighbors of the pi system |

A is the most permissive: bare topology, no chemistry. Each level outward adds more of the local electronic environment *as this scheme represents it*. E captures the largest environment the operator scheme reaches, not the full chemical environment, long-range electrostatics, conformation, protonation state, and solvent remain outside it entirely; and every level below E involves some extrapolation about whether the relevant orbital interactions are present.

**D-level is the default in ESOREX.** The pi shell is what distinguishes, for example, a phenolic hydroxyl from an aliphatic one: the aromatic ring's delocalized pi system changes the electronic character of the oxygen attached to it in a way that C-level (sigma-only neighborhood) cannot see. D-level captures this without requiring the full extended shell.

ESOREX uses EVODEX operators as a **reaction-pattern pre-filter**: before scoring a compound with the learned specificity model, it checks that the compound matches the reaction pattern the enzyme's training reactions exhibit at the chosen level. Passing establishes **structural compatibility with a learned reaction pattern**, not physical feasibility, a compound can match the local graph and still be incapable of productive catalysis for reasons outside the representation (binding, conformation, distal residues). This two-stage cascade, reaction-pattern eligibility first, learned specificity second, is what gives ESOREX its selectivity.

## The mechanistic tree

A real enzyme often acts on more than one mechanistic variant of the same reaction. An esterase may accept both esters and thioesters; a methyltransferase may methylate both oxygen and nitrogen nucleophiles. At the B-level EVODEX operator these are distinct, the atom types in the reactive center differ, but at the A-level they collapse to the same bond-change topology, because A-level treats all atoms as equivalent wildcards. The **mechanistic tree** exploits this hierarchy.

The mechanistic tree for an enzyme is a rooted tree of EVODEX operators, built from the full bag of training reactions:

- **Root (A-level):** the topology-only operator encoding the bond changes, with no atom-type information. All reactions that share the same A-level topology, same bond broken, same bond formed, same connectivity of the reactive center, collapse to the same root.
- **Children (B → C → D → E):** at each successive level, reactions are grouped by whether their operators agree at that level. Where they agree, they share a node; where they diverge (e.g. one reaction goes through an oxygen nucleophile, another through sulfur), the node branches.

The result is a tree that captures how specific the chemistry needs to be. At the root, an ester reaction and a thioester reaction are the same thing, same topology. Moving down, at B-level they split. Each path from root to leaf describes one mechanistic variant of the enzyme's catalytic activity.

**Why this matters for specificity modeling:** a substrate's passenger domain is defined as the atoms not matched by the operator at the abstraction level used for screening (D by default). At D-level this covers the reactive center plus its pi shell, more atoms than A alone. The tree is *rooted* at the A-level topology, but the boundary between "reactive domain" and "passenger" is drawn at whatever matching level is in use. The A-level topology is the same across all mechanistic variants, so the reactive center's atom-map numbering is carried over to every partial assigned to the same root, by remapping each partial onto the reference partial's atom map via a graph isomorphism on the A-level operator. Shared *unlabeled* topology does not by itself give a unique correspondence: when the topology has symmetries (automorphisms), several isomorphisms exist, and ESOREX fixes **one** deterministically, so a symmetric reactive center gets a consistent but arbitrary labeling rather than a canonical one. Given that fixed labeling, passenger addresses from different variants are commensurable: a bulky phenyl group adjacent to the reactive center in an ester substrate and the same group in a thioester substrate produce the same passenger [representation](representation.md), and the model can pool evidence across both.

**Matching at prediction time:** when screening a candidate compound, ESOREX applies the complete operators stored at a specified abstraction level (D by default). The complete operators are 1:1 partial SMIRKS derived from each training reaction, each maps one substrate to one product, and at prediction time they are applied to the candidate to test whether it matches the reactive pattern at the required specificity. The pre-filter level is a parameter: choosing D means the candidate must carry the reactive center's pi-shell context; choosing B would accept any atom-type match regardless of pi environment.

**One tree per A-level topology.** If the training reactions span two genuinely distinct A-level topologies, e.g. an enzyme that performs both a phosphorylation and an aminotransfer, two unrelated bond changes, `generate_mechanistic_tree` returns two separate roots, and a separate specificity model would need to be trained for each.

---

**Read next:** [Representation: the carbon-only ensemble](representation.md) →

*ESOREX docs · [Home](index.md) · **Theory** · [Representation](representation.md) · [Model](model.md) · [Demonstrations](demonstrations.md)*
