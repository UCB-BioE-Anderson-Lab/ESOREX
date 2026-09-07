"""The two-stage enzyme model: mechanistic feasibility gate + energetic specificity.

This is the join between the two halves of ESOREX that previously had to be wired by hand
at every call site.  `EnergeticSpecificityModel.train(mols, cores, ...)` takes `cores` as a
caller-supplied argument, so every consumer invented its own definition of the reactive
core; one demo hardcoded an amino-acid SMARTS and never touched the mechanistic layer at
all.  This module removes that freedom: the core comes from the A-tree, always.

    train(reactions, rates)
        build the mechanistic tree -> take the A-level root
        the root's `fragments` already carry, per training reaction, the reaction-center
        atoms located BY THE ATOM MAPPING and the passenger as their complement
        -> those are the cores handed to the specificity model

    predict(mol, level="D")
        STAGE 1, feasibility gate: run every complete operator at `level` over the
        candidate with `label_substrate`.  No match -> the chemistry does not apply ->
        rate 0.  This is a hard gate, not a low score.
        STAGE 2, specificity: locate the A-level reaction center on the candidate from the
        matched operator, and score the passenger with the specificity model.

The abstraction level is a SCREENING parameter, not a passenger boundary.  Raising it makes
the feasibility test stricter; it does not change what the specificity model sees.  The
atoms a C/D/E operator matches beyond the A-level center are still passenger and still
carry weight.  Locating the center by the atom mapping rather than by substructure search
is what makes it unambiguous: a bare SMARTS for the reactive pattern also matches the
guanidinium nitrogen of arginine and the aniline nitrogen of 4-aminophenylalanine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from rdkit import Chem
from rdkit.Chem import RWMol, rdChemReactions

from esorex.energetic_specificity import EnergeticSpecificityModel
from esorex.generate_mechanistic_tree import generate_mechanistic_tree
from esorex.label_substrate import label_substrate

DEFAULT_LEVEL = "D"


@dataclass
class EnzymePrediction:
    """A prediction plus its provenance, including why it is zero when it is."""
    rate: float
    energy: Optional[float]
    feasible: bool                  # did any complete operator match at `level`?
    level: str                      # the screening level actually used
    determined: Optional[bool]      # None when infeasible (the model was never consulted)
    novelty: Optional[float]
    core: Optional[frozenset]       # the A-level reaction center located on this substrate


def _strip_maps(mol):
    """A copy with explicit Hs and no atom maps, which is what label_substrate expects."""
    rw = RWMol(Chem.AddHs(mol))
    for a in rw.GetAtoms():
        a.SetAtomMapNum(0)
    return rw.GetMol()


class EnzymeModel:
    """One enzyme: its mechanistic tree and its energetic specificity model."""

    def __init__(self, RT: Optional[float] = None):
        self.tree = None
        self.specificity = EnergeticSpecificityModel() if RT is None \
            else EnergeticSpecificityModel(RT)
        self._ops: dict[str, list] = {}       # level -> [ChemicalReaction]
        self._center_maps: set[int] = set()   # atom-map numbers of the A-level center
        self.info: dict = {}

    # ── root selection ──────────────────────────────────────────────────────
    @staticmethod
    def _pick_root(trees):
        """Choose the A-level root that carries the substrate transformation.

        A fully mapped reaction decomposes into several 1:1 partials, so several A-level
        roots.  For the TyrB transamination there are three: the amine leaving (center =
        N), water arriving (center = O), and the alpha carbon being oxidized (center = C).
        They are the same chemistry seen from three atoms.

        The specificity model is built on the CARBON skeleton -- carbon_tree strips every
        heteroatom and hangs it back as a delta -- so the passenger decomposition is only
        defined when the reaction center sits on that skeleton.  A center that is a
        nitrogen or an oxygen leaves no carbon to root the passenger at.  So: keep the
        roots whose center is carbon in every training fragment, and among those take the
        one whose fragments carry the most atoms, i.e. the substrate transformation rather
        than a byproduct.
        """
        if not trees:
            raise ValueError("no A-level root: generate_mechanistic_tree returned nothing")
        scored = []
        for t in trees:
            frags = t.fragments or []
            if not frags:
                continue
            elements = {frags[i]["mol"].GetAtomWithIdx(j).GetSymbol()
                        for i in range(len(frags)) for j in frags[i]["operator_indices"]}
            size = sum(f["mol"].GetNumHeavyAtoms() for f in frags) / len(frags)
            scored.append((elements == {"C"}, size, t))
        carbon = [x for x in scored if x[0]]
        if not carbon:
            raise ValueError(
                "no A-level root has a carbon reaction center; the carbon-only "
                "representation cannot root a passenger on "
                f"{[sorted(x) for x in (s[0] for s in scored)]}")
        best = max(carbon, key=lambda x: x[1])
        return best[2]

    # ── training ────────────────────────────────────────────────────────────
    def train(self, reactions: Iterable, rates=None, energies=None) -> dict:
        """reactions: mapped 1:1 partial reactions, one per measured substrate, in the
        same order as `rates`/`energies`."""
        reactions = list(reactions)
        trees = generate_mechanistic_tree(reactions)
        self.tree = self._pick_root(trees)

        # The A-root operator's mapped atoms ARE the reaction center; for this tree
        # '*-[*:2]-*>>*-[*:2]=*' that is map 2.  Every complete operator at every level
        # carries the same map numbers, so this locates the center at any screening level.
        self._center_maps = {int(m) for m in re.findall(r':(\d+)', self.tree.smirks)}
        if not self._center_maps:
            raise ValueError(f"A-root operator carries no atom maps: {self.tree.smirks}")

        for lvl in "ABCDE":
            ops = []
            for node in self.tree.index.get(lvl, []):
                ops.extend(node.complete_operators)
            self._ops[lvl] = [rdChemReactions.ReactionFromSmarts(o)
                              for o in dict.fromkeys(ops)]

        frags = self.tree.fragments or []
        if len(frags) != len(reactions):
            raise ValueError(
                f"{len(frags)} labelled fragments for {len(reactions)} reactions; cannot "
                f"align them with the rate labels.")

        # Pair each fragment with ITS OWN reaction's measurement.  Fragment order follows
        # slot assignment inside tree construction, which is not promised to match the
        # order the reactions were passed in; zipping the two would silently mislabel every
        # substrate if it ever diverged, and nothing downstream would fail.
        order = [f.get("reaction_index") for f in frags]
        if any(i is None for i in order):
            raise RuntimeError(
                "training fragments carry no reaction_index; the tree was built by an "
                "older generate_mechanistic_tree that cannot say which reaction a "
                "fragment came from.")
        if sorted(order) != list(range(len(reactions))):
            raise RuntimeError(
                f"fragments do not cover the reactions one-to-one (indices {sorted(order)})")
        if rates is not None:
            rates = [list(rates)[i] for i in order]
        if energies is not None:
            energies = [list(energies)[i] for i in order]
        # Reactions arrive already through prepare_reaction, so the fragment mols are
        # ordinary molecules with full aromaticity perception.  Assert it rather than
        # assume: an unprepared reaction yields QUERY molecules whose rings have aromatic
        # bonds but non-aromatic atoms, and the featurizer's pi-system detection reads
        # atom.GetIsAromatic() -- so phenylalanine would silently featurize as having no
        # ring at all.
        mols = [f["mol"] for f in frags]
        for m in mols:
            if any(b.GetBondType() == Chem.BondType.AROMATIC for b in m.GetBonds()) and \
               not any(a.GetIsAromatic() for a in m.GetAtoms()):
                raise RuntimeError(
                    "training fragment has aromatic bonds but no aromatic atoms: the "
                    "reactions were not run through prepare_reaction, and the featurizer "
                    "would see no pi system.")
        cores = [set(f["operator_indices"]) for f in frags]
        for m, c, src in zip(mols, cores, frags):
            if any(a.IsInRing() for a in m.GetAtoms()) and \
               not any(a.GetIsAromatic() for a in m.GetAtoms()):
                n_ar = sum(1 for b in src["mol"].GetBonds()
                           if b.GetBondType() == Chem.BondType.AROMATIC)
                if n_ar:
                    raise RuntimeError(
                        "aromaticity perception failed on a training fragment: "
                        f"{n_ar} aromatic bonds but no aromatic atoms survived. The "
                        "featurizer would see no pi system.")

        self.info = self.specificity.train(mols, cores, rates=rates, energies=energies)
        self.info.update({
            "a_root_smirks": self.tree.smirks,
            "center_atom_maps": sorted(self._center_maps),
            "complete_operators_by_level": {k: len(v) for k, v in self._ops.items()},
            "core_sizes": sorted({len(c) for c in cores}),
        })
        return self.info

    # ── prediction ──────────────────────────────────────────────────────────
    def _gate(self, mol, level: str):
        """Return (matched_operator, label) for the first complete operator at `level`
        that applies, or (None, None).  This is stage one: feasibility."""
        probe = _strip_maps(mol)
        for op in self._ops.get(level, []):
            labels = label_substrate(probe, op)
            if labels:
                return op, labels[0]
        return None, None

    def predict(self, mol, level: str = DEFAULT_LEVEL) -> EnzymePrediction:
        if self.tree is None:
            raise RuntimeError("train() first")
        if level not in self._ops:
            raise ValueError(f"unknown abstraction level {level!r}; expected one of ABCDE")

        op, label = self._gate(mol, level)
        if op is None:
            # The reaction chemistry does not apply at the requested specificity.  Not a
            # low score: the enzyme cannot perform this transformation on this compound.
            return EnzymePrediction(rate=0.0, energy=None, feasible=False, level=level,
                                    determined=None, novelty=None, core=None)

        # Stage two.  The core is the A-LEVEL center, whatever level gated: take the atoms
        # the matched operator maps to the A-root's map numbers.  AddHs preserves
        # heavy-atom indices, so these indices are valid on the caller's `mol`.
        core = {i for i, mapnum in label["mapped_atoms"] if mapnum in self._center_maps}
        if not core:
            raise RuntimeError(
                f"operator matched at {level} but mapped no A-level center atom "
                f"(maps {sorted(self._center_maps)}); the tree and its complete operators "
                f"disagree about the reaction center.")

        p = self.specificity.predict(mol, core)
        return EnzymePrediction(rate=float(p.rate), energy=float(p.energy), feasible=True,
                                level=level, determined=bool(p.determined),
                                novelty=float(p.novelty), core=frozenset(core))

    def atom_contributions(self, mol, level: str = DEFAULT_LEVEL) -> dict:
        """Per-atom free-energy contributions for the A-level passenger, or {} if the
        candidate does not pass the feasibility gate at `level`."""
        op, label = self._gate(mol, level)
        if op is None:
            return {}
        core = {i for i, mapnum in label["mapped_atoms"] if mapnum in self._center_maps}
        return self.specificity.atom_contributions(mol, core)

    def core_of(self, mol, level: str = DEFAULT_LEVEL) -> set:
        """The A-level reaction center located on `mol`, or an empty set if infeasible."""
        op, label = self._gate(mol, level)
        if op is None:
            return set()
        return {i for i, mapnum in label["mapped_atoms"] if mapnum in self._center_maps}
