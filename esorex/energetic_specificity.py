"""End-to-end energetic specificity model.

Ties the carbon-only aligned-ensemble pipeline into one trainable object:

    rates --(E=-RT ln k)--> energies
    substrate - core --> carbon forest, aligned into a shared frame (carbon_tree)
              --> deltas on aligned carbons --> candidate features (carbon_featurize)
              --> identifiability collapse (reference_collapse)
              --> ConstraintSystem  (+ minimal cooperative terms for exactness)

Prediction returns energy, rate, and provenance (determined? / novelty), so a held-out
substrate carries an explicit statement of how supported its prediction is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from esorex.constraint_system import (
    ConstraintSystem, RT_KCAL_298, energy_from_rate, rate_from_energy,
)
from esorex.reference_collapse import Collapse
from esorex.carbon_featurize import CarbonFeaturizer, feature_vocabulary


@dataclass
class Prediction:
    energy: float
    rate: float
    determined: bool
    novelty: float


class EnergeticSpecificityModel:
    def __init__(self, RT: float = RT_KCAL_298):
        self.RT = RT
        self._featurizer: Optional[CarbonFeaturizer] = None
        self._collapse: Optional[Collapse] = None
        self._cs: Optional[ConstraintSystem] = None
        self.info: dict = {}

    def train(self, mols, cores, rates=None, energies=None) -> dict:
        if energies is None:
            energies = [float(energy_from_rate(k, self.RT)) for k in rates]
        energies = [float(e) for e in energies]

        self._featurizer = CarbonFeaturizer()
        raw_sets = self._featurizer.fit(mols, cores, energies)
        self._collapse = Collapse(raw_sets, feature_vocabulary(raw_sets))
        collapsed = self._collapse.transform_all(raw_sets)
        cs = ConstraintSystem(self._collapse.vocab, collapsed, energies)

        added, blocked = [], False
        if not cs.consistent:
            try:
                added = cs.restore_exactness(pick_pair=lambda c: sorted(c, key=str)[0])
            except RuntimeError:
                blocked = True
        self._cs = cs
        self.info = {
            "n_substrates": len(mols),
            "n_raw_features": len(feature_vocabulary(raw_sets)),
            "n_collapsed_features": len(self._collapse.vocab),
            **cs.diagnostics(),
            "cooperative_terms_added": len(added),
            "exact": cs.consistent,
            "exact_blocked_by_featurization": blocked,
        }
        return self.info

    def predict(self, mol, core) -> Prediction:
        q = self._collapse.transform(self._featurizer.transform_one(mol, core))
        E, determined, novelty = self._cs.predict_energy(q)
        return Prediction(E, float(rate_from_energy(E, self.RT)), determined, novelty)

    def atom_contributions(self, mol, core) -> dict:
        """Attribute each active collapsed variable's learned weight to the carbon(s) it
        addresses on this substrate, via the aligned coordinate.  Returns {carbon atom
        index: free-energy contribution}; summed over carbons plus the intercept it
        reproduces the predicted energy exactly.  For interpretability / painting."""
        from collections import defaultdict
        from esorex.carbon_tree import build_carbon_tree, align_to_frame, subdomains

        tree = build_carbon_tree(mol, core)
        coords = align_to_frame(self._featurizer.frame, tree)
        subs = subdomains(mol, core)
        path_sidx = {p: s for s, p in coords.items()}
        slot_carbons = defaultdict(list)
        sd_carbons = defaultdict(list)
        for s, p in coords.items():
            slot_carbons[p[0]].append(s)
            if s in subs:
                sd_carbons[(p[0], subs[s].addr)].append(s)

        def addressed(addr):
            if "/" not in addr:                       # passenger level: all its carbons
                return slot_carbons.get(int(addr[1:]), [])
            head, rest = addr.split("/", 1)
            slot = int(head[1:])
            if ":" in rest:                           # subdomain level: its carbons
                return sd_carbons.get((slot, rest), [])
            path = (slot,) + (tuple(int(x) for x in rest.split(".")) if rest else ())
            s = path_sidx.get(path)
            return [s] if s is not None else []

        raw = self._featurizer.transform_one(mol, core)
        active = self._collapse.transform(raw)
        widx = {v: i for i, v in enumerate(self._collapse.vocab)}
        w = self._cs.w_p
        contrib = defaultdict(float)
        for V in active:
            wV = float(w[widx[V]])
            members = self._collapse.members(V) & raw
            atoms = {s for m in members for s in addressed(m[2])}
            if not atoms:
                continue
            for s in atoms:
                contrib[s] += wV / len(atoms)
        return dict(contrib)
