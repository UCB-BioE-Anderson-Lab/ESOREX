"""Feature emission from the aligned carbon ensemble.

Each substrate's carbons are placed on the shared frame (esorex/carbon_tree.py); this
module turns the deltas hung on those aligned carbons into the binary feature sets the
identifiability collapse and constraint system consume.

Every feature is emitted at three positional scopes so the collapse can keep whichever
the data support, the atom -> subdomain -> passenger abstraction ladder:

  * ATOM: the exact aligned carbon coordinate.
  * SUBDOMAIN: the rigid body (ring / conjugated pi) or flexible linker the carbon lives
    in, addressed by hops-from-core in units of subdomains, so length inside a subdomain
    is abstracted (a 3- and a 4-methylene linker are the same subdomain).
  * PASSENGER: the whole fragment.

The subdomain's presence and kind (flexible-linker / aromatic-ring / ...) replace any
carbon-count feature: "there is a hydrophobic linker here" is a structural fact, not an
invented threshold.  Numeric quantities (oxidation state, proton count, ...) are pooled
into >threshold features across the training set, with the saturated-carbon reference
value seeded as a threshold.

Feature tuples are ('feat', pos_level, pos_addr, field, value), the format
reference_collapse and constraint_system already expect.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from esorex.carbon_tree import (
    CNode, build_carbon_tree, build_ensemble, align_to_frame, carbon_nodes, subdomains,
)

# saturated sp3-carbon reference values, seeded as thresholds so "deviates from the
# carbon reference" is always an available abstraction (protons>6 = any heteroatom).
_SEED = {"protons": 6, "lone_pairs": 0, "formal_charge": 0, "oxidation_state": 0}


def _raw(coords: Dict[int, tuple], nodes: Dict[int, CNode], subs: dict):
    """Return (numeric_raw, categorical) for one substrate, emitted at the three scopes."""
    numeric: List[tuple] = []
    categorical = set()
    seen_sd = set()

    for sidx, path in coords.items():
        node = nodes[sidx]
        sd = subs.get(sidx)
        slot = path[0] if path else 0
        a_addr = f"p{slot}/" + ".".join(map(str, path[1:]))
        p_addr = f"p{slot}"
        s_addr = f"p{slot}/{sd.addr}" if sd else p_addr
        scopes = (("atom", a_addr), ("subdomain", s_addr), ("passenger", p_addr))

        # skeletal: a carbon exists at this exact aligned coordinate (atom level)
        categorical.add(("feat", "atom", a_addr, "carbon", "present"))
        # subdomain structure (length-independent): the middle rung that replaces
        # carbon_count. "there is a <kind> subdomain at this hop", plus its properties.
        if sd is not None and (slot, sd.addr) not in seen_sd:
            seen_sd.add((slot, sd.addr))
            categorical.add(("feat", "subdomain", s_addr, "subdomain", sd.kind))
            categorical.add(("feat", "passenger", p_addr, f"has_{sd.kind}", "present"))
            if sd.aromatic:
                categorical.add(("feat", "subdomain", s_addr, "aromatic", "present"))
                categorical.add(("feat", "passenger", p_addr, "aromatic", "present"))
            if sd.in_ring:
                categorical.add(("feat", "subdomain", s_addr, "ring", "present"))
            if sd.hydrophobic:
                categorical.add(("feat", "subdomain", s_addr, "hydrophobic", "present"))

        # oxidation state of the carbon, at all three scopes
        for lvl, ad in scopes:
            numeric.append((lvl, ad, "oxidation_state", node.oxidation_state))
        # pi / aromatic on the carbon (atom level; aromatic is also a subdomain property)
        if node.pi_orders:
            field = "aromatic" if any(o == 1.5 for o in node.pi_orders) else "pi"
            categorical.add(("feat", "atom", a_addr, field, "present"))
        if node.ring_closures:
            categorical.add(("feat", "atom", a_addr, "ring", "present"))

        # heteroatom deltas at atom / subdomain / passenger
        clusters = list(node.substituents) + ([node.in_bridge] if node.in_bridge else [])
        for cl in clusters:
            kind = "hetbridge" if cl.is_bridge else "hetsubst"
            categorical.add(("feat", "atom", a_addr, kind, "present"))
            categorical.add(("feat", "subdomain", s_addr, "heteroatom", "present"))
            categorical.add(("feat", "passenger", p_addr, "heteroatom", "present"))
            for (z, q, nh, lp, ar) in cl.atoms:
                for lvl, ad in scopes:
                    numeric.append((lvl, ad, "protons", z))
                numeric.append(("atom", a_addr, "lone_pairs", lp))
                numeric.append(("subdomain", s_addr, "lone_pairs", lp))
                if q != 0:
                    numeric.append(("atom", a_addr, "formal_charge", q))
                    numeric.append(("subdomain", s_addr, "formal_charge", q))
                    numeric.append(("passenger", p_addr, "formal_charge", q))
                if ar:
                    categorical.add(("feat", "atom", a_addr, "aromatic", "present"))
            if any(o > 1.0 for o in cl.carbon_bond_orders):
                categorical.add(("feat", "atom", a_addr, "hetmultibond", "present"))
    return numeric, categorical


def _thresholds(raw_lists):
    values = defaultdict(set)
    for numeric_raw in raw_lists:
        for level, addr, field, val in numeric_raw:
            values[(level, addr, field)].add(val)
    for (level, addr, field), vals in values.items():
        if field in _SEED:
            vals.add(_SEED[field])
    return {k: sorted(v) for k, v in values.items()}


def _expand(numeric_raw, categorical, thresholds):
    out = set(categorical)
    buckets = defaultdict(list)
    for level, addr, field, val in numeric_raw:
        buckets[(level, addr, field)].append(val)
    for (level, addr, field), vals in buckets.items():
        hi = max(vals)
        for t in thresholds.get((level, addr, field), []):
            if hi > t:
                out.add(("feat", level, addr, field, f">{t}"))
    return frozenset(out)


class CarbonFeaturizer:
    """Fit the shared frame + thresholds on a training set; featurize held-out queries."""

    def __init__(self):
        self.frame: CNode = None
        self.thresholds: dict = {}

    def fit(self, mols, cores, energies):
        trees = [build_carbon_tree(m, c) for m, c in zip(mols, cores)]
        ens = build_ensemble(trees, energies)
        self.frame = ens.frame
        raws = [_raw(ens.coords[i], carbon_nodes(trees[i]), subdomains(mols[i], cores[i]))
                for i in range(len(trees))]
        self.thresholds = _thresholds([nr for nr, _ in raws])
        return [_expand(nr, cat, self.thresholds) for nr, cat in raws]

    def transform_one(self, mol, core) -> frozenset:
        tree = build_carbon_tree(mol, core)
        coords = align_to_frame(self.frame, tree)
        numeric, categorical = _raw(coords, carbon_nodes(tree), subdomains(mol, core))
        return _expand(numeric, categorical, self.thresholds)


def feature_vocabulary(feature_sets):
    vocab = set()
    for fs in feature_sets:
        vocab |= fs
    return sorted(vocab, key=str)
