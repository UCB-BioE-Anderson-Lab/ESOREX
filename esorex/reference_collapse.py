"""Identifiability-driven feature collapse (the 7-step representation-selection
algorithm).

The raw combined featurizer emits every physical feature at all positional x chemical
abstraction levels, the *candidate lattice*.  Fed directly to the constraint system
that is massive double-counting: the same effect appears as many perfectly
correlated columns.

The collapse selects an experimentally-supported representation:

  * partition the candidate features by their presence pattern across the training
    substrates.  Features in one partition are experimentally indistinguishable, no
    experiment here can separate their weights, so they become ONE energetic variable.
  * keep one representative per partition (the most abstract, for readable labels; the
    choice is cosmetic, a query activates the variable iff it shares ANY member of the
    partition, so behaviour depends on the member SET, not the label).
  * drop partitions present in every substrate or none: they discriminate nothing and
    are absorbed by the intercept E_0.

At prediction the same rule transfers energetic precedent up the hierarchy and
zeroes genuine novelty: a query feature that matches a training partition's
members activates that supported variable; a query feature in no partition (a truly
novel increment) contributes nothing.
"""
from __future__ import annotations

from collections import defaultdict


def _abstraction_key(feature):
    """Lower = more abstract.  feature = ('feat', pos_level, pos_addr, field, value).
    Positional generality carries the abstraction: passenger < subdomain < atom."""
    pos = {"passenger": 0, "subdomain": 1, "atom": 2}.get(feature[1], 3)
    return (pos, str(feature[3]), str(feature))


# Physics fields that describe the same structural channel are grouped so their
# nested-threshold features (e.g. carbon_count>1, >3) stay in one lineage.
_CHANNEL = {
    "carbon_count": "size", "linker_carbon_count": "size",
    "ring_count": "ring", "ring_size": "ring",
    "pi_size": "pi", "aromatic": "pi",
}


def _lineage(feature):
    """The physical channel a feature belongs to.  The collapse merges only
    within a channel, so distinct physical quantities (a proton count and a lone-pair
    count, a size feature and an oxidation state) are never fused merely because they
    share a column on a small training set.  Positional level is NOT part of the
    channel, so the atom/subdomain/passenger views of one quantity still merge."""
    _, _, pos_addr, field, _value = feature
    passenger = pos_addr.split("/")[0]
    return (passenger, _CHANNEL.get(field, field))


class Collapse:
    """Maps a raw candidate-feature set to the collapsed, identifiable representation."""

    def __init__(self, feature_sets, vocab):
        n = len(feature_sets)
        # partition by (physical channel, training presence pattern): features merge
        # only when they are the same channel AND experimentally indistinguishable.
        groups: dict[tuple, list] = defaultdict(list)
        for f in vocab:
            pattern = tuple(1 if f in feature_sets[i] else 0 for i in range(n))
            groups[(_lineage(f), pattern)].append(f)
        # one variable per discriminating partition; store (label, member_set)
        self.partitions = []
        for (lineage, pattern), members in groups.items():
            s = sum(pattern)
            if s == 0 or s == n:                      # discriminates nothing -> intercept
                continue
            label = min(members, key=_abstraction_key)
            self.partitions.append((label, frozenset(members)))
        self.partitions.sort(key=lambda p: _abstraction_key(p[0]))

    @property
    def vocab(self):
        return [label for label, _ in self.partitions]

    def transform(self, raw_feature_set):
        """Collapsed feature set: a variable is active iff the query shares any member."""
        raw = frozenset(raw_feature_set)
        return frozenset(label for label, members in self.partitions if raw & members)

    def transform_all(self, feature_sets):
        return [self.transform(fs) for fs in feature_sets]

    def members(self, label):
        for lbl, mem in self.partitions:
            if lbl == label:
                return mem
        return frozenset()
