"""Invariant tests for esorex.reference_collapse.Collapse.

Chemistry-free: features are synthetic 5-tuples of the documented shape

    (feat_name, pos_level, pos_addr, field, value)
     pos_level in {"passenger", "subdomain", "atom"}

so the collapse behaviour can be checked without any molecule. These pin the
identifiability rules the model and docs promise:

  * partition by (physical channel, training presence pattern)
  * one variable per discriminating partition, labelled by its MOST ABSTRACT
    member (passenger < subdomain < atom), while activation depends on the
    member SET, not the label
  * partitions present in every / no substrate are dropped (absorbed by E_0)
  * distinct physical channels never fuse, even with identical presence
  * deterministic vocabulary order
"""
from esorex.reference_collapse import Collapse, _abstraction_key


def _f(pos_level, pos_addr, field, value=1):
    return ("feat", pos_level, pos_addr, field, value)


# --- abstraction ordering ------------------------------------------------------

def test_abstraction_key_orders_passenger_below_subdomain_below_atom():
    passenger = _abstraction_key(_f("passenger", "p0", "protons"))
    subdomain = _abstraction_key(_f("subdomain", "p0/h1:ring", "protons"))
    atom = _abstraction_key(_f("atom", "p0/0", "protons"))
    assert passenger < subdomain < atom


# --- collapse: merge indistinguishable, label with the most abstract -----------

def test_same_channel_same_pattern_merge_labelled_by_most_abstract():
    f_pass = _f("passenger", "p0", "protons")
    f_atom = _f("atom", "p0/0", "protons")          # same channel + passenger
    # both present in substrate 0, both absent in substrate 1 -> identical pattern
    feature_sets = [frozenset({f_pass, f_atom}), frozenset()]
    vocab = [f_pass, f_atom]
    col = Collapse(feature_sets, vocab)

    assert col.vocab == [f_pass]                    # most-abstract member is the label
    assert col.members(f_pass) == frozenset({f_pass, f_atom})
    # a query carrying ONLY the atom-level member still activates the variable
    assert col.transform(frozenset({f_atom})) == frozenset({f_pass})


def test_different_presence_patterns_stay_separate():
    g1 = _f("atom", "p0/0", "protons")
    g2 = _f("atom", "p0/1", "protons")              # same channel, different pattern
    feature_sets = [frozenset({g1}), frozenset({g2})]
    col = Collapse(feature_sets, [g1, g2])
    assert len(col.partitions) == 2


def test_distinct_channels_never_fuse_even_with_identical_pattern():
    protons = _f("passenger", "p0", "protons")
    lone = _f("passenger", "p0", "lone_pairs")      # different physical channel
    # identical presence pattern (both in sub 0 only) but must NOT merge
    feature_sets = [frozenset({protons, lone}), frozenset()]
    col = Collapse(feature_sets, [protons, lone])
    assert len(col.partitions) == 2


# --- collapse: drop non-discriminating partitions ------------------------------

def test_features_present_everywhere_or_nowhere_are_dropped():
    everywhere = _f("passenger", "p0", "protons")
    nowhere = _f("passenger", "p0", "lone_pairs")
    discriminating = _f("passenger", "p0", "formal_charge")
    feature_sets = [
        frozenset({everywhere, discriminating}),
        frozenset({everywhere}),
    ]
    # `nowhere` is in the vocab but in no substrate
    col = Collapse(feature_sets, [everywhere, nowhere, discriminating])
    assert col.vocab == [discriminating]


# --- determinism ---------------------------------------------------------------

def test_vocab_order_is_deterministic():
    feats = [
        _f("atom", "p0/1", "protons"),
        _f("passenger", "p0", "formal_charge"),
        _f("subdomain", "p0/h1:ring", "oxidation_state"),
    ]
    feature_sets = [frozenset({feats[0]}), frozenset({feats[1]}), frozenset({feats[2]})]
    a = Collapse(feature_sets, list(feats))
    b = Collapse(feature_sets, list(reversed(feats)))
    assert a.vocab == b.vocab                        # order independent of input order
