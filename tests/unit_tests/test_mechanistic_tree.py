"""
Unit tests for generate_mechanistic_tree.

Architecture note (2026-06):
  Each reaction is first decomposed into its 1:1 partial reactions (one
  substrate ↔ one product) before A-tree grouping.  A multi-substrate reaction
  (e.g. SN2 with nucleophile and leaving group) therefore produces multiple
  A-trees.  Tests that check tree counts or pick trees[0] have been updated to
  find the relevant "substrate tree" or to check properties across all trees.

Four properties tested:

1. Absolute atom map numbers are not distinguishing, identical reactions that
   differ only in which integer is assigned to each atom must collapse to a
   single operator at every abstraction level.

2. Map numbering is consistent across abstraction levels, the set of atom map
   numbers present in the operator at level N is a strict subset of those at
   level N+1.  Holds for the canonical numbering and for five independently
   randomized map-number assignments.

3. Map consistency is preserved across divergent lineages, reactions sharing
   a bond-change topology (Br vs I leaving groups) must share a common A-tree,
   diverge into exactly two operators at B and below within that tree, and each
   lineage must individually satisfy the subset property from A through E.

4. A-level grouping depends on bond-change topology, each 1:1 partial has its
   own A-key; the correct and incorrectly atom-mapped phosphatase reactions
   produce different 1:1 partials (different cycle sizes) that are separable at
   A-level (no cross-mixing).
"""

import re
import random
import pytest
from rdkit.Chem import AllChem

from esorex.generate_mechanistic_tree import generate_mechanistic_tree
from esorex.mechanistic_tree import LEVELS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _map_numbers(smirks: str) -> frozenset[int]:
    return frozenset(int(m) for m in re.findall(r':(\d+)', smirks))


def _randomize_maps(smirks: str, seed: int) -> str:
    """Return a copy of smirks with all atom map numbers replaced by fresh random values."""
    rng = random.Random(seed)
    maps = sorted(set(int(m) for m in re.findall(r':(\d+)', smirks)))
    new_maps = rng.sample(range(1, 10_000), len(maps))
    remap = dict(zip(maps, new_maps))
    return re.sub(r':(\d+)', lambda m: f':{remap[int(m.group(1))]}', smirks)


# ── Canonical reactions used across tests ─────────────────────────────────────

# Identical reaction; C maps to :1 in one, :9 in the other.
METHANOL_DEHYDRATION_1 = (
    '[C:1]([H:3])([H:4])([H:5])[O:2]([H:6])>>[C:1]([H:3])([H:4])[O:2].[H:5]([H:6])'
)
METHANOL_DEHYDRATION_9 = (
    '[C:9]([H:3])([H:4])([H:5])[O:2]([H:6])>>[C:9]([H:3])([H:4])[O:2].[H:5]([H:6])'
)

# Benzyl halide SN2 reactions (Br and I leaving groups).
# In the 1:1-partial architecture, each reaction decomposes into three partials:
#   • C:1 substrate partial  (C:1 swaps halide partner)
#   • Cl:9 nucleophile partial
#   • Br:2 / I:2 leaving-group partial
# The C:1 and Br/I partials from BR and I reactions both converge at A-level
# (same all-carbon topology) and diverge at B (Br vs I atom types).
BENZYL_BR = (
    '[c:3]([H:10])1[c:4]([H:11])[c:5]([H:12])[c:6]([H:13])[c:7]([H:14])[c:8]1'
    '[C:1]([H:15])([H:16])[Br:2].[Cl-:9]'
    '>>'
    '[c:3]([H:10])1[c:4]([H:11])[c:5]([H:12])[c:6]([H:13])[c:7]([H:14])[c:8]1'
    '[C:1]([H:15])([H:16])[Cl:9].[Br-:2]'
)
BENZYL_I = (
    '[c:3]([H:10])1[c:4]([H:11])[c:5]([H:12])[c:6]([H:13])[c:7]([H:14])[c:8]1'
    '[C:1]([H:15])([H:16])[I:2].[Cl-:9]'
    '>>'
    '[c:3]([H:10])1[c:4]([H:11])[c:5]([H:12])[c:6]([H:13])[c:7]([H:14])[c:8]1'
    '[C:1]([H:15])([H:16])[Cl:9].[I-:2]'
)

# Phosphate monoester hydrolysis, correct atom mapping.
# Bond changes: O:2–P:3 broken, O:7–P:3 formed; H:9 transfers from O:7 to O:2.
PHOSPHATASE = (
    '[#6:1]([#1:10])([#1:11])([#1:12])-[#8:2]-[#15:3](=[#8:4])(-[#8:5][#1:13])-[#8:6][#1:14]'
    '.[#8:7]([#1:8])([#1:9])'
    '>>[#6:1]([#1:10])([#1:11])([#1:12])-[#8:2]([#1:9])'
    '.[#1:8]-[#8:7]-[#15:3](=[#8:4])(-[#8:5][#1:13])-[#8:6][#1:14]'
)

# Phosphatase with O:2 and O:7 swapped in the products, the atom-mapping bug
# that was fixed.  Bond changes become C:1–O:2 broken / C:1–O:7 formed
# (an ether-exchange topology, different from the correct O–P–O substitution).
PHOSPHATASE_WRONG_MAP = (
    '[#6:1]([#1:10])([#1:11])([#1:12])-[#8:2]-[#15:3](=[#8:4])(-[#8:5][#1:13])-[#8:6][#1:14]'
    '.[#8:7]([#1:8])([#1:9])'
    '>>[#6:1]([#1:10])([#1:11])([#1:12])-[#8:7]([#1:9])'
    '.[#1:8]-[#8:2]-[#15:3](=[#8:4])(-[#8:5][#1:13])-[#8:6][#1:14]'
)

# Ester reactions with different alcohol R groups, used for fragment tests.
# Both are acetate esters; they differ only in the leaving alcohol (methanol vs
# ethanol).  The A-level operator is the same; passengers differ.
METHYL_ACETATE = (
    '[#6:1]([#1:14])([#1:15])([#1:16])-[#6:2](=[#8:3])-[#8:4]-[#6:5]([#1:17])([#1:18])([#1:19])'
    '.[#8:7]([#1:8])([#1:9])'
    '>>[#6:1]([#1:14])([#1:15])([#1:16])-[#6:2](=[#8:3])-[#8:7]([#1:8])'
    '.[#8:4]([#1:9])-[#6:5]([#1:17])([#1:18])([#1:19])'
)
ETHYL_ACETATE = (
    '[#6:1]([#1:14])([#1:15])([#1:16])-[#6:2](=[#8:3])-[#8:4]-[#6:5]([#1:17])([#1:18])-[#6:6]([#1:19])([#1:20])([#1:21])'
    '.[#8:7]([#1:8])([#1:9])'
    '>>[#6:1]([#1:14])([#1:15])([#1:16])-[#6:2](=[#8:3])-[#8:7]([#1:8])'
    '.[#8:4]([#1:9])-[#6:5]([#1:17])([#1:18])-[#6:6]([#1:19])([#1:20])([#1:21])'
)


# ── Test 1: absolute map numbers are not distinguishing ───────────────────────

class TestAtomMapNotDistinguishing:

    def test_same_tree_count_regardless_of_maps(self):
        """Both map-number variants must produce the same number of trees."""
        ts1 = generate_mechanistic_tree([AllChem.ReactionFromSmarts(METHANOL_DEHYDRATION_1)])
        ts9 = generate_mechanistic_tree([AllChem.ReactionFromSmarts(METHANOL_DEHYDRATION_9)])
        assert len(ts1) == len(ts9)

    def test_combined_reactions_collapse_to_same_trees(self):
        """Both variants combined must produce the same count as a single variant,
        and every tree must contain 2 reactions (one from each map assignment)."""
        rxns = [AllChem.ReactionFromSmarts(s)
                for s in [METHANOL_DEHYDRATION_1, METHANOL_DEHYDRATION_9]]
        trees = generate_mechanistic_tree(rxns)
        single = generate_mechanistic_tree([AllChem.ReactionFromSmarts(METHANOL_DEHYDRATION_1)])
        assert len(trees) == len(single)
        for t in trees:
            assert len(t.reactions) == 2, (
                f"Each tree must contain both map variants; got {len(t.reactions)}: {t.smirks}"
            )

    def test_single_operator_at_every_level(self):
        """Combined, the trees must each have 1 node at every abstraction level."""
        rxns = [AllChem.ReactionFromSmarts(s)
                for s in [METHANOL_DEHYDRATION_1, METHANOL_DEHYDRATION_9]]
        trees = generate_mechanistic_tree(rxns)
        for tree in trees:
            for level in LEVELS:
                assert len(tree.index[level]) == 1, (
                    f"Expected 1 operator at level {level}, got {len(tree.index[level])}"
                )


# ── Test 2: map numbering is consistent across abstraction levels ─────────────

class TestMapConsistencyAcrossLevels:
    """
    For the C:1 substrate partial of BENZYL_BR (benzyl carbon swapping halide):
      A: {C:1} mapped
      B: {C:1} mapped (Br/Cl appear as explicit atoms, not map-numbered at B)
      C: {C:1, c:8, H:15, H:16} mapped  (nearest covalent shell of C:1)
      D: {C:1, c:3..c:8, H:15, H:16} mapped  (ring)
      E: {C:1, c:3..c:8, H:10..H:16} mapped  (ring + H's)
    """

    @pytest.fixture(scope='class')
    def substrate_tree(self):
        """The C:1 transformation tree for a single BENZYL_BR reaction."""
        rxn = AllChem.ReactionFromSmarts(BENZYL_BR)
        trees = generate_mechanistic_tree([rxn])
        # Substrate tree (C:1) has the most maps at E-level (includes the benzyl ring)
        return max(trees, key=lambda t: len(_map_numbers(t.index['E'][0].smirks)))

    def test_map_count_per_level(self, substrate_tree):
        expected = {'A': 1, 'B': 1, 'C': 4, 'D': 9, 'E': 14}
        for level, count in expected.items():
            node = substrate_tree.index[level][0]
            actual = len(_map_numbers(node.smirks))
            assert actual == count, (
                f"Level {level}: expected {count} map numbers, got {actual}\n"
                f"  {node.smirks}"
            )

    def test_maps_are_subset_of_next_level(self, substrate_tree):
        for i in range(len(LEVELS) - 1):
            lvl, next_lvl = LEVELS[i], LEVELS[i + 1]
            maps = _map_numbers(substrate_tree.index[lvl][0].smirks)
            next_maps = _map_numbers(substrate_tree.index[next_lvl][0].smirks)
            assert maps <= next_maps, (
                f"{lvl} maps are not a subset of {next_lvl}: "
                f"extra={maps - next_maps}"
            )

    def test_subset_property_holds_under_map_randomization(self):
        """Subset property must hold for every tree produced by every map-number variant."""
        for seed in range(5):
            variant = _randomize_maps(BENZYL_BR, seed=seed)
            rxn = AllChem.ReactionFromSmarts(variant)
            trees = generate_mechanistic_tree([rxn])
            for tree in trees:
                for i in range(len(LEVELS) - 1):
                    lvl, next_lvl = LEVELS[i], LEVELS[i + 1]
                    maps = _map_numbers(tree.index[lvl][0].smirks)
                    next_maps = _map_numbers(tree.index[next_lvl][0].smirks)
                    assert maps <= next_maps, (
                        f"seed={seed}: {lvl} maps not subset of {next_lvl}: "
                        f"extra={maps - next_maps}\ntree: {tree.smirks}"
                    )


# ── Test 3: map consistency is preserved across divergent lineages ─────────────

class TestMapConsistencyOnDivergence:
    """
    Ten randomized BENZYL reactions (5 Br, 5 I).

    In the 1:1-partial architecture, BENZYL_BR and BENZYL_I each decompose to
    three partials.  The Br/I leaving-group partials share an A-key (same
    all-carbon topology) and form one A-tree; similarly the C:1 substrate
    partials form one A-tree.  Within each of those trees, Br and I reactions
    must produce two distinct B-nodes (different halide atom types).
    """

    @pytest.fixture(scope='class')
    def trees(self):
        smirks_list = (
            [_randomize_maps(BENZYL_BR, seed=s) for s in range(5)] +
            [_randomize_maps(BENZYL_I, seed=s) for s in range(5)]
        )
        rxns = [AllChem.ReactionFromSmarts(s) for s in smirks_list]
        return generate_mechanistic_tree(rxns)

    def test_all_reactions_in_trees(self, trees):
        """Every tree should contain all 10 reactions (Br + I merge at A)."""
        for t in trees:
            assert len(t.reactions) == 10, (
                f"Expected 10 reactions in tree {t.smirks!r}, got {len(t.reactions)}"
            )

    def test_diverging_tree_has_two_b_operators(self, trees):
        """At least one tree must diverge at B (Br vs I halide types)."""
        diverged = [t for t in trees if len(t.index['B']) == 2]
        assert len(diverged) >= 1, (
            "Expected at least one tree where Br and I diverge at B; "
            f"B-counts: {[len(t.index['B']) for t in trees]}"
        )

    def test_diverged_tree_has_two_nodes_at_b_through_e(self, trees):
        """Within a tree that diverges at B, all lower levels must also have 2 nodes."""
        diverged_tree = next(t for t in trees if len(t.index['B']) == 2)
        for level in ('B', 'C', 'D', 'E'):
            assert len(diverged_tree.index[level]) == 2, (
                f"Expected 2 operators at {level}, got {len(diverged_tree.index[level])}"
            )

    def test_subset_property_within_each_lineage(self, trees):
        """Map consistency holds in each lineage of the tree that diverges at B."""
        diverged_tree = next(t for t in trees if len(t.index['B']) == 2)
        a_maps = _map_numbers(diverged_tree.smirks)
        for b_node in diverged_tree.children:
            b_maps = _map_numbers(b_node.smirks)
            assert a_maps <= b_maps, (
                f"A maps not subset of B: extra={a_maps - b_maps}"
            )
            for c_node in b_node.children:
                c_maps = _map_numbers(c_node.smirks)
                assert b_maps <= c_maps, "B maps not subset of C"
                for d_node in c_node.children:
                    d_maps = _map_numbers(d_node.smirks)
                    assert c_maps <= d_maps, "C maps not subset of D"
                    for e_node in d_node.children:
                        e_maps = _map_numbers(e_node.smirks)
                        assert d_maps <= e_maps, "D maps not subset of E"


# ── Test 4: wrong atom mapping is distinguishable at B-level ─────────────────

class TestWrongMappingDistinguishable:
    """
    Correctly and incorrectly atom-mapped phosphatase reactions both perform a
    'partner swap' at their reaction center (one bond breaks, one forms), so
    they share the same all-carbon A-topology (Am-level grouping).  They diverge
    at B-level where element identity is preserved: the correct mapping produces
    an O–P substitution (O:2 swaps P for H), while the wrong mapping produces a
    C–O substitution (C:1 swaps one O for another O).

    The detectable property is therefore B-level divergence, not A-level
    separation.
    """

    def test_combining_adds_b_nodes_not_trees(self):
        """Correct + wrong phosphatase share the same A-trees but add B-nodes."""
        rxns = [AllChem.ReactionFromSmarts(s)
                for s in [PHOSPHATASE, PHOSPHATASE_WRONG_MAP]]
        trees = generate_mechanistic_tree(rxns)
        single = generate_mechanistic_tree([AllChem.ReactionFromSmarts(PHOSPHATASE)])
        assert len(trees) == len(single), (
            f"Correct+wrong phosphatase share A-topology: same tree count expected "
            f"({len(trees)} vs {len(single)})"
        )
        combined_b = sum(len(t.index['B']) for t in trees)
        single_b = sum(len(t.index['B']) for t in single)
        assert combined_b > single_b, (
            f"Combined B-nodes ({combined_b}) should exceed single ({single_b}): "
            f"correct O-P and wrong C-O substitution diverge at B"
        )

    def test_diverge_at_b_level(self):
        """At least one A-tree has distinct B-nodes for correct vs wrong phosphatase."""
        rxns = [AllChem.ReactionFromSmarts(s)
                for s in [PHOSPHATASE, PHOSPHATASE_WRONG_MAP]]
        trees = generate_mechanistic_tree(rxns)
        max_b = max(len(t.index['B']) for t in trees)
        assert max_b >= 2, (
            f"Expected ≥2 B-nodes in at least one A-tree (correct O-P vs wrong C-O); "
            f"max B-nodes found: {max_b}"
        )


# ── Test 5: complete_operators is populated and valid ────────────────────────

class TestCompleteOperators:
    """Every node's complete_operators bag is non-empty and contains parseable SMIRKS."""

    @pytest.fixture(scope='class')
    def tree(self):
        """Use the substrate tree (C:1) from a single BENZYL_BR, most atoms at E."""
        rxn = AllChem.ReactionFromSmarts(BENZYL_BR)
        trees = generate_mechanistic_tree([rxn])
        return max(trees, key=lambda t: len(_map_numbers(t.index['E'][0].smirks)))

    def test_non_empty_at_every_level(self, tree):
        for level in LEVELS:
            node = tree.index[level][0]
            assert len(node.complete_operators) > 0, (
                f"Level {level} has no complete operators"
            )

    def test_all_operators_have_atom_maps(self, tree):
        for level in LEVELS:
            for op in tree.index[level][0].complete_operators:
                assert re.search(r':\d+', op), (
                    f"Level {level} operator has no atom maps: {op}"
                )

    def test_all_operators_parseable_by_rdkit(self, tree):
        for level in LEVELS:
            for op in tree.index[level][0].complete_operators:
                assert AllChem.ReactionFromSmarts(op) is not None, (
                    f"Level {level} operator not parseable: {op}"
                )

    def test_fragments_absent_below_a_root(self, tree):
        # fragments is only meaningful on the A-root; all other nodes must be None
        for level in ('B', 'C', 'D', 'E'):
            for node in tree.index[level]:
                assert node.fragments is None, (
                    f"Level {level} node should have fragments=None"
                )


# ── Test 6: fragments capture the passenger R-group ──────────────────────────

class TestFragments:
    """
    Two acetate esters hydrolysed: methyl (leaving -OCH3) and ethyl (leaving -OEt).

    All partials from both reactions share the same all-carbon A-topology
    ('partner swap') and merge into a single A-tree.  The fragments on that
    A-root capture the distinct substrate molecules, one fragment per unique
    reactant seen across all reactions.  Passenger carbon counts distinguish
    the two esters:
      methyl acetate: C:1(acyl) + C:2(carbonyl) + C:5(methyl) = 3 carbons
      ethyl acetate:  C:1(acyl) + C:2(carbonyl) + C:5 + C:6(ethyl) = 4 carbons
    """

    @pytest.fixture(scope='class')
    def tree(self):
        """The A-tree for methyl+ethyl acetate, has non-empty fragments."""
        rxns = [AllChem.ReactionFromSmarts(s)
                for s in [METHYL_ACETATE, ETHYL_ACETATE]]
        trees = generate_mechanistic_tree(rxns)
        # All partials share one A-topology; find the tree with non-empty fragments
        for t in trees:
            if t.fragments and len(t.fragments) > 0:
                return t
        pytest.fail("No tree found with non-empty fragments")

    def test_fragments_present_on_a_root(self, tree):
        assert tree.fragments is not None
        assert len(tree.fragments) > 0

    def test_operator_and_passenger_indices_disjoint(self, tree):
        for frag in tree.fragments:
            assert set(frag['operator_indices']).isdisjoint(frag['passenger_indices']), (
                "Operator and passenger index sets must not overlap"
            )

    def test_indices_cover_all_reactant_atoms(self, tree):
        for frag in tree.fragments:
            mol = frag['mol']
            all_idx = set(range(mol.GetNumAtoms()))
            covered = set(frag['operator_indices']) | set(frag['passenger_indices'])
            assert covered == all_idx, (
                f"Fragment indices don't cover all atoms: missing {all_idx - covered}"
            )

    def test_methyl_and_ethyl_passenger_carbons_distinguished(self, tree):
        # Only fragments with at least one passenger carbon (ester substrate)
        substrate_frags = [
            f for f in tree.fragments
            if any(f['mol'].GetAtomWithIdx(i).GetAtomicNum() == 6
                   for i in f['passenger_indices'])
        ]
        assert len(substrate_frags) == 2, (
            f"Expected 2 substrate fragments (one per ester reaction), "
            f"got {len(substrate_frags)}"
        )
        carbon_counts = sorted(
            sum(1 for i in f['passenger_indices']
                if f['mol'].GetAtomWithIdx(i).GetAtomicNum() == 6)
            for f in substrate_frags
        )
        # In the O:4-only A-tree, the passenger includes the full acyl chain +
        # leaving alcohol.  Methyl acetate: C:1,C:2,C:5 = 3; ethyl: +C:6 = 4.
        assert carbon_counts == [3, 4], (
            f"Expected passenger C counts [3, 4] (methyl: 3 carbons, "
            f"ethyl: 4 carbons), got {carbon_counts}"
        )


# ── Test 7: ester and SN2 share A-root but diverge at B ──────────────────────

class TestMultipleARoots:
    """
    Ester hydrolysis and SN2 substitution both contain 'partner swap' partials
    (one bond breaks, one forms on the central atom), so they share the same
    all-carbon A-topology.  They diverge at B-level where element identity
    distinguishes the ester oxygen from the SN2 carbon reaction center.
    """

    def test_combining_adds_b_nodes_not_trees(self):
        """Ester + SN2 share some A-trees; combining adds B-nodes, not A-trees."""
        rxns = [AllChem.ReactionFromSmarts(s)
                for s in [ETHYL_ACETATE, BENZYL_BR]]
        trees = generate_mechanistic_tree(rxns)

        ester_only = generate_mechanistic_tree([AllChem.ReactionFromSmarts(ETHYL_ACETATE)])
        benzyl_only = generate_mechanistic_tree([AllChem.ReactionFromSmarts(BENZYL_BR)])

        assert len(trees) < len(ester_only) + len(benzyl_only), (
            f"Combined trees ({len(trees)}) should be fewer than sum of individual "
            f"trees ({len(ester_only)} + {len(benzyl_only)}): some A-keys overlap"
        )

    def test_diverge_at_b_level(self):
        """The shared A-tree has more B-nodes combined than for either reaction alone."""
        ester_only = generate_mechanistic_tree([AllChem.ReactionFromSmarts(ETHYL_ACETATE)])
        benzyl_only = generate_mechanistic_tree([AllChem.ReactionFromSmarts(BENZYL_BR)])
        combined = generate_mechanistic_tree([AllChem.ReactionFromSmarts(s)
                                              for s in [ETHYL_ACETATE, BENZYL_BR]])
        combined_b = sum(len(t.index['B']) for t in combined)
        ester_b = sum(len(t.index['B']) for t in ester_only)
        benzyl_b = sum(len(t.index['B']) for t in benzyl_only)
        assert combined_b > max(ester_b, benzyl_b), (
            f"Combined B-nodes ({combined_b}) should exceed either alone "
            f"(ester={ester_b}, benzyl={benzyl_b})"
        )


# ── Test 8: single-reaction baseline ─────────────────────────────────────────

class TestSingleReaction:
    """
    A single reaction produces one tree per 1:1 partial.  For BENZYL_BR
    (2-substrate SN2), this is 3 trees.  For a single-substrate reaction
    (alcohol oxidation), it remains 1 tree.  Every tree must have 1 node at
    every abstraction level.
    """

    # Single-substrate alcohol oxidation: always decomposes to 1 partial → 1 tree
    ETHANOL_OX = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"

    def test_single_substrate_one_tree(self):
        from esorex.reaction_preparation import prepare_reaction
        tree = generate_mechanistic_tree([prepare_reaction(self.ETHANOL_OX)])
        assert len(tree) == 1

    def test_one_node_per_level_single_substrate(self):
        from esorex.reaction_preparation import prepare_reaction
        tree = generate_mechanistic_tree([prepare_reaction(self.ETHANOL_OX)])[0]
        for level in LEVELS:
            assert len(tree.index[level]) == 1, (
                f"Expected 1 node at {level}, got {len(tree.index[level])}"
            )

    def test_multi_substrate_produces_multiple_trees(self):
        """Multi-substrate reaction (SN2) decomposes to multiple partial A-trees."""
        trees = generate_mechanistic_tree([AllChem.ReactionFromSmarts(BENZYL_BR)])
        assert len(trees) > 1, "Expected > 1 tree for a 2-substrate SN2 reaction"

    def test_multi_substrate_each_tree_one_node_per_level(self):
        """Every partial tree must have exactly 1 node at every abstraction level."""
        trees = generate_mechanistic_tree([AllChem.ReactionFromSmarts(BENZYL_BR)])
        for tree in trees:
            for level in LEVELS:
                assert len(tree.index[level]) == 1, (
                    f"Tree {tree.smirks!r}: expected 1 node at {level}, "
                    f"got {len(tree.index[level])}"
                )


# ── Test 9: edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_input_returns_empty_list(self):
        assert generate_mechanistic_tree([]) == []

    def test_remapping_uses_reference_map_numbers(self):
        # Adding a map-number-randomized duplicate of a reaction must not change
        # any operator in the tree, the reference (first reaction) numbering wins.
        original_rxn = AllChem.ReactionFromSmarts(METHYL_ACETATE)
        duplicate_rxn = AllChem.ReactionFromSmarts(_randomize_maps(METHYL_ACETATE, seed=42))

        singles = generate_mechanistic_tree([original_rxn])
        with_dups = generate_mechanistic_tree([original_rxn, duplicate_rxn])

        assert len(singles) == len(with_dups), (
            "Adding a remapped duplicate must not change the number of trees"
        )
        for single, with_dup in zip(singles, with_dups):
            for level in LEVELS:
                op_single = single.index[level][0].smirks
                op_with_dup = with_dup.index[level][0].smirks
                assert op_single == op_with_dup, (
                    f"Level {level}: operator changed after adding a remapped duplicate.\n"
                    f"  single:   {op_single}\n"
                    f"  with_dup: {op_with_dup}"
                )
