"""
Integration tests for esorex.generate_mechanistic_tree.generate_mechanistic_tree.

generate_mechanistic_tree(reactions: list[ChemicalReaction]) -> list[MechanisticTree]

Architecture note (updated 2026-06):
  Each full reaction is first decomposed into its 1:1 partial reactions (one
  substrate ↔ one product). A-trees are built on these partials, so each tree
  corresponds to a single substrate transformation and its specificity model
  targets that one substrate. Multi-substrate reactions (e.g. ester hydrolysis
  with water) therefore produce multiple A-trees, one per molecule involved.

  The key benefit: full reactions (with explicit cofactors) and substrate-only
  partial reactions that encode the same transformation reduce to the same A-tree.
  Mixing full and partial data no longer splits the substrate tree.

These tests cover:
  - Single-reaction input: one tree, correct A–E operator hierarchy
  - Multiple reactions sharing the same A-level abstraction: grouped into one tree
  - Multiple reactions with different A-level topologies: separate trees
  - Reactions that diverge at D/E but share A/B/C: substrate tree branches at D
  - Integration handover: prepare_reaction output → generate_mechanistic_tree
  - Edge cases: empty input
  - Full vs partial completeness: both collapse into same substrate A-tree

Testing strategy (mirrors test_prepare_reaction.py):
  Human reviewer validates chemistry via reports/mechanistic_tree.html.
  Golden strings are captured from validated runs and locked in here.
  These tests are regression guards, a failing test means the algorithm output
  changed; the developer must re-review the report before updating the golden.

Running example:
  Alcohol oxidation:  C–OH → C=O   (no second molecule, double bond formed)
  Ester hydrolysis:   R–COOR + H₂O → R–COOH + ROH   (bond cleavage, 2 molecules)
"""

import pytest
from esorex.reaction_preparation import prepare_reaction
from esorex.generate_mechanistic_tree import generate_mechanistic_tree
from esorex.mechanistic_tree import LEVELS


# ---------------------------------------------------------------------------
# Shared reaction SMILES (heavy-atom mapped; prepare_reaction adds H maps)
# ---------------------------------------------------------------------------

ETHANOL_OX   = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
PROPANOL_OX  = "[CH3:4][CH2:3][CH2:1][OH:2]>>[CH3:4][CH2:3][CH:1]=[O:2]"

ALKYL_ESTER  = "[CH3:1][C:2](=[O:3])[O:4][CH2:5][CH3:6].[OH2:7]>>[CH3:1][C:2](=[O:3])[OH:7].[OH:4][CH2:5][CH3:6]"
ARYL_ESTER   = ("[CH3:1][C:2](=[O:3])[O:4][c:5]1[cH:9][cH:10][cH:11][cH:12][cH:13]1.[OH2:7]"
                ">>[CH3:1][C:2](=[O:3])[OH:7].[OH:4][c:5]1[cH:9][cH:10][cH:11][cH:12][cH:13]1")


# ---------------------------------------------------------------------------
# Scenario 1: single reaction → exactly one tree
# ---------------------------------------------------------------------------

# Golden values validated via reports/mechanistic_tree.html
SINGLE_A_SMIRKS = "*-[*:2]-[*:3]-*>>[*:2]=[*:3]"
SINGLE_B_SMIRKS = "[#1]-[#6:2]-[#8:3]-[#1]>>[#6:2]=[#8:3]"
SINGLE_COMPLETE_OPS = ["*-[*:2]-[*:3]-*>>[*:2]=[*:3]"]


def test_single_reaction_returns_one_tree():
    rxn = prepare_reaction(ETHANOL_OX)
    trees = generate_mechanistic_tree([rxn])
    assert len(trees) == 1


def test_single_reaction_a_smirks():
    rxn = prepare_reaction(ETHANOL_OX)
    tree = generate_mechanistic_tree([rxn])[0]
    assert tree.smirks == SINGLE_A_SMIRKS


def test_single_reaction_all_levels_populated():
    rxn = prepare_reaction(ETHANOL_OX)
    tree = generate_mechanistic_tree([rxn])[0]
    for level in LEVELS:
        assert len(tree.index[level]) == 1, f"Expected 1 node at level {level}"


def test_single_reaction_b_smirks():
    rxn = prepare_reaction(ETHANOL_OX)
    tree = generate_mechanistic_tree([rxn])[0]
    assert tree.index["B"][0].smirks == SINGLE_B_SMIRKS


def test_single_reaction_complete_operators():
    rxn = prepare_reaction(ETHANOL_OX)
    tree = generate_mechanistic_tree([rxn])[0]
    assert sorted(tree.complete_operators) == sorted(SINGLE_COMPLETE_OPS)


def test_single_reaction_has_one_reaction_reference():
    rxn = prepare_reaction(ETHANOL_OX)
    tree = generate_mechanistic_tree([rxn])[0]
    assert len(tree.reactions) == 1


def test_single_reaction_fragments_populated():
    rxn = prepare_reaction(ETHANOL_OX)
    tree = generate_mechanistic_tree([rxn])[0]
    assert tree.fragments is not None
    assert len(tree.fragments) == 1
    frag = tree.fragments[0]
    assert "mol" in frag
    assert "operator_indices" in frag
    assert "passenger_indices" in frag
    assert len(frag["operator_indices"]) > 0
    assert len(frag["passenger_indices"]) > 0


# ---------------------------------------------------------------------------
# Scenario 2: two alcohol oxidations, same A-group, one tree
# ---------------------------------------------------------------------------

SAME_A_SMIRKS = "*-[*:2]-[*:3]-*>>[*:2]=[*:3]"
SAME_A_COMPLETE_OPS_SORTED = [
    "*-[*:1]-[*:2]-*>>[*:1]=[*:2]",
    "*-[*:2]-[*:3]-*>>[*:2]=[*:3]",
]


def test_same_a_group_returns_one_tree():
    rxns = [prepare_reaction(s) for s in (ETHANOL_OX, PROPANOL_OX)]
    trees = generate_mechanistic_tree(rxns)
    assert len(trees) == 1


def test_same_a_group_reaction_count():
    rxns = [prepare_reaction(s) for s in (ETHANOL_OX, PROPANOL_OX)]
    tree = generate_mechanistic_tree(rxns)[0]
    assert len(tree.reactions) == 2


def test_same_a_group_a_smirks():
    rxns = [prepare_reaction(s) for s in (ETHANOL_OX, PROPANOL_OX)]
    tree = generate_mechanistic_tree(rxns)[0]
    assert tree.smirks == SAME_A_SMIRKS


def test_same_a_group_single_e_node():
    """Ethanol and propanol differ only in passenger length; same operator at all levels."""
    rxns = [prepare_reaction(s) for s in (ETHANOL_OX, PROPANOL_OX)]
    tree = generate_mechanistic_tree(rxns)[0]
    assert len(tree.index["E"]) == 1


def test_same_a_group_complete_operators():
    rxns = [prepare_reaction(s) for s in (ETHANOL_OX, PROPANOL_OX)]
    tree = generate_mechanistic_tree(rxns)[0]
    assert sorted(tree.complete_operators) == SAME_A_COMPLETE_OPS_SORTED


def test_same_a_group_two_fragments():
    rxns = [prepare_reaction(s) for s in (ETHANOL_OX, PROPANOL_OX)]
    tree = generate_mechanistic_tree(rxns)[0]
    assert len(tree.fragments) == 2


# ---------------------------------------------------------------------------
# Scenario 3: different chemical transformations → separate A-trees
#
# Alcohol oxidation (C–OH → C=O) is a single-substrate reaction → 1 partial.
# Ester hydrolysis (ester + H₂O → acid + alcohol) has two substrate molecules →
# decomposes to 3 one-to-one partials (leaving-group, water, acyl-carrier).
# Combined, the tree set has 4 A-trees; they never group alcohol ox with ester.
# ---------------------------------------------------------------------------

ALCOHOL_OX_A_SMIRKS = "*-[*:2]-[*:3]-*>>[*:2]=[*:3]"


def test_different_a_groups_produces_multiple_trees():
    """Alcohol oxidation and ester hydrolysis are distinct transformations → separate trees."""
    rxns = [prepare_reaction(ETHANOL_OX), prepare_reaction(ALKYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    assert len(trees) > 1


def test_different_a_groups_alcohol_ox_tree_present():
    """The alcohol oxidation tree is always present as its own A-tree."""
    rxns = [prepare_reaction(ETHANOL_OX), prepare_reaction(ALKYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    found = {t.smirks for t in trees}
    assert ALCOHOL_OX_A_SMIRKS in found


def test_different_a_groups_alcohol_ox_has_one_reaction():
    rxns = [prepare_reaction(ETHANOL_OX), prepare_reaction(ALKYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    alc_tree = next(t for t in trees if t.smirks == ALCOHOL_OX_A_SMIRKS)
    assert len(alc_tree.reactions) == 1


def test_different_a_groups_no_cross_mixing():
    """No single tree should contain reactions from both transformations."""
    rxns = [prepare_reaction(ETHANOL_OX), prepare_reaction(ALKYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    for t in trees:
        assert len(t.reactions) == 1, (
            f"A tree with >1 reaction would imply cross-mixing of distinct transformations: {t.smirks}"
        )


# ---------------------------------------------------------------------------
# Scenario 4: same substrate-transformation A-tree but diverge at D
#, alkyl ester vs aryl ester leaving group
#
# Both ester reactions decompose into three 1:1 partials. The leaving-group
# partial (O:4 departs from ester C) groups both alkyl and aryl together,
# because they share the same bond-change topology at A/B/C. They diverge at D
# when the leaving group's aliphatic vs aromatic character is introduced.
# ---------------------------------------------------------------------------

# Leaving-group substrate tree (tree where O:4 leaves from ester)
ESTER_LEAVING_B_SMIRKS = "[#6]-[#8:4]>>[#1]-[#8:4]"
ALKYL_ESTER_D_SMIRKS = "[#6]-[#6](=[#8])-[#8:4]-[#6:5]>>[#1]-[#8:4]-[#6:5]"
ARYL_ESTER_D_SMIRKS = (
    "[#6]-[#6](=[#8])-[#8:4]-[#6:5]1:[#6:9]:[#6:10]:[#6:11]:[#6:12]:[#6:13]:1"
    ">>[#1]-[#8:4]-[#6:5]1:[#6:9]:[#6:10]:[#6:11]:[#6:12]:[#6:13]:1"
)


def _ester_leaving_tree(trees):
    """Return the leaving-group substrate tree (2 reactions, diverges at D)."""
    for t in trees:
        if len(t.reactions) == 2 and len(t.index.get("D", [])) == 2:
            return t
    return None


def test_diverge_at_d_leaving_group_tree_exists():
    """A tree where both ester reactions group together, diverging at D, must exist."""
    rxns = [prepare_reaction(ALKYL_ESTER), prepare_reaction(ARYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    assert _ester_leaving_tree(trees) is not None


def test_diverge_at_d_reaction_count():
    rxns = [prepare_reaction(ALKYL_ESTER), prepare_reaction(ARYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    tree = _ester_leaving_tree(trees)
    assert len(tree.reactions) == 2


def test_diverge_at_d_shared_b_node():
    rxns = [prepare_reaction(ALKYL_ESTER), prepare_reaction(ARYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    tree = _ester_leaving_tree(trees)
    assert len(tree.index["B"]) == 1
    assert tree.index["B"][0].smirks == ESTER_LEAVING_B_SMIRKS


def test_diverge_at_d_two_d_nodes():
    rxns = [prepare_reaction(ALKYL_ESTER), prepare_reaction(ARYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    tree = _ester_leaving_tree(trees)
    assert len(tree.index["D"]) == 2


def test_diverge_at_d_node_smirks():
    rxns = [prepare_reaction(ALKYL_ESTER), prepare_reaction(ARYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    tree = _ester_leaving_tree(trees)
    d_smirks = {n.smirks for n in tree.index["D"]}
    assert ALKYL_ESTER_D_SMIRKS in d_smirks
    assert ARYL_ESTER_D_SMIRKS in d_smirks


def test_diverge_at_d_complete_operators():
    rxns = [prepare_reaction(ALKYL_ESTER), prepare_reaction(ARYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    tree = _ester_leaving_tree(trees)
    assert sorted(tree.complete_operators) == ["*-[*:4]>>*-[*:4]"]


def test_diverge_at_d_level_node_counts():
    """A/B/C share; D/E branch, golden node counts at each level."""
    rxns = [prepare_reaction(ALKYL_ESTER), prepare_reaction(ARYL_ESTER)]
    trees = generate_mechanistic_tree(rxns)
    tree = _ester_leaving_tree(trees)
    expected = {"A": 1, "B": 1, "C": 1, "D": 2, "E": 2}
    for level, count in expected.items():
        assert len(tree.index[level]) == count, f"Level {level}: expected {count} nodes"


# ---------------------------------------------------------------------------
# Scenario 5: integration handover, prepare_reaction → generate_mechanistic_tree
# ---------------------------------------------------------------------------

def test_prepare_reaction_output_accepted():
    """Output of prepare_reaction must be accepted by generate_mechanistic_tree
    without error and produce a non-empty result."""
    rxn = prepare_reaction(ETHANOL_OX)
    trees = generate_mechanistic_tree([rxn])
    assert len(trees) == 1
    assert trees[0].smirks is not None


def test_prepare_reaction_pipeline_complete_operators_non_empty():
    """complete_operators must not be empty when prepare_reaction feeds the tree builder.

    Historically, ReactionToSmarts could serialise atoms in ways split_reaction
    rejected, leaving complete_operators = [].  This test guards that regression.
    """
    rxn = prepare_reaction(ETHANOL_OX)
    tree = generate_mechanistic_tree([rxn])[0]
    assert len(tree.complete_operators) > 0


def test_prepare_reaction_pipeline_fragments_non_empty():
    rxn = prepare_reaction(ETHANOL_OX)
    tree = generate_mechanistic_tree([rxn])[0]
    assert tree.fragments and len(tree.fragments) > 0


# ---------------------------------------------------------------------------
# Scenario 6: edge cases
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_list():
    trees = generate_mechanistic_tree([])
    assert trees == []


def test_index_has_all_level_keys():
    """Tree index must contain all ABCDE keys even if some levels are empty."""
    rxn = prepare_reaction(ETHANOL_OX)
    tree = generate_mechanistic_tree([rxn])[0]
    for level in LEVELS:
        assert level in tree.index


# ---------------------------------------------------------------------------
# Scenarios 7-9: reaction completeness, full vs partial methylation
#
# Full:    R-OH + CH₃I → R-OCH₃ + HI   (cofactor + byproduct tracked)
# Partial: R-O[H] → R-OCH₃             (substrate-only; H written explicitly
#                                        to bypass evodex heavy-atom validator)
#
# Architecture: each full reaction decomposes to multiple 1:1 partials.
# The alcohol O-methylation partial (O:3 transformed) has the same A-key
# whether it comes from a full or partial reaction. Full+partial inputs for
# the same enzyme therefore collapse into one substrate A-tree.
# ---------------------------------------------------------------------------

ETHANOL_METHYL_FULL  = "[CH3:1][CH2:2][OH:3].[CH3:4][I:5]>>[CH3:1][CH2:2][O:3][CH3:4].[H][I:5]"
PROPANOL_METHYL_FULL = "[CH3:6][CH2:7][CH2:2][OH:3].[CH3:4][I:5]>>[CH3:6][CH2:7][CH2:2][O:3][CH3:4].[H][I:5]"
ETHANOL_METHYL_PART  = "[CH3:1][CH2:2][O:3][H]>>[CH3:1][CH2:2][O:3][CH3]"
PROPANOL_METHYL_PART = "[CH3:6][CH2:7][CH2:2][O:3][H]>>[CH3:6][CH2:7][CH2:2][O:3][CH3]"

# Substrate O-methylation tree (O:3 transformed), same A-key for full and partial
FULL_SUBSTRATE_A_SMIRKS = "*-[*:3]>>[*:3]-*"
FULL_SUBSTRATE_B_SMIRKS = "[#1]-[#8:3]>>[#8:3]-[#6]"
FULL_SUBSTRATE_COMPLETE_OPS_SORTED = ["*-[*:3]>>[*:3]-*"]

PART_A_SMIRKS = "*-[*:3]>>*-[*:3]"
PART_B_SMIRKS = "[#1]-[#8:3]>>[#6]-[#8:3]"
PART_COMPLETE_OPS_SORTED = ["*-[*:3]>>*-[*:3]"]


def _methylation_substrate_tree(trees):
    """Find the O-methylation substrate tree among the A-trees."""
    # The substrate tree has the most reactions when mixing full+partial;
    # for pure-full or pure-partial it is the unique or dominant tree.
    return max(trees, key=lambda t: (len(t.reactions), ':3' in t.smirks))


def test_full_methylation_produces_multiple_trees():
    """Full multi-substrate reaction decomposes to 1:1 partials → multiple A-trees."""
    rxns = [prepare_reaction(ETHANOL_METHYL_FULL), prepare_reaction(PROPANOL_METHYL_FULL)]
    trees = generate_mechanistic_tree(rxns)
    assert len(trees) > 1


def test_full_methylation_substrate_tree_groups_both_reactions():
    """Both full methylation reactions must appear in the substrate O-methylation tree."""
    rxns = [prepare_reaction(ETHANOL_METHYL_FULL), prepare_reaction(PROPANOL_METHYL_FULL)]
    trees = generate_mechanistic_tree(rxns)
    substrate = next((t for t in trees if t.smirks == FULL_SUBSTRATE_A_SMIRKS), None)
    assert substrate is not None, f"Expected substrate tree with smirks={FULL_SUBSTRATE_A_SMIRKS}"
    assert len(substrate.reactions) == 2


def test_full_methylation_substrate_a_smirks():
    rxns = [prepare_reaction(ETHANOL_METHYL_FULL), prepare_reaction(PROPANOL_METHYL_FULL)]
    trees = generate_mechanistic_tree(rxns)
    substrate = next(t for t in trees if t.smirks == FULL_SUBSTRATE_A_SMIRKS)
    assert substrate.smirks == FULL_SUBSTRATE_A_SMIRKS


def test_full_methylation_substrate_b_smirks():
    rxns = [prepare_reaction(ETHANOL_METHYL_FULL), prepare_reaction(PROPANOL_METHYL_FULL)]
    trees = generate_mechanistic_tree(rxns)
    substrate = next(t for t in trees if t.smirks == FULL_SUBSTRATE_A_SMIRKS)
    assert substrate.index["B"][0].smirks == FULL_SUBSTRATE_B_SMIRKS


def test_full_methylation_substrate_complete_operators():
    rxns = [prepare_reaction(ETHANOL_METHYL_FULL), prepare_reaction(PROPANOL_METHYL_FULL)]
    trees = generate_mechanistic_tree(rxns)
    substrate = next(t for t in trees if t.smirks == FULL_SUBSTRATE_A_SMIRKS)
    assert sorted(substrate.complete_operators) == FULL_SUBSTRATE_COMPLETE_OPS_SORTED


def test_partial_methylation_both_reactions_in_one_tree():
    rxns = [prepare_reaction(ETHANOL_METHYL_PART), prepare_reaction(PROPANOL_METHYL_PART)]
    trees = generate_mechanistic_tree(rxns)
    assert len(trees) == 1


def test_partial_methylation_a_smirks():
    rxns = [prepare_reaction(ETHANOL_METHYL_PART), prepare_reaction(PROPANOL_METHYL_PART)]
    tree = generate_mechanistic_tree(rxns)[0]
    assert tree.smirks == PART_A_SMIRKS


def test_partial_methylation_b_smirks():
    rxns = [prepare_reaction(ETHANOL_METHYL_PART), prepare_reaction(PROPANOL_METHYL_PART)]
    tree = generate_mechanistic_tree(rxns)[0]
    assert tree.index["B"][0].smirks == PART_B_SMIRKS


def test_partial_methylation_complete_operators():
    rxns = [prepare_reaction(ETHANOL_METHYL_PART), prepare_reaction(PROPANOL_METHYL_PART)]
    tree = generate_mechanistic_tree(rxns)[0]
    assert sorted(tree.complete_operators) == PART_COMPLETE_OPS_SORTED


def test_mixed_completeness_substrate_tree_merges():
    """Full + partial for the same transformation → substrate tree collects both.

    Previously this split into two separate trees (before 1:1-partial redesign).
    Now: the O-methylation partial from the full reaction has the same A-key as
    the O-methylation partial reaction, so they group into one substrate tree.
    """
    rxns = [prepare_reaction(ETHANOL_METHYL_FULL), prepare_reaction(PROPANOL_METHYL_PART)]
    trees = generate_mechanistic_tree(rxns)
    substrate = next((t for t in trees if len(t.reactions) >= 2), None)
    assert substrate is not None, "Expected a substrate tree merging full and partial reactions"
    assert len(substrate.reactions) == 2


def test_mixed_completeness_all_four_reactions_substrate_tree():
    """Full+partial for both ethanol and propanol methylations → 4-reaction substrate tree."""
    rxns = [
        prepare_reaction(ETHANOL_METHYL_FULL), prepare_reaction(PROPANOL_METHYL_FULL),
        prepare_reaction(ETHANOL_METHYL_PART), prepare_reaction(PROPANOL_METHYL_PART),
    ]
    trees = generate_mechanistic_tree(rxns)
    substrate = next((t for t in trees if len(t.reactions) == 4), None)
    assert substrate is not None, "Expected a substrate tree with all 4 reactions"
