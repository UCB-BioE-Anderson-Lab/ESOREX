"""
Unit tests for esorex.label_reaction.label_evodex_operator.

label_evodex_operator(rxn) looks up the first EVODEX-E operator that matches the
prepared reaction, returning a result dict with evodex_id, evodex_smirks, and the
standard label_reaction fields.

NOTE: label_evodex_operator reads EVODEX data CSV files from esorex/data/ which are
not checked in to this repository.  The mock-based tests below run always.  The
live-data tests at the bottom of this file are in PURGATORY, they test functionality
(CSV-based EVODEX operator lookup) that is not part of the current pipeline.  The
current approach computes operators fresh via extract_operator_by_abstraction rather
than looking them up in pre-mined CSV files.  These tests are retained in case that
approach is revisited, but they are permanently skipped until esorex/data/ is
populated and the design question is resolved.
"""
import os
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "esorex", "data")
EVODEX_E_PATH = os.path.join(DATA_DIR, "EVODEX-E_synthesis_subset.csv")
data_available = pytest.mark.skipif(
    not os.path.exists(EVODEX_E_PATH),
    reason="EVODEX-E data files not present in esorex/data/"
)


def test_returns_none_for_unmatched_reaction():
    """When no operator matches, label_evodex_operator returns None."""
    # Use a mock that always returns no E candidates.
    from unittest.mock import patch
    from esorex.reaction_preparation import prepare_reaction
    from esorex.label_reaction import label_evodex_operator

    smiles = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
    rxn = prepare_reaction(smiles)

    # Patch the global index to return an empty candidate list for every key.
    with patch("esorex.label_reaction._E_INDEX", {}):
        result = label_evodex_operator(rxn)

    assert result is None


def test_result_has_required_keys_when_matched():
    """When a match is found the result dict contains evodex_id, evodex_smirks,
    mapped_atoms, unmapped_matched_atoms, and unmatched_atoms."""
    from unittest.mock import patch
    from rdkit.Chem import rdChemReactions
    from esorex.reaction_preparation import prepare_reaction
    from esorex.label_reaction import label_evodex_operator

    smiles = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
    rxn = prepare_reaction(smiles)

    # Fake E operator that structurally matches ethanol oxidation.
    op_smirks = "[C:1]([H])[O:2]([H])>>[C:1]=[O:2]"
    fake_index = {
        "ANY": [{
            "_id": "TEST-E001",
            "smirks": op_smirks,
            "rxn": rdChemReactions.ReactionFromSmarts(op_smirks),
            "evodex_f_id_str": "TEST-F001",
        }]
    }

    # Also patch _calculate_formula_diff so it returns a key we control.
    with (
        patch("esorex.label_reaction._E_INDEX", fake_index),
        patch("esorex.label_reaction._calculate_formula_diff", return_value=("ANY", {})),
    ):
        result = label_evodex_operator(rxn)

    assert result is not None
    for key in ("evodex_id", "evodex_smirks", "mapped_atoms",
                "unmapped_matched_atoms", "unmatched_atoms"):
        assert key in result, f"Missing key: {key}"
    assert result["evodex_id"] == "TEST-E001"


# ---------------------------------------------------------------------------
# PURGATORY, tests for CSV-based EVODEX operator lookup
#
# These test label_evodex_operator with real data files.  The current pipeline
# does not use pre-mined CSV lookup; operators are computed fresh via
# extract_operator_by_abstraction.  Retained in case design is revisited.
# Will never run until esorex/data/ CSV files are present AND the design
# question (pre-mined lookup vs. fresh computation) is resolved.
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="PURGATORY: CSV-based EVODEX lookup not part of current pipeline")
def test_alcohol_oxidation_matches_known_operator():
    """Ethanol oxidation should match a known EVODEX-E alcohol-dehydrogenase operator."""
    from esorex.reaction_preparation import prepare_reaction
    from esorex.label_reaction import label_evodex_operator

    smiles = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
    rxn = prepare_reaction(smiles)
    result = label_evodex_operator(rxn)

    assert result is not None, "Expected ethanol oxidation to match an EVODEX-E operator"
    assert result["evodex_id"].startswith("EVODEX")
    assert result["mapped_atoms"][0], "Expected mapped atoms on reactant side"


@pytest.mark.skip(reason="PURGATORY: CSV-based EVODEX lookup not part of current pipeline")
def test_a_level_map_numbers_subset_of_e_level():
    """The A-level operator map numbers should be a subset of the E-level map numbers
    returned by label_evodex_operator (they describe the same reactive centre)."""
    import re
    from esorex.reaction_preparation import prepare_reaction
    from esorex.label_reaction import label_evodex_operator

    smiles = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
    rxn = prepare_reaction(smiles)
    result = label_evodex_operator(rxn)
    assert result is not None

    e_map_nums = frozenset(int(m) for m in re.findall(r":(\d+)", result["evodex_smirks"]))
    mapped_nums = frozenset(mapnum for _, mapnum in result["mapped_atoms"][0])
    assert mapped_nums <= e_map_nums
