"""
Unit tests for esorex.label_substrate.label_substrate.

label_substrate(mol_with_Hs, operator) → List[Dict]
  - Applies the operator's reactant SMARTS to find all matching sites.
  - Each match is a dict with mapped_atoms, unmapped_matched_atoms, unmatched_atoms.
  - Returns an empty list when the operator doesn't match.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from esorex.label_substrate import label_substrate

# Alcohol-oxidation operator: C-OH → C=O, map nums 1 (C) and 2 (O).
OP_SMARTS = "[C:1]([H])[O:2]([H])>>[C:1]=[O:2]"


def _op():
    return rdChemReactions.ReactionFromSmarts(OP_SMARTS)


def _mol(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


# ---------------------------------------------------------------------------
# No match
# ---------------------------------------------------------------------------

def test_no_match_returns_empty_list():
    results = label_substrate(_mol("c1ccccc1"), _op())
    assert results == []


def test_ether_no_match():
    """Ether oxygen has no H, operator requires [O]([H]), so no match."""
    results = label_substrate(_mol("CCOCC"), _op())
    assert results == []


# ---------------------------------------------------------------------------
# Single-site molecule
# ---------------------------------------------------------------------------

def test_ethanol_returns_at_least_one_labeling():
    results = label_substrate(_mol("CCO"), _op())
    assert len(results) >= 1


def test_mapped_atoms_have_correct_map_numbers():
    """Mapped atoms must carry the operator's map numbers (1 for C, 2 for O)."""
    results = label_substrate(_mol("CCO"), _op())
    map_nums = {mapnum for _, mapnum in results[0]["mapped_atoms"]}
    assert map_nums == {1, 2}


def test_mapped_atoms_are_idx_mapnum_tuples():
    results = label_substrate(_mol("CCO"), _op())
    for item in results[0]["mapped_atoms"]:
        assert isinstance(item, tuple) and len(item) == 2
        idx, mapnum = item
        assert isinstance(idx, int)
        assert isinstance(mapnum, int)


def test_all_atom_indices_accounted_for():
    """Union of all three index sets must equal the full atom index range."""
    mol = _mol("CCO")
    results = label_substrate(mol, _op())
    for r in results:
        covered = (
            {idx for idx, _ in r["mapped_atoms"]}
            | set(r["unmapped_matched_atoms"])
            | set(r["unmatched_atoms"])
        )
        assert covered == set(range(mol.GetNumAtoms()))


# ---------------------------------------------------------------------------
# Multi-site molecule
# ---------------------------------------------------------------------------

def test_diol_returns_multiple_labelings():
    """Ethylene glycol has two OH groups → at least two distinct reactive sites."""
    results = label_substrate(_mol("OCCO"), _op())
    # Each match gives a labeling; with uniquify=False we may see duplicates,
    # but there should be labelings for both OH groups.
    assert len(results) >= 2


def test_diol_reactive_atoms_differ_between_sites():
    """The two sites of a diol should involve different C and O atom indices."""
    results = label_substrate(_mol("OCCO"), _op())
    mapped_sets = [frozenset(idx for idx, _ in r["mapped_atoms"]) for r in results]
    # At least two distinct sets of reactive atom indices
    assert len(set(mapped_sets)) >= 2


# ---------------------------------------------------------------------------
# Unmapped matched atoms
# ---------------------------------------------------------------------------

def test_unmapped_matched_atoms_not_in_mapped():
    """unmapped_matched_atoms must not overlap with mapped_atoms indices."""
    results = label_substrate(_mol("CCO"), _op())
    for r in results:
        mapped_idxs = {idx for idx, _ in r["mapped_atoms"]}
        assert not (mapped_idxs & set(r["unmapped_matched_atoms"]))
