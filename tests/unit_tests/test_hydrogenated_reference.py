"""Tests for the hydrogenated-reference representation H(S) and its transformations.

Two guarantees:
  * H(S) matches the design anchors (benzene->cyclohexane, pyridine->pentane, ...);
  * the reconstruction invariant  H(S) + T(S) == S  holds constitutionally and
    electronically (stereochemistry excluded by design) for canonical motifs and
    real dataset substrates.
"""
import pytest
from rdkit import Chem

from esorex.hydrogenated_reference import build_reference, verify_reconstruction


def _canon(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


# H(S) is the saturated carbon-only graph: delete heteroatoms (a ring closed by a
# heteroatom opens), saturate C-C bonds.  NOT ordinary hydrogenation.
ANCHORS = [
    ("c1ccccc1",              "C1CCCCC1"),     # benzene -> cyclohexane
    ("c1ccncc1",              "CCCCC"),        # pyridine -> pentane (N deleted, ring opens)
    ("Oc1ccccc1",             "C1CCCCC1"),     # phenol -> cyclohexane
    ("COc1ccccc1",            "C.C1CCCCC1"),   # anisole -> cyclohexane + methane
    ("CC(=O)C",               "CCC"),          # acetone -> propane
    ("CC(=O)[O-]",            "CC"),           # acetate -> ethane
    ("CC(N)=O",               "CC"),           # acetamide -> ethane
    ("[O-][N+](=O)c1ccccc1",  "C1CCCCC1"),     # nitrobenzene -> cyclohexane
]

RECONSTRUCTABLE = [s for s, _ in ANCHORS] + [
    "N[C@@H](Cc1ccccc1)C(=O)O",                     # phenylalanine
    "N[C@@H](CC(=O)O)C(=O)O",                        # aspartate
    "N[C@@H](Cc1ccc(cc1)[N+](=O)[O-])C(=O)O",        # 4-nitrophenylalanine
    "N[C@@H](Cc1ccc(O)cc1)C(=O)O",                   # tyrosine
    "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",              # tryptophan
    "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21",           # naringenin (flavanone)
    "COC1OC(CO)C(O)C(O)C1O",                         # methyl glucoside
]


@pytest.mark.parametrize("smiles,expected_ref", ANCHORS)
def test_reference_anchor(smiles, expected_ref):
    ref = build_reference(Chem.MolFromSmiles(smiles))
    assert _canon(ref.reference_smiles) == _canon(expected_ref)


@pytest.mark.parametrize("smiles", RECONSTRUCTABLE)
def test_reconstruction_invariant(smiles):
    ok, original, rebuilt = verify_reconstruction(Chem.MolFromSmiles(smiles))
    assert ok, f"{original} != {rebuilt}"


def test_anisole_reference_is_two_fragments():
    """A pendant group bonded only through a heteroatom detaches (decision: carbon-only)."""
    ref = build_reference(Chem.MolFromSmiles("COc1ccccc1"))
    assert ref.reference_smiles.count(".") == 1        # cyclohexane + methane


def test_heteroatom_transformation_records_connectivity():
    """A heteroatom feature carries the loci it connects (decision 1)."""
    ref = build_reference(Chem.MolFromSmiles("COc1ccccc1"))   # anisole ether O
    ethers = [h for h in ref.hetero if h.atomic_num == 8]
    assert len(ethers) == 1
    # the ether O connects two distinct carbon loci (ring carbon + methyl carbon)
    carbon_targets = [t for kind, t, _ in ethers[0].connections if kind == "C"]
    assert len(set(carbon_targets)) == 2
