"""
Snapshot tests for esorex.reaction_preparation.prepare_reaction.

Each scenario has a golden SMILES string produced by the algorithm and validated
by a human reviewer via reports/prepare_reaction.html.  These tests exist to
catch regressions, if the output changes, a test fails and the developer must
verify the new chemistry in the report before updating the golden string.

Color key in the report drawings:
  Teal, heavy atom with map number
  Blue, H atom with map number (spectator or transferred, tracked by evodex
           or assigned by _assign_missing_h_maps)
  Gray, H atom without map number (consumed by the reaction, no product
           counterpart on the same heavy atom)

prepare_reaction(reaction_smiles: str) -> ChemicalReaction
  - Validates >> notation and heavy-atom map numbers via evodex
  - Calls evodex.map_hydrogens_in_reaction to assign H map numbers
  - Parses with SanitizeMol+AddHs to materialise implicit Hs (e.g. ring Hs)
  - Assigns map numbers to remaining unmapped Hs via _assign_missing_h_maps
"""

import pytest
from rdkit.Chem import rdChemReactions

from esorex.reaction_preparation import prepare_reaction


def _smiles(rxn_smiles: str) -> str:
    return rdChemReactions.ReactionToSmiles(prepare_reaction(rxn_smiles))


# ---------------------------------------------------------------------------
# Golden SMILES, validated by reviewing reports/prepare_reaction.html
# ---------------------------------------------------------------------------

ETHANOL_OX = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
ETHANOL_OX_GOLDEN = (
    "[H][C:2]([C:1]([H:4])([H:5])[H:6])([O:3][H])[H:7]"
    ">>"
    "[C:1]([C:2](=[O:3])[H:7])([H:4])([H:5])[H:6]"
)

MULTI = "[CH3:1][CH2:2][OH:3].[NH3:4]>>[CH3:1][CH:2]=[O:3].[NH4+:4]"
MULTI_GOLDEN = (
    "[H][C:2]([C:1]([H:5])([H:6])[H:7])([O:3][H])[H:8]"
    ".[N:4]([H:11])([H:12])[H:13]"
    ">>"
    "[C:1]([C:2](=[O:3])[H:8])([H:5])([H:6])[H:7]"
    ".[H][N+:4]([H:11])([H:12])[H:13]"
)

DIOL = "[OH:1][CH2:2][CH2:3][OH:4]>>[O:1]=[CH:2][CH2:3][OH:4]"
DIOL_GOLDEN = (
    "[H][O:1][C:2]([H])([C:3]([O:4][H:10])([H:8])[H:9])[H:6]"
    ">>"
    "[O:1]=[C:2]([C:3]([O:4][H:10])([H:8])[H:9])[H:6]"
)

METHANOL_OX = "[CH3:1][OH:2]>>[CH2:1]=[O:2]"
METHANOL_OX_GOLDEN = (
    "[H][C:1]([O:2][H])([H:3])[H:4]"
    ">>"
    "[C:1](=[O:2])([H:3])[H:4]"
)

BENZYL = (
    "[c:1]1[c:7][c:8][c:9][c:10][c:11]1[CH2:2][OH:3]"
    ">>"
    "[c:1]1[c:7][c:8][c:9][c:10][c:11]1[CH:2]=[O:3]"
)
BENZYL_GOLDEN = (
    "[H][C:2]([O:3][H])([c:11]1[c:1]([H:13])[c:7]([H:14])"
    "[c:8]([H:15])[c:9]([H:16])[c:10]1[H:17])[H:12]"
    ">>"
    "[c:1]1([H:13])[c:7]([H:14])[c:8]([H:15])[c:9]([H:16])"
    "[c:10]([H:17])[c:11]1[C:2](=[O:3])[H:12]"
)

SMIRKS_HASH = "[#6:1][#6:2][#8:3]>>[#6:1][#6:2]=[#8:3]"
SMIRKS_HASH_GOLDEN = (
    "[H][C:2]([C:1]([H:4])([H:5])[H:6])([O:3][H])[H:7]"
    ">>"
    "[C:1]([C:2](=[O:3])[H:7])([H:4])([H:5])[H:6]"
)

LARGE_MAPS = "[CH3:100][CH2:200][OH:300]>>[CH3:100][CH:200]=[O:300]"
LARGE_MAPS_GOLDEN = (
    "[H][C:200]([C:100]([H:301])([H:302])[H:303])([O:300][H])[H:304]"
    ">>"
    "[C:100]([C:200](=[O:300])[H:304])([H:301])([H:302])[H:303]"
)

# O-methylation: product-only CH3 (cofactor-derived, map 0) gets gray Hs only.
# The incoming methyl carbon serialises as a plain aliphatic `C`, not a bracketed
# `[C]`: prepare_reaction now de-queries the molecule (round-trip through SMILES)
# so it is an ordinary RDKit atom rather than a SMARTS QueryAtom.  Chemically
# identical to the pre-de-query output (both canonicalise to C[O:2][CH3:1]).
OMETHYL = "[CH3:1][O:2][H]>>[CH3:1][O:2][CH3]"
OMETHYL_GOLDEN = (
    "[H][O:2][C:1]([H:3])([H:4])[H:5]"
    ">>"
    "[H]C([H])([H])[O:2][C:1]([H:3])([H:4])[H:5]"
)


def test_standard_golden():
    assert _smiles(ETHANOL_OX) == ETHANOL_OX_GOLDEN

def test_multi_fragment_golden():
    assert _smiles(MULTI) == MULTI_GOLDEN

def test_diol_golden():
    assert _smiles(DIOL) == DIOL_GOLDEN

def test_methanol_golden():
    assert _smiles(METHANOL_OX) == METHANOL_OX_GOLDEN

def test_benzyl_golden():
    assert _smiles(BENZYL) == BENZYL_GOLDEN

def test_smirks_hash_golden():
    assert _smiles(SMIRKS_HASH) == SMIRKS_HASH_GOLDEN

def test_large_maps_golden():
    assert _smiles(LARGE_MAPS) == LARGE_MAPS_GOLDEN

def test_omethyl_golden():
    assert _smiles(OMETHYL) == OMETHYL_GOLDEN


# ---------------------------------------------------------------------------
# Aromaticity, prepare_reaction must return an ordinary, fully-perceived
# molecule, NOT a SMARTS query molecule.  On a query molecule RDKit leaves
# ring ATOMS non-aromatic while their BONDS are aromatic; the sigma/pi
# partition downstream reads atom.GetIsAromatic() and would then see no pi
# system in a benzene ring at all.  These tests pin the invariant directly.
# ---------------------------------------------------------------------------

# 3-pyridylmethanol -> pyridine-3-carbaldehyde: heteroaromatic ring so the
# ring N (and its lone pair) must also be perceived as aromatic.
PYRIDYL = (
    "[c:1]1[c:7][c:8][n:9][c:10][c:11]1[CH2:2][OH:3]"
    ">>"
    "[c:1]1[c:7][c:8][n:9][c:10][c:11]1[CH:2]=[O:3]"
)


def _reactant(rxn_smiles: str):
    # Keep the reaction alive: GetReactants() returns a view that is invalidated
    # once the parent ChemicalReaction is garbage-collected.
    rxn = prepare_reaction(rxn_smiles)
    return rxn.GetReactants()[0]


@pytest.mark.parametrize("rxn_smiles,n_aromatic", [
    (BENZYL, 6),    # benzene ring: all six carbons aromatic
    (PYRIDYL, 6),   # pyridine ring: five carbons + one nitrogen aromatic
])
def test_ring_atoms_are_aromatic(rxn_smiles, n_aromatic):
    mol = _reactant(rxn_smiles)
    assert sum(a.GetIsAromatic() for a in mol.GetAtoms()) == n_aromatic


@pytest.mark.parametrize("rxn_smiles", [
    ETHANOL_OX, MULTI, DIOL, METHANOL_OX, SMIRKS_HASH, LARGE_MAPS, BENZYL, PYRIDYL,
])
def test_atom_and_bond_aromaticity_consistent(rxn_smiles):
    """No atom may disagree with its bonds about aromaticity, the exact
    corruption a returned SMARTS query molecule produces."""
    rxn = prepare_reaction(rxn_smiles)
    mols = ([rxn.GetReactants()[i] for i in range(rxn.GetNumReactantTemplates())]
            + [rxn.GetProducts()[i] for i in range(rxn.GetNumProductTemplates())])
    for mol in mols:
        for atom in mol.GetAtoms():
            bonds_aromatic = any(b.GetIsAromatic() for b in atom.GetBonds())
            assert atom.GetIsAromatic() == bonds_aromatic, (
                f"atom {atom.GetIdx()} ({atom.GetSymbol()}): "
                f"GetIsAromatic()={atom.GetIsAromatic()} but "
                f"bonds_aromatic={bonds_aromatic}"
            )


# ---------------------------------------------------------------------------
# Idempotency, running the output back through must be a no-op
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    ETHANOL_OX, MULTI, DIOL, METHANOL_OX, SMIRKS_HASH, LARGE_MAPS, BENZYL, PYRIDYL,
    # OMETHYL excluded: unmapped product C serialises as [C] then C on second pass
    # (RDKit bracket normalisation), which is harmless but breaks exact string equality.
])
def test_idempotent(smiles):
    out1 = _smiles(smiles)
    out2 = _smiles(out1)
    assert out2 == out1


# ---------------------------------------------------------------------------
# Invalid inputs, must raise ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,match", [
    ("CC[CH2:1][OH:2]>>CC[CH:1]=[O:2]",                             "valid map number"),
    ("[CH3:0][CH2:1][OH:2]>>[CH3:0][CH:1]=[O:2]",                   "valid map number"),
    ("[C@@H:1]([CH3:2])([OH:3])[CH3:4]>>[C:1](=[O:3])[CH3:2]",      "Mismatch"),
    ("[HO:1][CH2:2]>>[O:1]=[CH:2]",                                  "Invalid reaction SMILES"),
    ("[CH3:1][CH2:2][OH:3]>[H:4]>[CH3:1][CH:2]=[O:3]",              "Invalid reaction SMILES"),
])
def test_invalid_inputs_raise_valueerror(smiles, match):
    with pytest.raises(ValueError, match=match):
        prepare_reaction(smiles)


# ---------------------------------------------------------------------------
# Atom-map well-formedness, the three rules that make an atom-mapped reaction
# interpretable.  Each rule is tested on BOTH the evodex path (no explicit [H],
# where evodex.map_hydrogens_in_reaction enforces it for heavy atoms) AND the
# explicit-[H] bypass path (which skips evodex; esorex._validate_atom_maps is
# what enforces the rule there, over every atom incl. hydrogens).
# ---------------------------------------------------------------------------

# Rule 1, there are no atom maps at all on either side of the reaction.
@pytest.mark.parametrize("smiles", [
    "CCO>>CC=O",        # evodex path: rejected as an unmapped heavy atom
    "[H]OCC>>[H]OCC",   # bypass path: rejected as "No atom maps in the reaction"
])
def test_rule1_no_atom_maps_rejected(smiles):
    with pytest.raises(ValueError):
        prepare_reaction(smiles)


# Rule 2, two atoms carry the same non-zero atom map on one side.  Exercised
# with the duplicate on the reactant side and on the product side.
@pytest.mark.parametrize("smiles", [
    "[CH3:1][CH2:1][OH:3]>>[CH3:1][CH:1]=[O:3]",               # dup on reactant, evodex path
    "[CH3:1][CH2:1][OH:3][H:9]>>[CH3:1][CH:1]([H:9])=[O:3]",   # dup on reactant, bypass path
    "[CH3:1][CH2:2]([H:8])[OH:3]>>[CH3:1][CH:2]([H:8])=[O:2]", # dup on product, bypass path
])
def test_rule2_duplicate_map_rejected(smiles):
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        prepare_reaction(smiles)


# Rule 3, a non-zero atom map is unpaired across the reaction (present on one
# side, absent on the other).  Exercised reactant-only and product-only.
@pytest.mark.parametrize("smiles", [
    "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:4]=[O:3]",               # reactant-only (map 2 vs 4), evodex
    "[CH3:1][CH2:2]([H:10])[OH:3]>>[CH3:1][CH:2]=[O:3]",       # reactant-only H:10, bypass path
    "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]([H:10])=[O:3]",       # product-only H:10, bypass path
])
def test_rule3_unpaired_map_rejected(smiles):
    with pytest.raises(ValueError, match="[Mm]ismatch"):
        prepare_reaction(smiles)


def test_balanced_premapped_hydrogen_accepted():
    """A pre-mapped hydrogen is legitimate when it is balanced across the
    reaction, nothing is fundamentally wrong with mapping an H.  Here H:10 is
    the benzylic H that persists from the CH2 alcohol into the CHO aldehyde."""
    rxn = prepare_reaction("[CH3:1][CH2:2]([H:10])[OH:3]>>[CH3:1][C:2]([H:10])=[O:3]")

    def hmaps(mol):
        return {a.GetAtomMapNum() for a in mol.GetAtoms()
                if a.GetAtomicNum() == 1 and a.GetAtomMapNum()}

    r_h = hmaps(rxn.GetReactants()[0])
    p_h = hmaps(rxn.GetProducts()[0])
    assert 10 in r_h and 10 in p_h            # the caller's map is preserved, both sides

    def allmaps(get, n):
        return {a.GetAtomMapNum() for i in range(n) for a in get()[i].GetAtoms()
                if a.GetAtomMapNum()}
    r = allmaps(rxn.GetReactants, rxn.GetNumReactantTemplates())
    p = allmaps(rxn.GetProducts, rxn.GetNumProductTemplates())
    assert r == p                              # every non-zero map is balanced
