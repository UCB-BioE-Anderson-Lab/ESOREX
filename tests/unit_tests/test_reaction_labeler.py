from rdkit.Chem import rdChemReactions
from esorex.reaction_preparation import prepare_reaction
from esorex.label_reaction import label_reaction


def test_label_reaction_matches_a_c_oh_site():
    """label_reaction should map the operator's atoms onto a genuine C-OH site and
    partition every atom into mapped / matched / unmatched.

    Asserts the semantic invariants rather than exact atom indices, which are an RDKit
    canonicalization detail and not part of the contract.
    """
    # Pre-mapped heavy-atom SMILES for OCC(C)CCCCO>>OCC(C)CCCC=O; the operator oxidizes
    # one terminal C-OH to C=O (prepare_reaction adds explicit-H maps).
    test_smiles = (
        "[OH:1][CH2:2][CH:3]([CH3:4])[CH2:5][CH2:6][CH2:7][CH2:8][OH:9]"
        ">>"
        "[OH:1][CH2:2][CH:3]([CH3:4])[CH2:5][CH2:6][CH2:7][CH:8]=[O:9]"
    )
    operator_smarts = "[C:1]([H])[O:2]([H])>>[C:1]=[O:2]"

    rxn = prepare_reaction(test_smiles)
    operator = rdChemReactions.ReactionFromSmarts(operator_smarts)
    result = label_reaction(rxn, operator)

    # mapped_atoms entries are (atom_index, operator_map_number)
    react_mapped = {mn: idx for idx, mn in result["mapped_atoms"][0]}
    prod_mapped = {mn: idx for idx, mn in result["mapped_atoms"][1]}

    # the operator maps exactly its two atoms (C:1, O:2) on each side
    assert set(react_mapped) == {1, 2}
    assert set(prod_mapped) == {1, 2}

    # op:1 is a carbon bonded to op:2, an oxygen: a real C-OH site
    reactant = rxn.GetReactants()[0]
    carbon = reactant.GetAtomWithIdx(react_mapped[1])
    oxygen = reactant.GetAtomWithIdx(react_mapped[2])
    assert carbon.GetSymbol() == "C"
    assert oxygen.GetSymbol() == "O"
    assert reactant.GetBondBetweenAtoms(react_mapped[1], react_mapped[2]) is not None

    # mapped / unmapped-matched / unmatched partition every atom, disjointly
    mols = (rxn.GetReactants()[0], rxn.GetProducts()[0])
    for side, mol in enumerate(mols):
        mapped = {idx for idx, _ in result["mapped_atoms"][side]}
        matched = set(result["unmapped_matched_atoms"][side])
        unmatched = set(result["unmatched_atoms"][side])
        assert mapped | matched | unmatched == set(range(mol.GetNumAtoms()))
        assert not (mapped & matched)
        assert not (mapped & unmatched)
        assert not (matched & unmatched)
