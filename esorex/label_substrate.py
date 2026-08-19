from rdkit import Chem
from rdkit.Chem import rdChemReactions
from typing import List, Dict, Tuple

def label_substrate(substrate: Chem.Mol, operator: rdChemReactions.ChemicalReaction) -> List[Dict[str, List]]:
    """
    Identifies and annotates substructure matches of a reaction operator’s reactant pattern 
    within a single substrate molecule.

    For each match found between the operator and the substrate, atoms are classified into 
    three categories:
      * Mapped atoms: atoms that match mapped atoms in the operator (i.e., the operator 
        atom has a map number). Each is recorded as a (atom_idx, map_num) tuple.
      * Unmapped matched atoms: atoms that are matched structurally to the operator but do 
        not have a map number in the operator.
      * Unmatched atoms: atoms in the substrate that are not part of the matched substructure.

    This method supports multiple matchings of the operator pattern within the same molecule 
    and returns labeling results for all of them.

    Inputs:
      - substrate (rdkit.Chem.Mol): The substrate molecule (with explicit hydrogens) to be 
        evaluated for reaction operator matches.
      - operator (rdkit.Chem.rdChemReactions.ChemicalReaction): A reaction operator containing 
        at least one reactant side with atom-mapped SMARTS used to identify possible substrate 
        matching locations.

    Returns:
      List[Dict[str, List]]: A list where each element corresponds to one match, and contains:
        {
            "mapped_atoms": List[Tuple[int, int]],            # (atom_idx, atom_map_num)
            "unmapped_matched_atoms": List[int],              # atom indices matched but not mapped
            "unmatched_atoms": List[int],                     # atom indices not matched
        }

    This structure allows downstream tools to build substrate trees, analyze participation 
    in chemical transformations, and filter by matched or excluded structural motifs.
    """
    operator_reactant = operator.GetReactants()[0]
    matches = substrate.GetSubstructMatches(operator_reactant, uniquify=False)

    results = []
    all_indices = set(range(substrate.GetNumAtoms()))

    for match in matches:
        mapped_atoms = []
        unmapped_matched_atoms = []
        matched_indices = set()

        for i, atom_idx in enumerate(match):
            atom_map_num = operator_reactant.GetAtomWithIdx(i).GetAtomMapNum()
            if atom_map_num > 0:
                mapped_atoms.append((atom_idx, atom_map_num))
            else:
                unmapped_matched_atoms.append(atom_idx)
            matched_indices.add(atom_idx)

        unmatched_atoms = sorted(all_indices - matched_indices)

        results.append({
            "mapped_atoms": mapped_atoms,
            "unmapped_matched_atoms": unmapped_matched_atoms,
            "unmatched_atoms": unmatched_atoms,
        })

    return results

if __name__ == "__main__":
    substrate_smiles = "OCC(C)CCCCO"
    operator_smarts = "[C:1]([H])[O:2]([H])>>[C:1]=[O:2]"

    substrate = Chem.AddHs(Chem.MolFromSmiles(substrate_smiles))
    operator = rdChemReactions.ReactionFromSmarts(operator_smarts)

    labeled_substrates = label_substrate(substrate, operator)

    from pprint import pprint
    pprint(labeled_substrates)