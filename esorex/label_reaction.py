from rdkit import Chem
from typing import Dict
import hashlib

def label_reaction(rxn, operator) -> Dict:
    """
    Performs a structural alignment between a chemical reaction and a reaction operator (RO), 
    identifying atom-level correspondences, resolving regiochemistry, and categorizing atoms 
    based on their involvement in the transformation.

    This function evaluates all possible combinations of substructure matches between the 
    reactant and product sides of the input reaction and the operator, selecting the pairing 
    that aligns best with the overall structure (i.e., the one whose unmatched portions are 
    identical). This process ensures that regiochemistry is correctly resolved.

    Atom mappings from the reaction operator are extracted and propagated onto the output 
    labels. Atoms are categorized into three sets:
      * Mapped atoms: atoms present in the operator with a map number; for these, the atom 
        indices and map numbers are recorded and mapped onto the reaction.
      * Unmapped matched atoms: atoms in the input reaction that are structurally matched by 
        the operator but lack a map number in the operator.
      * Unmatched atoms: atoms in the input reaction not participating in the operator (i.e., 
        not matched).

    Inputs:
      - rxn (rdkit.Chem.rdChemReactions.ChemicalReaction): The input reaction, which must 
        contain exactly one reactant and one product, both with explicit hydrogens.
      - operator (rdkit.Chem.rdChemReactions.ChemicalReaction): A complete reaction operator 
        with atom-mapped SMARTS that defines the transformation and atom mappings between 
        substrate and product.

    Returns:
      Dict with the following structure:
      {
          "rxn": rdkit.Chem.rdChemReactions.ChemicalReaction,  # The original input reaction
          "mapped_atoms": Tuple[List[Tuple[int, int]], List[Tuple[int, int]]],  # Atom index 
            and map number for reactant and product; for each, a list of (atom_idx, map_num)
          "unmapped_matched_atoms": Tuple[List[int], List[int]],  # Atom indices of structurally 
            matched but unmapped atoms, for reactant and product
          "unmatched_atoms": Tuple[List[int], List[int]],  # Atom indices not participating in 
            the reaction operator, for reactant and product
      }

    The returned structure encodes, for both reactant and product, the atom indices and map 
    numbers for matched atoms (mapped_atoms), as well as detailed breakdowns for unmapped-matched 
    and unmatched atom sets.
    """

    # Find all substructure matches of the operator in the reaction reactant and product
    rxn_reactant = rxn.GetReactants()[0]
    operator_reactant = operator.GetReactants()[0]
    reactant_matches = rxn_reactant.GetSubstructMatches(operator_reactant, uniquify=False)
    # print("reactant_matches", reactant_matches)

    rxn_product = rxn.GetProducts()[0]
    operator_product = operator.GetProducts()[0]
    product_matches = rxn_product.GetSubstructMatches(operator_product, uniquify=False)
    # print("product_matches", product_matches)

    # Generate reduced representations by removing matched atoms from the reaction
    # Note:  This assumes a complete operator and will not work on arbitrary other operators
    # It is designed to work with EVODEX-E, C, N.  See EVODEX publication for explanation of 'complete'.
    reduced_reactants = []
    for reactant_match in reactant_matches:
        reduced_reactant_smiles = _reduce_mol(rxn_reactant, reactant_match)
        reduced_reactants.append(reduced_reactant_smiles)

    reduced_products = []
    for product_match in product_matches:
        reduced_product_smiles = _reduce_mol(rxn_product, product_match)
        reduced_products.append(reduced_product_smiles)

    # Compare reduced structures to identify the pairing that relates subsrate and product
    final_reactant_match = None
    final_product_match = None
    for i, reactant_smiles in enumerate(reduced_reactants):
        for j, product_smiles in enumerate(reduced_products):
            if reactant_smiles == product_smiles:
                final_reactant_match = reactant_matches[i]
                final_product_match = product_matches[j]
                break

    # print("final_reactant_match", final_reactant_match)
    # print("final_product_match", final_product_match)

    # Classify atoms based on operator mapping
    mapped_atoms = ([], [])
    unmapped_matched_atoms = ([], [])
    unmatched_atoms = ([], [])

    # Guard clause: return early if no matches found
    if final_reactant_match is None or final_product_match is None:
        return {
            "rxn": rxn,
            "mapped_atoms": mapped_atoms,
            "unmapped_matched_atoms": unmapped_matched_atoms,
            "unmatched_atoms": unmatched_atoms,
        }

    # Filter the matched atoms into their bins
    for i, rxn_idx in enumerate(final_reactant_match):
        if operator_reactant.GetAtomWithIdx(i).GetAtomMapNum() > 0:
            mapped_atoms[0].append((rxn_idx, operator_reactant.GetAtomWithIdx(i).GetAtomMapNum()))
        else:
            unmapped_matched_atoms[0].append(rxn_idx)

    for i, rxn_idx in enumerate(final_product_match):
        if operator_product.GetAtomWithIdx(i).GetAtomMapNum() > 0:
            mapped_atoms[1].append((rxn_idx, operator_product.GetAtomWithIdx(i).GetAtomMapNum()))
        else:
            unmapped_matched_atoms[1].append(rxn_idx)

    # Atoms in neither mapped nor unmapped_matched_atoms sets are unmatched
    all_r_indices = set(range(rxn.GetReactants()[0].GetNumAtoms()))
    all_p_indices = set(range(rxn.GetProducts()[0].GetNumAtoms()))
    matched_r = set(idx for idx, _ in mapped_atoms[0]) | set(unmapped_matched_atoms[0])
    matched_p = set(idx for idx, _ in mapped_atoms[1]) | set(unmapped_matched_atoms[1])

    unmatched_atoms = (
        sorted(all_r_indices - matched_r),
        sorted(all_p_indices - matched_p),
    )

    return {
        "rxn": rxn,
        "mapped_atoms": mapped_atoms,
        "unmapped_matched_atoms": unmapped_matched_atoms,
        "unmatched_atoms": unmatched_atoms,
    }

def _reduce_mol(mol, remove_indices):
    rw_mol = Chem.RWMol()
    idx_map = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in remove_indices:
            new_idx = rw_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
            idx_map[idx] = new_idx
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in idx_map and a2 in idx_map:
            rw_mol.AddBond(idx_map[a1], idx_map[a2], bond.GetBondType())
    return Chem.MolToSmiles(rw_mol.GetMol(), canonical=True)


import os
import csv
from rdkit.Chem import rdChemReactions

_E_INDEX = None  # Lazy-loaded global

def label_evodex_operator(rxn):
    """
    Attempts to assign a matching EVODEX reaction operator to a given input reaction.

    This function operates directly on CSV data files from the EVODEX.1 distribution:
      - EVODEX-E_synthesis_subset.csv: contains SMIRKS and IDs of E-level operators
      - EVODEX-P_partial_reactions.csv: links E operators to their supporting partial reactions
      - EVODEX-F_unique_formulas.csv: maps partial reactions to formula difference identifiers (EVODEX-F)

    It follows a two-step process:
      1. **Formula Prefiltering**:
         Calculates the formula-level difference of the input reaction and selects all EVODEX-E operators
         associated with that formula change. This narrows the search space.
      2. **Structural Labeling**:
         Runs `label_reaction()` with each candidate operator, returning the first one that structurally
         matches the input reaction.

    Input:
        rxn (rdkit.Chem.rdChemReactions.ChemicalReaction):
            A preprocessed reaction with explicit hydrogens and one reactant/product.

    Returns:
        Dict: A dictionary containing:
          - "rxn": The original input reaction
          - "mapped_atoms", "unmapped_matched_atoms", "unmatched_atoms": from `label_reaction()`
          - "evodex_id": EVODEX ID of the matched E operator (e.g., "EVODEX.1-E1001")
          - "evodex_smirks": SMIRKS string of the matched operator
          - "evodex_f_id": Formula difference ID that matched this operator set

        Returns None if no matching operator is found.
    """
    global _E_INDEX
    if _E_INDEX is None:
        _E_INDEX = _build_f_to_e_index()

    f_key, formula_diff = _calculate_formula_diff(rxn)
    # print(f"Computed formula key: {f_key}")
    # if f_key in _E_INDEX:
    #     print(f"Matched F ID: {f_key}")
    #     print(f"E operators for this F: {[e['_id'] for e in _E_INDEX[f_key]]}")
    # else:
    #     print("No matching F ID found in index.")
    for e in _E_INDEX.get(f_key, []):
        # print(f"Trying E operator {e['_id']} with SMIRKS: {e['smirks']}")
        result = label_reaction(rxn, e['rxn'])
        if result['mapped_atoms'][0] or result['mapped_atoms'][1]:
            result['evodex_id'] = e['_id']
            result['evodex_smirks'] = e['smirks']
            result['evodex_f_id'] = e.get("evodex_f_id_str", None)
            result['evodex_f_hash'] = f_key
            result["formula_diff"] = formula_diff
            return result
    return None


def _build_f_to_e_index():
    data_path = os.path.join(os.path.dirname(__file__), "data")
    e_path = os.path.join(data_path, "EVODEX-E_synthesis_subset.csv")
    p_path = os.path.join(data_path, "EVODEX-P_partial_reactions.csv")
    f_path = os.path.join(data_path, "EVODEX-F_unique_formulas.csv")

    e_to_p = {}
    with open(e_path) as f:
        reader = csv.DictReader(f)
        e_rows = list(reader)
        # print(f"Loaded {len(e_rows)} E operators")
        for row in e_rows:
            e_id = row["id"]
            sources = row["sources"].split(",")
            e_entry = {
                "_id": e_id,
                "smirks": row["smirks"],
                "sources": sources,
                "rxn": rdChemReactions.ReactionFromSmarts(row["smirks"]),
            }
            e_to_p[e_id] = e_entry

    p_to_f = {}
    with open(f_path) as f:
        reader = csv.DictReader(f)
        f_rows = list(reader)
        # print(f"Loaded {len(f_rows)} F entries")
        for row in f_rows:
            formula_hash = row["original_id"]
            for p_id in row["sources"].split(","):
                p_to_f[p_id] = formula_hash

    f_to_e = {}
    count_links = 0
    for e_id, e_entry in e_to_p.items():
        for p_id in e_entry["sources"]:
            f_id = p_to_f.get(p_id)
            if f_id:
                f_to_e.setdefault(f_id, []).append(e_entry)
                e_entry["evodex_f_id_str"] = row["id"]
                count_links += 1

    # print(f"Linked {count_links} E-to-F connections across {len(f_to_e)} unique F entries")
    return f_to_e


def _calculate_formula_diff(rxn):
    """
    Computes the atom-type formula difference (products - reactants) for a reaction.
    Returns a tuple: (formula_hash, atom_diff) where:
      - formula_hash is a SHA256 hash of the canonicalized atom_diff
      - atom_diff is a dictionary like {'C': 1, 'H': -2}
    """
    atom_diff = {}
    reactant_atoms = {}
    for reactant in rxn.GetReactants():
        for atom in reactant.GetAtoms():
            atom_type = atom.GetSymbol()
            if atom_type == 'At':
                atom_type = 'H'
            reactant_atoms[atom_type] = reactant_atoms.get(atom_type, 0) + 1

    product_atoms = {}
    for product in rxn.GetProducts():
        for atom in product.GetAtoms():
            atom_type = atom.GetSymbol()
            if atom_type == 'At':
                atom_type = 'H'
            product_atoms[atom_type] = product_atoms.get(atom_type, 0) + 1

    all_atoms = set(reactant_atoms.keys()).union(set(product_atoms.keys()))
    for atom in all_atoms:
        diff = product_atoms.get(atom, 0) - reactant_atoms.get(atom, 0)
        if diff != 0:
            atom_diff[atom] = diff

    formula_frozenset = frozenset(atom_diff.items())
    formula_hash = hashlib.sha256(str(formula_frozenset).encode()).hexdigest()

    return formula_hash, atom_diff

if __name__ == "__main__":
    test_smiles = "OCC(C)CCCCO>>OCC(C)CCCC=O"
    operator_smarts = "[C:1]([H])[O:2]([H])>>[C:1]=[O:2]"

    from rdkit.Chem import rdChemReactions
    from esorex.reaction_preparation import prepare_reaction
    from pprint import pprint
    rxn = prepare_reaction(test_smiles)

    print("Prepared Reaction (SMILES with IDs):")
    print(rdChemReactions.ReactionToSmiles(rxn))

    operator = rdChemReactions.ReactionFromSmarts(operator_smarts)

    result = label_reaction(rxn, operator)
    pprint(result)

    print("\nTrying EVODEX operator match...")
    evodex_result = label_evodex_operator(rxn)
    pprint(evodex_result)
