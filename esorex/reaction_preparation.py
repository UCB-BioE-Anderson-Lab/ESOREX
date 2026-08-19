import re

from rdkit import Chem
from rdkit.Chem import AllChem
from evodex.hydrogen_mapper import map_hydrogens_in_reaction

# Matches [H] or [H:N], explicit H atoms as separate nodes in the mol graph.
# Does NOT match [OH:1], [CH3:1] etc. (H-count inside a heavy-atom bracket).
_EXPLICIT_H_RE = re.compile(r'\[H(?::\d+)?\]')


def _parse_mol(smarts_smi: str):
    """Parse one molecule from evodex H-mapped output and ensure all Hs are explicit.

    evodex returns SMARTS-style notation ([#6:n], aromatic bonds as colons).
    MolFromSmarts does not calculate implicit valence, so SanitizeMol is required
    before AddHs can add any missing hydrogens (e.g. Hs on aromatic ring carbons
    that evodex leaves implicit because the input used lowercase-c notation).

    MolFromSmarts yields a *query* molecule whose atoms are QueryAtoms; RDKit
    never perceives atom-level aromaticity on such a molecule, so an aromatic
    ring comes back with aromatic *bonds* but non-aromatic *atoms*.  Downstream
    the sigma/pi partition reads atom.GetIsAromatic() and would see no pi system
    at all (see the pi-system detection in the featurizer).  Round-tripping
    through canonical SMILES converts the query molecule into an ordinary one
    with full ring/aromaticity/hybridisation perception; removeHs=False keeps
    every explicit H (mapped and unmapped) as a graph atom so the downstream
    H-map assignment still sees them.
    """
    mol = Chem.MolFromSmarts(smarts_smi)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    plain = Chem.MolFromSmiles(Chem.MolToSmiles(mol), params)
    # Fall back to the query molecule only if the round-trip somehow fails to
    # parse, better a mol with weak aromaticity than no reaction at all.
    return plain if plain is not None else mol


def _assign_missing_h_maps(reactant_mols, product_mols):
    """Assign map numbers to Hs that AddHs added but evodex never saw.

    evodex only maps Hs that appear in the input SMILES.  Aromatic ring Hs and
    other implicit Hs that SanitizeMol+AddHs materialises afterward arrive with
    map number 0.

    For each heavy atom with map number M, we map min(r_count, p_count) Hs,
    where the counts are the number of unmapped Hs on M in the reactant and
    product.  This covers two cases:
      - Spectator atoms (r == p): all unmapped Hs are paired and mapped.
      - Active-site atoms (r != p): the Hs that persist in the product
        (min count) are mapped; the ones that leave (the excess) stay gray.
    """
    def unmapped_h_counts(mols):
        counts = {}
        for mol in mols:
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1 and atom.GetAtomMapNum() == 0:
                    for nb in atom.GetNeighbors():
                        if nb.GetAtomicNum() != 1:
                            pm = nb.GetAtomMapNum()
                            # A hydrogen on an untracked (map 0) parent stays
                            # untracked; never mint a map for it, otherwise a
                            # reaction with no heavy-atom maps would acquire
                            # spurious hydrogen maps and escape validation.
                            if pm == 0:
                                continue
                            counts[pm] = counts.get(pm, 0) + 1
        return counts

    r_counts = unmapped_h_counts(reactant_mols)
    p_counts = unmapped_h_counts(product_mols)

    # Number of Hs to map per parent atom = how many persist into the product.
    to_map = {
        pm: min(n, p_counts.get(pm, 0))
        for pm, n in r_counts.items()
        if min(n, p_counts.get(pm, 0)) > 0
    }

    if not to_map:
        return

    all_maps = [
        a.GetAtomMapNum()
        for mols in (reactant_mols, product_mols)
        for mol in mols
        for a in mol.GetAtoms()
        if a.GetAtomMapNum() != 0
    ]
    # If nothing is mapped, there is no basis for numbering; leave assignment to
    # the caller's validation (_validate_atom_maps raises a clear R1 error).
    next_map = (max(all_maps) + 1) if all_maps else 1

    assignment = {}  # parent_map → [new_map_nums]
    for pm in sorted(to_map):
        n = to_map[pm]
        assignment[pm] = list(range(next_map, next_map + n))
        next_map += n

    for mols in (reactant_mols, product_mols):
        for mol in mols:
            cursor = {pm: 0 for pm in assignment}
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1 and atom.GetAtomMapNum() == 0:
                    for nb in atom.GetNeighbors():
                        if nb.GetAtomicNum() != 1:
                            pm = nb.GetAtomMapNum()
                            if pm in assignment and cursor[pm] < len(assignment[pm]):
                                atom.SetAtomMapNum(assignment[pm][cursor[pm]])
                                cursor[pm] += 1


def _validate_atom_maps(reactant_mols, product_mols):
    """Enforce that the reaction's non-zero atom maps are well-formed.

    Three rules, each of which makes an atom-mapped reaction uninterpretable if
    broken:

      R1  at least one atom carries a non-zero map, a reaction with no maps
          cannot be tracked from reactant to product at all;
      R2  no map number appears on two atoms on the same side, a duplicate
          makes the reactant→product correspondence ambiguous;
      R3  every non-zero map appears on both sides, an unpaired map asserts a
          correspondence that does not exist.

    evodex's ``map_hydrogens_in_reaction`` already enforces all three for heavy
    atoms, but only on its own path; the explicit-[H] bypass skips it entirely.
    Applied here over *every* atom (heavy and hydrogen) these rules cover the
    bypass path as well, including hydrogen maps the caller supplied.  Atoms
    consumed by the reaction carry map 0 and are ignored.
    """
    def side_counts(mols):
        counts: dict[int, int] = {}
        for mol in mols:
            for atom in mol.GetAtoms():
                m = atom.GetAtomMapNum()
                if m:
                    counts[m] = counts.get(m, 0) + 1
        return counts

    r, p = side_counts(reactant_mols), side_counts(product_mols)

    # R1, no maps at all
    if not r and not p:
        raise ValueError(
            "No atom maps in the reaction: at least one atom must carry a "
            "non-zero map number so the reaction can be tracked."
        )

    # R2, duplicate map on a single side
    dup_r = sorted(m for m, n in r.items() if n > 1)
    dup_p = sorted(m for m, n in p.items() if n > 1)
    if dup_r or dup_p:
        raise ValueError(
            f"Duplicate atom map number(s): reactant={dup_r}, product={dup_p}. "
            "A map number must identify exactly one atom on each side."
        )

    # R3, non-zero map unpaired across the reaction
    r_only, p_only = sorted(set(r) - set(p)), sorted(set(p) - set(r))
    if r_only or p_only:
        raise ValueError(
            "Mismatch between reactant and product atom maps: "
            f"reactant-only={r_only}, product-only={p_only}. "
            "A mapped atom must have a counterpart on the other side; "
            "atoms consumed by the reaction should be left unmapped (map 0)."
        )


def prepare_reaction(reaction_smiles: str):
    """
    Prepares an RDKit ChemicalReaction object from a SMILES string.

    - Calls evodex to assign H map numbers to the Hs it can identify.
    - Parses each fragment with SanitizeMol+AddHs to materialise any Hs
      that were implicit in the input (e.g. aromatic ring Hs).
    - Assigns fresh, consistent map numbers to those newly-explicit spectator Hs.

    Args:
        reaction_smiles (str): A reaction SMILES string of the form 'reactants>>products'.

    Returns:
        rdkit.Chem.rdChemReactions.ChemicalReaction with all Hs explicit and mapped.
    """
    if ">>" not in reaction_smiles:
        raise ValueError(f"Invalid reaction SMILES: must use >> notation, got: {reaction_smiles}")

    # If explicit H atoms are already present (e.g. re-running the output of this
    # function, or an input with pre-mapped Hs), skip evodex, it rejects such
    # inputs.  The Hs are already in the SMILES; _parse_mol and
    # _assign_missing_h_maps handle the rest.
    if _EXPLICIT_H_RE.search(reaction_smiles):
        hydrogen_mapped_smiles = reaction_smiles
    else:
        hydrogen_mapped_smiles = map_hydrogens_in_reaction(reaction_smiles)

    reactant_smiles, product_smiles = hydrogen_mapped_smiles.split(">>")
    reactant_mols = [_parse_mol(smi) for smi in reactant_smiles.split('.') if smi]
    product_mols  = [_parse_mol(smi) for smi in product_smiles.split('.') if smi]

    if any(mol is None for mol in reactant_mols + product_mols):
        raise ValueError(f"Could not parse hydrogen-mapped reaction: {hydrogen_mapped_smiles}")

    _assign_missing_h_maps(reactant_mols, product_mols)
    _validate_atom_maps(reactant_mols, product_mols)

    mapped_reaction = AllChem.ChemicalReaction()
    for mol in reactant_mols:
        mapped_reaction.AddReactantTemplate(mol)
    for mol in product_mols:
        mapped_reaction.AddProductTemplate(mol)

    return mapped_reaction


if __name__ == "__main__":
    from rdkit.Chem import rdChemReactions

    test_smiles = "[C:1][Cl:2]>>[C:1][Cl:2]"
    rxn = prepare_reaction(test_smiles)
    print("Prepared Reaction:")
    print(rdChemReactions.ReactionToSmiles(rxn))

    test_smiles = "[CH3:1][C:2]([CH3:3])([CH2:4][O:5][P:6](=[O:7])([OH:8])[O:9][P:10](=[O:11])([OH:12])[O:13][CH2:14][C@H:15]1[O:16][C@@H:17]([n:18]2[cH:19][n:20][c:21]3[c:22]([NH2:23])[n:24][cH:25][n:26][c:27]23)[C@H:28]([OH:29])[C@@H:30]1[O:31][P:32](=[O:33])([OH:34])[OH:35])[C@@H:36]([OH:37])[C:38](=[O:39])[NH:40][CH2:41][CH2:42][C:43](=[O:44])[NH:45][CH2:46][CH2:47][S:48][C:49](=[O:50])/[CH:51]=[CH:52]/[c:53]1[cH:54][cH:55][c:56]([OH:57])[c:58]([OH:59])[cH:60]1.[CH3:61][S+:62]([CH2:63][CH2:64][C@H:65]([NH2:66])[C:67](=[O:68])[OH:69])[CH2:70][C@H:71]1[O:72][C@@H:73]([n:74]2[cH:75][n:76][c:77]3[c:78]([NH2:79])[n:80][cH:81][n:82][c:83]23)[C@H:84]([OH:85])[C@@H:86]1[OH:87]>>[CH3:1][C:2]([CH3:3])([CH2:4][O:5][P:6](=[O:7])([OH:8])[O:9][P:10](=[O:11])([OH:12])[O:13][CH2:14][C@H:15]1[O:16][C@@H:17]([n:18]2[cH:19][n:20][c:21]3[c:22]([NH2:23])[n:24][cH:25][n:26][c:27]23)[C@H:28]([OH:29])[C@@H:30]1[O:31][P:32](=[O:33])([OH:34])[OH:35])[C@@H:36]([OH:37])[C:38](=[O:39])[NH:40][CH2:41][CH2:42][C:43](=[O:44])[NH:45][CH2:46][CH2:47][S:48][C:49](=[O:50])/[CH:51]=[CH:52]/[c:53]1[cH:54][cH:55][c:56]([OH:57])[c:58]([O:59][CH3:61])[cH:60]1.[H+].[S:62]([CH2:63][CH2:64][C@H:65]([NH2:66])[C:67](=[O:68])[OH:69])[CH2:70][C@H:71]1[O:72][C@@H:73]([n:74]2[cH:75][n:76][c:77]3[c:78]([NH2:79])[n:80][cH:81][n:82][c:83]23)[C@H:84]([OH:85])[C@@H:86]1[OH:87]"
    rxn = prepare_reaction(test_smiles)
    print("Prepared Reaction:")
    print(rdChemReactions.ReactionToSmiles(rxn))
