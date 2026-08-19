"""FucTIII regioselectivity: 2 positives (LacNAc + LNB) + inferred negatives.

Training set (build):
  Positive:  LacNAc GlcNAc-O3 and LNB GlcNAc-O4 (natural fucosylation sites)
  Negatives: all other B-level C-OH sites on each natural substrate

Test set (build_test):
  21-compound panel from the paper, for each of which we predict site preference
  and compare the top-predicted site against the known fucosylation site (from
  product SMILES). Measured activity is the normalized kcat/KM score.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rdkit import Chem

from esorex.reaction_preparation import prepare_reaction
from esorex.generate_mechanistic_tree import generate_mechanistic_tree
from esorex.generate_mechanistic_tree import (
    _operator_map_numbers,
    _collect_operators_by_level,
)

import re
from rdkit.Chem import AllChem
from esorex.label_substrate import label_substrate

_OH_MAP_RE = re.compile(r'\[OH:(\d+)\]')
_AZIDE = (
    "[O:700][CH2:701][CH2:702][CH2:703][CH2:704][CH2:705][CH2:706]"
    "[N:710]=[N+:711]=[N-:712]"
)

def _explicit_h_reaction(rxn_smiles: str) -> str:
    reactant, product = rxn_smiles.split(">>")
    reactant = _OH_MAP_RE.sub(r'[O:\1][H]', reactant)
    return f"{reactant}>>{product}"

def _strip_azide(smiles: str) -> str:
    return smiles.replace(_AZIDE, "[OH:700]")

def actual_fucosylation_site(prod_smiles: str):
    if not prod_smiles:
        return None
    mol = Chem.MolFromSmiles(prod_smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.GetAtomMapNum() == 0:
            continue
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 6 and nb.GetAtomMapNum() == 0:
                return atom.GetAtomMapNum()
    return None

def _all_b_level_sites(mol, b_level_ops, a_map_nums):
    sites = []
    seen: set = set()
    for op_smirks in b_level_ops:
        op_rxn = AllChem.ReactionFromSmarts(op_smirks)
        if op_rxn is None:
            continue
        for label in label_substrate(mol, op_rxn):
            if not label["mapped_atoms"]:
                continue
            a_atoms = [(idx, mn) for idx, mn in label["mapped_atoms"] if mn in a_map_nums]
            if not a_atoms:
                continue
            o_idx, o_map = a_atoms[0]
            if o_idx in seen:
                continue
            seen.add(o_idx)
            passenger_indices = [
                a.GetIdx() for a in mol.GetAtoms()
                if a.GetIdx() != o_idx and a.GetAtomicNum() > 1
            ]
            sites.append((o_idx, o_map, passenger_indices))
    return sites

def _natural_substrates():
    from reports.fuctiii_compound_data import COMPOUNDS
    c11 = COMPOUNDS[11]
    c19 = COMPOUNDS[19]
    return {
        "LacNAc": {
            "label": "LacNAc (Type II, Galβ1-4GlcNAc)",
            "sub":   _strip_azide(c11["substrate_smiles"]),
            "prod":  _strip_azide(c11["product_smiles"]),
            "activity": c11["kinetics"]["normalized_score"],
        },
        "LNB": {
            "label": "LNB (Type I, Galβ1-3GlcNAc)",
            "sub":   _strip_azide(c19["substrate_smiles"]),
            "prod":  _strip_azide(c19["product_smiles"]),
            "activity": c19["kinetics"]["normalized_score"],
        },
    }

METADATA = {
    "description": (
        "FucTIII (H. pylori α1,3/4-fucosyltransferase) regioselectivity inference. "
        "Training: 2 positives (LacNAc O3, LNB O4) + inferred negatives. "
        "Test: 21-compound panel with known reactive site and kcat/KM."
    ),
    "citation": "Rabbani et al., Biochemistry 2005; Rasko et al., J Biol Chem 2000",
    "n_expected": 14,
    "target_type": "continuous",
    "experiment_type": 2,
}


def _get_b_ops():
    """Derive B-level operators from LacNAc natural reaction."""
    natural = _natural_substrates()
    lacnac = natural["LacNAc"]
    rxn_smi = _explicit_h_reaction(f"{lacnac['sub']}>>{lacnac['prod']}")
    rxn = prepare_reaction(rxn_smi)
    trees_mech = generate_mechanistic_tree([rxn])
    tree = trees_mech[0]
    op_map_nums = _operator_map_numbers(tree.smirks)
    b_ops = _collect_operators_by_level(tree).get("B", [])
    return b_ops, op_map_nums


# ── energetic model (current) ───────────────────────────────────────────────────
# Regioselectivity as the energetic specificity model: each candidate C-OH site is one
# (mol, core={the O}, energy) example.  The reactive site is favorable, the other sites
# unfavorable, scaled by the substrate's measured activity, so the model ranks sites within
# a molecule and the lowest-energy (fastest) site is the predicted point of fucosylation.

def build_energetic():
    """Training site examples for `esorex.energetic_specificity.EnergeticSpecificityModel`:
    returns (mols, cores, energies, labels), one entry per candidate C-OH site on the two
    natural substrates."""
    b_ops, op_map_nums = _get_b_ops()
    natural = _natural_substrates()
    best = max(info["activity"] for info in natural.values())
    mols, cores, energies, labels = [], [], [], []
    for name, info in natural.items():
        actual = actual_fucosylation_site(info["prod"])
        mol = Chem.AddHs(Chem.MolFromSmiles(info["sub"]))
        for o_idx, o_map, _ in _all_b_level_sites(mol, b_ops, op_map_nums):
            this_map = mol.GetAtomWithIdx(o_idx).GetAtomMapNum()
            is_pos = this_map == actual
            mols.append(mol)
            cores.append({o_idx})
            energies.append(-0.5 * (info["activity"] / best) if is_pos else 0.5)
            labels.append(f"{name}_{'pos' if is_pos else 'neg'}_map{this_map}")
    return mols, cores, energies, labels


def build_test_energetic():
    """Test panel for the energetic model: list of dicts with the molecule, its candidate
    sites as (o_idx, site_map), where site_map is the oxygen's substrate atom-map number (the
    identity comparable to `known_site`), and the known reactive site's map number (or None)."""
    from reports.fuctiii_compound_data import COMPOUNDS
    b_ops, op_map_nums = _get_b_ops()
    out = []
    for cn, c in COMPOUNDS.items():
        sub = c.get("substrate_smiles") or ""
        prod = c.get("product_smiles") or ""
        if not sub:
            continue
        mol = Chem.AddHs(Chem.MolFromSmiles(sub))
        if mol is None:
            continue
        sites = [(o_idx, mol.GetAtomWithIdx(o_idx).GetAtomMapNum())
                 for o_idx, _, _ in _all_b_level_sites(mol, b_ops, op_map_nums)]
        if not sites:
            continue
        out.append({
            "name": f"{cn} {c['name']}",
            "mol": mol,
            "sites": sites,
            "known_site": actual_fucosylation_site(prod) if prod else None,
            "measured_activity": c["kinetics"].get("normalized_score"),
        })
    return out
