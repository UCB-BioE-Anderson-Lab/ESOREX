"""
Preprocess FucTIII substrate data from Tsai et al. ACS Catal. 2019, 9, 10712-10720.

Outputs data/fucosyltransferases/FucTIII_reactions.csv with one row per substrate:
  - All kinetic and synthesis data from the paper
  - Ring-form azide-tagged substrate SMILES (constructed or PubChem-sourced)
  - Product SMILES (fucosylated substrate, where constructable)
  - Normalized energy score (0-1) from free_energy.py for kinetic data;
    1.0/0.0 boolean for synthesis-only substrates
  - Atom-map index of the fucosylation site on the substrate (when SMILES available)

SMILES status codes:
  azide-tagged-pubchem   PubChem already has the full azide-tagged compound
  rdkit-constructed      Built from known sugar building blocks; RDKit-validated
  pubchem-ring-azide     PubChem free sugar (ring form) + RDKit azide addition
  needs-construction     Open-chain reducing end in PubChem; requires manual build
"""

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from esorex.free_energy import rate_to_energy, compute_scaling_factor

try:
    from rdkit import Chem
    from rdkit.Chem import RWMol, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available; SMILES construction skipped", file=sys.stderr)

OUT = Path(__file__).parent / "FucTIII_reactions.csv"

# ── Normalization constants ──────────────────────────────────────────────────
# All kcat/KM values in mM-1 s-1.  x0=1.0 is the standard reference rate.
# xmax = 1.40 (compound 12, the best substrate).
X0   = 1.0
XMAX = 1.40
SCALING_FACTOR = compute_scaling_factor(XMAX, X0)


def kinetic_energy(kcat_km: float) -> float:
    """Normalized energy score from kcat/KM in mM-1 s-1."""
    return rate_to_energy(kcat_km, X0, SCALING_FACTOR)


# ── SMILES building blocks ────────────────────────────────────────────────────

# Previously constructed and RDKit-validated in reports/fuctiii_smiles_review.py
SMILES_VERIFIED = {
    # GlcNAcβ1,3-Galβ1,4GlcNAc-6azidohexyl (compound 12, training positive)
    12: "CC(=O)N[C@@H]1[C@@H](O)[C@@H](O)[C@H](O[C@H]2[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]3[C@H](O)[C@@H](NC(C)=O)[C@H](OCCCCCCN=[N+]=[N-])O[C@@H]3CO)[C@@H]2O)O[C@@H]1CO",
    # GlcNAcβ1,3-Galβ1,3GlcNAc-6azidohexyl (compound 20)
    20: "CC(=O)N[C@@H]1[C@@H](O)[C@@H](O)[C@H](O[C@H]2[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]3[C@H](O)[C@@H](CO)OC(OCCCCCCN=[N+]=[N-])[C@@H]3NC(C)=O)[C@@H]2O)O[C@@H]1CO",
    # Siaα2,3-Galβ1,3GlcNAc-6azidohexyl (compound 22, explicit negative)
    22: "CC(=O)N[C@@H]1[C@@H]([C@H](O)[C@H](O)CO)O[C@@](O[C@H]2[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]3[C@H](O)[C@@H](CO)OC(OCCCCCCN=[N+]=[N-])[C@@H]3NC(C)=O)[C@@H]2O)(C(=O)O)C[C@@H]1O",
    # LNnDFH II (compound 7): PubChem CID 172638591 already has azide tag
    7:  "C[C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O[C@H]2[C@@H]([C@H](O[C@H]([C@@H]2O[C@@H]3[C@H]([C@@H]([C@@H]([C@@H](O3)CO)O)O)O)CO)O[C@@H]4[C@@H]([C@@H](O[C@@H]([C@H]4O)O[C@H]5[C@@H](O[C@@H]([C@H]([C@@H]5O[C@@H]6[C@@H]([C@H]([C@H]([C@H](O6)C)O)O)O)O)OCCCCCCN=[N+]=[N-])CO)CO)O)NC(=O)C)O)O)O",
}

# PubChem free sugars (ring form) where azide addition via RDKit works
PUBCHEM_FREE_SUGARS = {
    11: ("439271",  "CC(=O)N[C@@H]1[C@H]([C@@H]([C@H](O[C@H]1O)CO)O[C@H]2[C@@H]([C@H]([C@H]([C@H](O2)CO)O)O)O)O"),   # LacNAc
    19: ("440994",  "CC(=O)N[C@@H]1[C@H]([C@@H]([C@H](OC1O)CO)O)O[C@H]2[C@@H]([C@H]([C@H]([C@H](O2)CO)O)O)O"),          # LNB
     3: ("440993",  "CC(=O)N[C@@H]1[C@H]([C@@H]([C@H](O[C@H]1O[C@H]2[C@H]([C@H](O[C@H]([C@@H]2O)O[C@@H]3[C@H](OC([C@@H]([C@H]3O)O)O)CO)CO)O)CO)O)O[C@H]4[C@@H]([C@H]([C@H]([C@H](O4)CO)O)O)O"),  # LNT
}

# Ring-form SMILES constructed manually (reducing-end Glc/GlcNAc replaced or extended)
MANUALLY_CONSTRUCTED = {
    # Lc3 = GlcNAcβ1,3-Galβ1,4Glc-6azidohexyl
    # Derived from compound 12 by replacing reducing-end GlcNAc C2-NHAc with C2-OH
    # Expected formula: C26H46N4O16
    2:  "CC(=O)N[C@@H]1[C@@H](O)[C@@H](O)[C@H](O[C@H]2[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]3[C@H](O)[C@@H](O)[C@H](OCCCCCCN=[N+]=[N-])O[C@@H]3CO)[C@@H]2O)O[C@@H]1CO",
}

# Compounds needing manual ring-form SMILES construction (open-chain PubChem SMILES)
NEEDS_CONSTRUCTION = {
    4:  ("121853",  "LNnT = Galβ1,4GlcNAcβ1,3Galβ1,4Glc; open-chain Glc in PubChem; expected C32H56N4O21"),
    21: (None,      "Fucα1,2-Galβ1,3GlcNAc (H-1 antigen); no suitable PubChem entry with azide"),
}


def add_azidohexyl_ring_anomer(free_smiles: str):
    """Add -(CH2)6-N=[N+]=[N-] to the free anomeric OH of a ring-form sugar."""
    if not HAS_RDKIT or not free_smiles:
        return None
    mol = Chem.MolFromSmiles(free_smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    anomeric_o = None
    for atom in mol.GetAtoms():
        if not atom.IsInRing() or atom.GetAtomicNum() != 6:
            continue
        ring_os = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and n.IsInRing()]
        exo_os  = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and not n.IsInRing()]
        if len(ring_os) == 1 and len(exo_os) == 1:
            exo_o = exo_os[0]
            exo_c = [n for n in exo_o.GetNeighbors() if n.GetAtomicNum() != 1]
            if len(exo_c) == 1:
                anomeric_o = exo_o
                break

    if anomeric_o is None:
        return None

    h_to_remove = next((n for n in anomeric_o.GetNeighbors() if n.GetAtomicNum() == 1), None)
    if h_to_remove is None:
        return None

    rwmol = RWMol(mol)
    rwmol.RemoveAtom(h_to_remove.GetIdx())

    mol2 = Chem.AddHs(rwmol.GetMol())
    rwmol2 = RWMol(mol2)

    anomer_o_idx = None
    for atom in rwmol2.GetAtoms():
        if not atom.IsInRing() or atom.GetAtomicNum() != 6:
            continue
        ring_os = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and n.IsInRing()]
        exo_os  = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and not n.IsInRing()]
        if len(ring_os) == 1 and len(exo_os) == 1:
            exo_o = exo_os[0]
            exo_c = [n for n in exo_o.GetNeighbors() if n.GetAtomicNum() != 1]
            exo_h = [n for n in exo_o.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(exo_c) == 1 and len(exo_h) == 0:
                anomer_o_idx = exo_o.GetIdx()
                break

    if anomer_o_idx is None:
        return None

    chain_start = rwmol2.AddAtom(Chem.Atom(6))
    for _ in range(5):
        nc = rwmol2.AddAtom(Chem.Atom(6))
        rwmol2.AddBond(nc - 1, nc, Chem.BondType.SINGLE)
    n3_idx = rwmol2.AddAtom(Chem.Atom(7))
    n2_idx = rwmol2.AddAtom(Chem.Atom(7))
    n1_idx = rwmol2.AddAtom(Chem.Atom(7))
    rwmol2.GetAtomWithIdx(n2_idx).SetFormalCharge(1)
    rwmol2.GetAtomWithIdx(n1_idx).SetFormalCharge(-1)
    rwmol2.AddBond(chain_start + 5, n3_idx, Chem.BondType.SINGLE)
    rwmol2.AddBond(n3_idx, n2_idx, Chem.BondType.DOUBLE)
    rwmol2.AddBond(n2_idx, n1_idx, Chem.BondType.DOUBLE)
    rwmol2.AddBond(anomer_o_idx, chain_start, Chem.BondType.SINGLE)

    try:
        mol_final = rwmol2.GetMol()
        Chem.SanitizeMol(mol_final)
        mol_final = Chem.RemoveHs(mol_final)
        return Chem.MolToSmiles(mol_final)
    except Exception:
        return None


def mol_formula(smiles: str) -> str:
    if not HAS_RDKIT or not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    return rdMolDescriptors.CalcMolFormula(mol)


def find_reaction_site_atom(substrate_smiles: str, site_description: str):
    """
    Return the atom index of the fucosylation site (C3 or C4 of GlcNAc, or C3 of Glc).
    Returns (atom_idx, note) or (None, reason).

    Strategy: find the relevant ring (reducing-end GlcNAc or Glc), then identify
    C3 or C4 based on neighbor patterns.
    TODO: implement full atom-map resolution per site_description.
    """
    return None, "atom-map not yet implemented"


# ── Paper data ────────────────────────────────────────────────────────────────
# All substrates tested with FucTIII as acceptor.
# Compound numbers are the paper's compound numbers (Tsai et al. ACS Catal. 2019).

PAPER_DATA = [
    # ── KINETIC SUBSTRATES (Table 1) ──────────────────────────────────────────
    # compound 2: Lc3 = GlcNAcβ1,3-Galβ1,4Glc (type-2 chain, C3 of Glc)
    # FucTIII α1,3-fucosylates the C3 of the Glc reducing-end unit.
    dict(
        compound_number=2,
        name="Lc3-6azidohexyl",
        sugar_structure="GlcNAcβ1,3-Galβ1,4Glc",
        chain_type="type-2",
        fucosylation_site="C3_Glc",
        fucose_linkage="alpha1_3",
        role="kinetic_substrate",
        kcat_s=0.29, kcat_err_s=0.02,
        KM_mM=4.79,  KM_err_mM=0.94,
        kcat_KM_mM_s=0.06, kcat_KM_err_mM_s=0.02,
        synthesis_yield_pct=None,
        paper_activity=0.06, paper_activity_type="kcat_KM_mM_s",
        normalized_energy=kinetic_energy(0.06),
        pubchem_cid="53477860",
        smiles_key=("manually_constructed", 2),
        notes="Type-2 chain (LacNAc→Glc). FucTIII α1,3-fucosylates C3 of Glc (reducing end). "
              "kcat/KM 39× lower than LacNAc disaccharide alone; medium affinity.",
    ),
    # compound 11: LacNAc = Galβ1,4GlcNAc (type-2, C3 of GlcNAc)
    dict(
        compound_number=11,
        name="LacNAc-6azidohexyl",
        sugar_structure="Galβ1,4GlcNAc",
        chain_type="type-2",
        fucosylation_site="C3_GlcNAc",
        fucose_linkage="alpha1_3",
        role="kinetic_substrate",
        kcat_s=0.35, kcat_err_s=0.03,
        KM_mM=9.61,  KM_err_mM=1.58,
        kcat_KM_mM_s=0.04, kcat_KM_err_mM_s=0.02,
        synthesis_yield_pct=71,
        paper_activity=0.04, paper_activity_type="kcat_KM_mM_s",
        normalized_energy=kinetic_energy(0.04),
        pubchem_cid="439271",
        smiles_key=("pubchem_ring_azide", 11),
        notes="Type-2 disaccharide (LacNAc). Synthesis to Lex in 71% yield (23h). "
              "39× worse kcat/KM than GlcNAcβ1,3-LacNAc (cmpd 12).",
    ),
    # compound 12: GlcNAcβ1,3-LacNAc = GlcNAcβ1,3-Galβ1,4GlcNAc (type-2, BEST)
    dict(
        compound_number=12,
        name="GlcNAcb1-3-LacNAc-6azidohexyl",
        sugar_structure="GlcNAcβ1,3-Galβ1,4GlcNAc",
        chain_type="type-2",
        fucosylation_site="C3_GlcNAc",
        fucose_linkage="alpha1_3",
        role="kinetic_substrate",
        kcat_s=2.40, kcat_err_s=0.08,
        KM_mM=1.72,  KM_err_mM=0.24,
        kcat_KM_mM_s=1.40, kcat_KM_err_mM_s=0.32,
        synthesis_yield_pct=71,
        paper_activity=1.40, paper_activity_type="kcat_KM_mM_s",
        normalized_energy=kinetic_energy(1.40),
        pubchem_cid=None,
        smiles_key=("rdkit_constructed", 12),
        notes="BEST substrate. Additional GlcNAc at non-reducing end boosts kcat 7× and lowers KM 6×. "
              "Synthesis to GlcNAcβ1,3-Lex in 71% yield (6h, faster than compound 11). C28H49N5O16.",
    ),
    # compound 19: LNB = Galβ1,3GlcNAc (type-1, WORST)
    # kcat/KM footnote: reported as 0.93 × 10⁻³ mM⁻¹s⁻¹ = 0.00093 mM⁻¹s⁻¹
    dict(
        compound_number=19,
        name="LNB-6azidohexyl",
        sugar_structure="Galβ1,3GlcNAc",
        chain_type="type-1",
        fucosylation_site="C4_GlcNAc",
        fucose_linkage="alpha1_4",
        role="kinetic_substrate",
        kcat_s=0.02, kcat_err_s=0.00,
        KM_mM=18.14, KM_err_mM=2.00,
        kcat_KM_mM_s=0.00093, kcat_KM_err_mM_s=0.00048,
        synthesis_yield_pct=47,
        paper_activity=0.00093, paper_activity_type="kcat_KM_mM_s",
        normalized_energy=kinetic_energy(0.00093),
        pubchem_cid="440994",
        smiles_key=("pubchem_ring_azide", 19),
        notes="WORST substrate. Type-1 disaccharide (LNB). Paper footnote: value is ×10⁻³ mM⁻¹s⁻¹ "
              "(= 0.93 μM⁻¹s⁻¹). Synthesis to Lea in 47% yield (77% brsm, 48h). "
              "39× worse than LacNAc (type-2 counterpart).",
    ),
    # compound 20: GlcNAcβ1,3-LNB (type-1 extended)
    dict(
        compound_number=20,
        name="GlcNAcb1-3-LNB-6azidohexyl",
        sugar_structure="GlcNAcβ1,3-Galβ1,3GlcNAc",
        chain_type="type-1",
        fucosylation_site="C4_GlcNAc",
        fucose_linkage="alpha1_4",
        role="kinetic_substrate",
        kcat_s=0.24, kcat_err_s=0.01,
        KM_mM=5.65,  KM_err_mM=0.68,
        kcat_KM_mM_s=0.04, kcat_KM_err_mM_s=0.02,
        synthesis_yield_pct=55,
        paper_activity=0.04, paper_activity_type="kcat_KM_mM_s",
        normalized_energy=kinetic_energy(0.04),
        pubchem_cid=None,
        smiles_key=("rdkit_constructed", 20),
        notes="65× better kcat/KM than LNB (cmpd 19). Additional GlcNAc at non-reducing end "
              "lowers KM 3× and raises kcat 12×. Synthesis to GlcNAcβ1,3-Lea in 55% (89% brsm, 42h). "
              "C28H49N5O16.",
    ),
    # ── SYNTHESIS-ONLY SUBSTRATES ─────────────────────────────────────────────
    # compound 3: LNT = Galβ1,3GlcNAcβ1,3Galβ1,4Glc (type-1, 70% yield → LNFP V)
    # FucTIII α1,4-fucosylates C4 of GlcNAc in the type-1 chain.
    dict(
        compound_number=3,
        name="LNT-6azidohexyl",
        sugar_structure="Galβ1,3GlcNAcβ1,3Galβ1,4Glc",
        chain_type="type-1",
        fucosylation_site="C4_GlcNAc",
        fucose_linkage="alpha1_4",
        role="synthesis_substrate",
        kcat_s=None, kcat_err_s=None,
        KM_mM=None,  KM_err_mM=None,
        kcat_KM_mM_s=None, kcat_KM_err_mM_s=None,
        synthesis_yield_pct=70,
        paper_activity=70, paper_activity_type="synthesis_yield_pct",
        normalized_energy=1.0,
        pubchem_cid="440993",
        smiles_key=("pubchem_ring_azide", 3),
        notes="LNT (lacto-N-tetraose). FucTIII gives LNFP V at 70% yield; "
              "double fucosylation with 2.5 equiv GDP-Fuc gives LNDFH II at 53% (86% brsm). "
              "Type-1 chain; fucosylation at GlcNAc C4.",
    ),
    # compound 4: LNnT = Galβ1,4GlcNAcβ1,3Galβ1,4Glc (type-2, 70% → LNFP VI)
    # FucTIII α1,3-fucosylates C3 of the reducing-end Glc (not the LacNAc GlcNAc).
    dict(
        compound_number=4,
        name="LNnT-6azidohexyl",
        sugar_structure="Galβ1,4GlcNAcβ1,3Galβ1,4Glc",
        chain_type="type-2",
        fucosylation_site="C3_Glc",
        fucose_linkage="alpha1_3",
        role="synthesis_substrate",
        kcat_s=None, kcat_err_s=None,
        KM_mM=None,  KM_err_mM=None,
        kcat_KM_mM_s=None, kcat_KM_err_mM_s=None,
        synthesis_yield_pct=70,
        paper_activity=70, paper_activity_type="synthesis_yield_pct",
        normalized_energy=1.0,
        pubchem_cid="121853",
        smiles_key=("needs_construction", 4),
        notes="LNnT (lacto-N-neotetraose). FucTIII α1,3-fucosylates Glc C3 (reducing end, not LacNAc) "
              "giving LNFP VI (70%) and LNnDFH II (10%) simultaneously. "
              "Refucosylating LNFP VI gives LNnDFH II at 65% (88% brsm). "
              "Open-chain Glc in PubChem SMILES; ring-form construction needed. Expected C32H56N4O21.",
    ),
    # compound 21: Fucα1,2-LNB = Fucα1,2-Galβ1,3GlcNAc (H-1 antigen, type-1)
    # FucTIII α1,4-fucosylates C4 of GlcNAc despite α1,2-Fuc at Gal.
    dict(
        compound_number=21,
        name="Fuca1-2-LNB-6azidohexyl",
        sugar_structure="Fucα1,2-Galβ1,3GlcNAc",
        chain_type="type-1",
        fucosylation_site="C4_GlcNAc",
        fucose_linkage="alpha1_4",
        role="synthesis_substrate",
        kcat_s=None, kcat_err_s=None,
        KM_mM=None,  KM_err_mM=None,
        kcat_KM_mM_s=None, kcat_KM_err_mM_s=None,
        synthesis_yield_pct=51,
        paper_activity=51, paper_activity_type="synthesis_yield_pct",
        normalized_energy=1.0,
        pubchem_cid=None,
        smiles_key=("needs_construction", 21),
        notes="H-1 antigen (Fucα1,2-LNB). Synthesis → Leb (both α1,2 and α1,4 Fuc) at 51% (91% brsm, 46h). "
              "α1,2-Fuc at Gal does NOT block FucTIII from adding α1,4-Fuc at GlcNAc C4. "
              "SMILES not yet constructed.",
    ),
    # ── EXPLICIT NEGATIVE ────────────────────────────────────────────────────
    # compound 22: Siaα2,3-LNB = Siaα2,3-Galβ1,3GlcNAc
    # Confirmed: NO product detected. Sialylation at Gal C3 blocks α1,4-fucosylation.
    dict(
        compound_number=22,
        name="Siaa2-3-LNB-6azidohexyl",
        sugar_structure="Siaα2,3-Galβ1,3GlcNAc",
        chain_type="type-1",
        fucosylation_site="inactive",
        fucose_linkage="none",
        role="inactive",
        kcat_s=None, kcat_err_s=None,
        KM_mM=None,  KM_err_mM=None,
        kcat_KM_mM_s=None, kcat_KM_err_mM_s=None,
        synthesis_yield_pct=0,
        paper_activity=0, paper_activity_type="boolean",
        normalized_energy=0.0,
        pubchem_cid=None,
        smiles_key=("rdkit_constructed", 22),
        notes="EXPLICIT NEGATIVE. Siaα2,3-Galβ1,3GlcNAc: Sia at Gal C3 (where GlcNAc is normally at C3 "
              "in LNB-type substrates) completely abolishes α1,4-fucosylation. "
              "Confirmed no detectable product in synthesis attempt. C31H53N5O19.",
    ),
    # ── SYNTHESIS REFERENCE (PubChem azide-tagged product) ───────────────────
    # compound 7: LNnDFH II (di-fucosylated product from LNnT)
    # This is a product, not a substrate. Included for completeness and future model use.
    dict(
        compound_number=7,
        name="LNnDFH-II-6azidohexyl",
        sugar_structure="Fucα1,3-Galβ1,4GlcNAcβ1,3Galβ1,4[Fucα1,3]Glc",
        chain_type="type-2",
        fucosylation_site="product",
        fucose_linkage="alpha1_3_double",
        role="synthesis_product",
        kcat_s=None, kcat_err_s=None,
        KM_mM=None,  KM_err_mM=None,
        kcat_KM_mM_s=None, kcat_KM_err_mM_s=None,
        synthesis_yield_pct=65,
        paper_activity=65, paper_activity_type="synthesis_yield_pct",
        normalized_energy=None,
        pubchem_cid="172638591",
        smiles_key=("azide_tagged_pubchem", 7),
        notes="Lacto-N-neodifucohexaose II. Di-fucosylated product from LNnT (cmpd 4). "
              "65% yield (88% brsm) by refucosylating LNFP VI; only 10% in single step from LNnT. "
              "PubChem has azide-tagged SMILES.",
    ),
]

# ── SMILES resolution ─────────────────────────────────────────────────────────

def resolve_smiles(entry: dict) -> tuple:
    """Returns (substrate_smiles, smiles_status, pubchem_cid_used)."""
    key_type, key_id = entry["smiles_key"]

    if key_type == "rdkit_constructed":
        smi = SMILES_VERIFIED.get(key_id)
        return smi, "rdkit-constructed", None

    if key_type == "azide_tagged_pubchem":
        smi = SMILES_VERIFIED.get(key_id)
        return smi, "azide-tagged-pubchem", entry.get("pubchem_cid")

    if key_type == "manually_constructed":
        smi = MANUALLY_CONSTRUCTED.get(key_id)
        return smi, "rdkit-constructed", None

    if key_type == "pubchem_ring_azide":
        cid, free_smiles = PUBCHEM_FREE_SUGARS[key_id]
        tagged = add_azidohexyl_ring_anomer(free_smiles)
        if tagged:
            return tagged, "pubchem-ring-azide", cid
        return free_smiles, "free-sugar-only", cid

    if key_type == "needs_construction":
        return None, "needs-construction", entry.get("pubchem_cid")

    return None, "unknown", None


# ── CSV columns ────────────────────────────────────────────────────────────────
COLUMNS = [
    "compound_number", "name", "sugar_structure", "chain_type",
    "fucosylation_site", "fucose_linkage", "role",
    "kcat_s", "kcat_err_s", "KM_mM", "KM_err_mM",
    "kcat_KM_mM_s", "kcat_KM_err_mM_s",
    "synthesis_yield_pct",
    "paper_activity", "paper_activity_type",
    "normalized_energy",
    "energy_x0", "energy_xmax", "energy_scaling_factor",
    "pubchem_cid", "smiles_status", "substrate_smiles", "substrate_formula",
    "product_smiles",
    "notes",
]


def main():
    rows = []
    for entry in PAPER_DATA:
        substrate_smiles, smiles_status, pubchem_cid_used = resolve_smiles(entry)
        formula = mol_formula(substrate_smiles) if substrate_smiles else ""

        row = {k: entry.get(k, "") for k in COLUMNS}
        row["pubchem_cid"]       = pubchem_cid_used or entry.get("pubchem_cid") or ""
        row["smiles_status"]     = smiles_status
        row["substrate_smiles"]  = substrate_smiles or ""
        row["substrate_formula"] = formula
        row["product_smiles"]    = ""   # populated in next step (add_fuc_to_substrate.py)
        row["energy_x0"]         = X0
        row["energy_xmax"]       = XMAX
        row["energy_scaling_factor"] = round(SCALING_FACTOR, 6)
        # format None as empty string
        for k in row:
            if row[k] is None:
                row[k] = ""
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT}")
    print()

    # Summary
    for row in rows:
        smi = row["substrate_smiles"]
        e   = row["normalized_energy"]
        e_str = f"{float(e):.4f}" if e != "" else "N/A (product)"
        print(f"  Cmpd {str(row['compound_number']):2s}: {row['name'][:40]:40s} "
              f"status={row['smiles_status']:25s} formula={row['substrate_formula']:18s} energy={e_str}")


if __name__ == "__main__":
    main()
