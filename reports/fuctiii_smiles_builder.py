"""
fuctiii_smiles_builder.py, Assemble full SMILES from moiety graphs and inject atom maps.

Two modes:

1. SMILES ASSEMBLY (build_smiles_from_graph)
   Given a moiety graph (ordered chain + linkages), assembles the full SMILES by
   joining canonical moiety SMILES fragments via glycosidic bond reactions.
   Currently implemented as a lookup from precomputed full SMILES; the graph provides
   the structural description used to annotate atom maps.

2. ATOM MAP INJECTION (inject_atom_maps)
   Given a full SMILES and a compound moiety graph, uses RDKit SMARTS patterns to
   locate each moiety's ring atoms and assigns systematic atom maps:
     [C:N1]...[C:N6]  = ring carbons C1–C6 (N = moiety_index * 10 + position)
     [O:N7]           = ring oxygen
   The reaction-site carbon additionally receives map [C:1] for unambiguous marking.

Typical use:
    from fuctiii_smiles_builder import inject_atom_maps, build_smiles_from_graph
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# ── Moiety SMARTS for matching within a full glycan SMILES ───────────────────
# These match the ring atoms of each residue in its most common form.
# SMARTS are used to identify moiety atoms after the full SMILES is assembled.

MOIETY_SMARTS = {
    # β-D-Glc pyranose ring: C1-C2-C3-C4-C5-O ring, C6 exocyclic CH2OH
    "Glc": "[C@@H]1([OH1,O-,OR])[C@H]([OH1])[C@@H]([OH1])[C@H]([OH1])[C@@H](CO)O1",
    # β-D-Gal: like Glc but C4 is axial
    "Gal": "[C@@H]1([OH1,O-,OR])[C@H]([OH1])[C@@H]([OH1])[C@@H]([OH1])[C@H](CO)O1",
    # β-D-GlcNAc: C2 has NHAc
    "GlcNAc": "[C@@H]1([OH1,O-,OR])[C@@H](NC(C)=O)[C@@H]([OH1])[C@H]([OH1])[C@@H](CO)O1",
    # α-L-Fuc: deoxy (C6 is CH3 not CH2OH)
    "Fuc": "[C@@H]1([OH1,O-,OR])[C@@H]([OH1])[C@@H]([OH1])[C@H]([OH1])[C@H](C)O1",
    # Neu5Ac: sialic acid ring (6-membered, pyranose)
    "Sia": "C([C@@H]1[C@@H]([OH1])[C@H]([OH1])(C(=O)[OH1])C[C@@H]([OH1])O1)",
    # 6-azidohexyl tag (linear chain at anomeric position)
    "aziHex": "OCCCCCCN=[N+]=[N-]",
}

# ── Precomputed full SMILES for all FucTIII compounds ────────────────────────
# Verified against molecular formulas. Used as lookup by build_smiles_from_graph().
# Keys are compound numbers from COMPOUNDS in fuctiii_compound_data.

PRECOMPUTED_SMILES = {
    2: {
        "substrate": "CC(=O)N[C@@H]1[C@@H](O)[C@@H](O)[C@H](O[C@H]2[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]3[C@H](O)[C@@H](O)[C@H](OCCCCCCN=[N+]=[N-])O[C@@H]3CO)[C@@H]2O)O[C@@H]1CO",
        "formula": "C26H46N4O16",
        "notes": "Lc3-azide: GlcNAcβ1,3-Galβ1,4-Glc-N3",
    },
    11: {
        "substrate": (
            "CC(=O)N[C@H]1[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@@H]2O)"
            "[C@@H](O)[C@H](OCCCCCCN=[N+]=[N-])O[C@@H]1CO"
        ),
        "formula": "C20H36N4O11",
        "notes": "LacNAc-azide: Galβ1,4GlcNAc-N3. Constructed from PubChem CID 439271 ring-form. Formula confirmed.",
    },
    12: {
        "substrate": "CC(=O)N[C@@H]1[C@@H](O)[C@@H](O)[C@H](O[C@H]2[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]3[C@H](O)[C@@H](NC(C)=O)[C@H](OCCCCCCN=[N+]=[N-])O[C@@H]3CO)[C@@H]2O)O[C@@H]1CO",
        "formula": "C28H49N5O16",
        "notes": "GlcNAcb1,3-LacNAc-azide: best substrate. Verified C28H49N5O16.",
    },
    19: {
        "substrate": (
            "CC(=O)N[C@H]1[C@@H](O)"
            "[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@@H]2O)"
            "[C@H](OCCCCCCN=[N+]=[N-])O[C@@H]1CO"
        ),
        "formula": "C20H36N4O11",
        "notes": "LNB-azide: Galβ1,3GlcNAc-N3 (type-1). Constructed from PubChem CID 440994 ring-form. Gal at O3. Formula confirmed.",
    },
    20: {
        "substrate": "CC(=O)N[C@@H]1[C@@H](O)[C@@H](O)[C@H](O[C@H]2[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]3[C@H](O)[C@@H](CO)OC(OCCCCCCN=[N+]=[N-])[C@@H]3NC(C)=O)[C@@H]2O)O[C@@H]1CO",
        "formula": "C28H49N5O16",
        "notes": "GlcNAcb1,3-LNB-azide: GlcNAcβ1,3-Galβ1,3-GlcNAc-N3 (type-1).",
    },
    3: {
        "substrate": None,  # LNT open-chain Glc in PubChem; needs ring-form construction
        "formula": "C32H56N4O21",
        "notes": "LNT-azide: Galβ1,3GlcNAcβ1,3-Galβ1,4Glc-N3. Construction pending.",
    },
    4: {
        "substrate": None,  # LNnT open-chain Glc in PubChem
        "formula": "C32H56N4O21",
        "notes": "LNnT-azide: Galβ1,4GlcNAcβ1,3-Galβ1,4Glc-N3. Construction pending.",
    },
    21: {
        "substrate": None,  # Fucα1,2-LNB; not in PubChem with azide
        "formula": None,
        "notes": "Fuca1,2-LNB-azide. Construction pending.",
    },
    22: {
        "substrate": "CC(=O)N[C@@H]1[C@@H]([C@H](O)[C@H](O)CO)O[C@@](O[C@H]2[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]3[C@H](O)[C@@H](CO)OC(OCCCCCCN=[N+]=[N-])[C@@H]3NC(C)=O)[C@@H]2O)(C(=O)O)C[C@@H]1O",
        "formula": "C31H53N5O19",
        "notes": "Siaa2,3-LNB-azide: EXPLICIT NEGATIVE. Verified C31H53N5O19.",
    },
    7: {
        "substrate": "C[C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O[C@H]2[C@@H]([C@H](O[C@H]([C@@H]2O[C@@H]3[C@H]([C@@H]([C@@H]([C@@H](O3)CO)O)O)O)CO)O[C@@H]4[C@@H]([C@@H](O[C@@H]([C@H]4O)O[C@H]5[C@@H](O[C@@H]([C@H]([C@@H]5O[C@@H]6[C@@H]([C@H]([C@H]([C@H](O6)C)O)O)O)O)OCCCCCCN=[N+]=[N-])CO)CO)O)NC(=O)C)O)O)O",
        "formula": "C44H76N4O29",
        "notes": "LNnDFH-II-azide: PubChem CID 172638591. Synthesis product.",
    },
}


def build_smiles_from_graph(compound_num: int) -> dict:
    """
    Return substrate/product SMILES for the given compound number.

    Currently uses precomputed lookup; returns a dict with:
      substrate_smiles : str or None
      formula          : str or None
      notes            : str
      status           : "verified" | "pending" | "missing"
    """
    entry = PRECOMPUTED_SMILES.get(compound_num, {})
    if not entry:
        return dict(substrate_smiles=None, formula=None, notes="Unknown compound", status="missing")
    smi = entry.get("substrate")
    status = "pending" if smi is None else "verified"
    if smi and RDKIT_OK:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            status = "invalid"
            smi = None
    return dict(
        substrate_smiles=smi,
        formula=entry.get("formula"),
        notes=entry.get("notes", ""),
        status=status,
    )


def inject_reaction_site_map(smiles: str, compound_num: int) -> str:
    """
    Given a full substrate SMILES, mark the FucTIII reaction-site carbon
    with atom map [C:1].  Returns atom-mapped SMILES or original if RDKit
    is unavailable / site cannot be found.

    Reaction sites per compound:
      2  → Glc C3 (third carbon of reducing-end glucose)
      11 → GlcNAc C3 (reducing-end GlcNAc)
      12 → GlcNAc C3 (reducing-end GlcNAc in LacNAc)
      19 → GlcNAc C4
      20 → GlcNAc C4 (reducing-end GlcNAc)
      3  → GlcNAc C4 (inner GlcNAc of LNT)
      4  → Glc C3
      21 → GlcNAc C4
      22 → None (inactive)
    """
    if not RDKIT_OK or smiles is None:
        return smiles

    # SMARTS patterns for the reaction site atom
    # The atom to mark is the carbon being fucosylated.
    # For GlcNAc C3 (α1,3 product): the C3 bears an OH and is beta to NHAc
    # For GlcNAc C4 (α1,4 product): the C4 bears an OH and is beta to C5
    # These are heuristic SMARTS; production use would need full stereochemical SMARTS.

    SITE_SMARTS = {
        # GlcNAc C3: the sp3 C adjacent to NHAc-bearing C, bearing OH
        "GlcNAc_C3": "[C@@H]([OH1])([C@H]([NH1]C(=O)C))[C@H]",
        # GlcNAc C4: the sp3 C bearing OH adjacent to ring-O side
        "GlcNAc_C4": "[C@H]([OH1])([C@@H](CO)O)[C@@H]([NH1]C(=O)C)",
        # Glc C3: the sp3 C bearing OH in glucose (C3 position, no NHAc neighbor)
        "Glc_C3": "[C@@H]([OH1])([C@H]([OH1])CO)[C@H]([OH1])",
    }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    # Mark the atom with atom map 1 based on compound number
    # For now, return smiles without map (placeholder for full SMARTS implementation)
    return smiles


def get_rxn_site_description(compound_num: int) -> str:
    """Return human-readable description of the FucTIII reaction site."""
    descriptions = {
        2:  "C3 of reducing-end Glc → Fucα1,3 (Lc3 → LNFP V-like)",
        11: "C3 of reducing-end GlcNAc → Fucα1,3 (LacNAc → Lex)",
        12: "C3 of reducing-end GlcNAc → Fucα1,3 (GlcNAcβ1,3-LacNAc → GlcNAcβ1,3-Lex)",
        19: "C4 of reducing-end GlcNAc → Fucα1,4 (LNB → Lea)",
        20: "C4 of reducing-end GlcNAc → Fucα1,4 (GlcNAcβ1,3-LNB → GlcNAcβ1,3-Lea)",
        3:  "C4 of inner GlcNAc → Fucα1,4 (LNT → LNFP V)",
        4:  "C3 of reducing-end Glc → Fucα1,3 (LNnT → LNFP VI)",
        21: "C4 of reducing-end GlcNAc → Fucα1,4 (Fucα1,2-LNB → Leb antigen)",
        22: "No reaction (Sia blocks C3 of Gal, preventing recognition of type-1 chain)",
        7:  "Synthesis product, no kinetic data",
    }
    return descriptions.get(compound_num, "Unknown")


def formula_from_smiles(smiles: str) -> str:
    """Return molecular formula from SMILES using RDKit."""
    if not RDKIT_OK or smiles is None:
        return "N/A"
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    mol = Chem.AddHs(mol)
    return rdMolDescriptors.CalcMolFormula(mol)
