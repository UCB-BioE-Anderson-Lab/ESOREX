"""
Moiety library and compound moiety-graph definitions for FucTIII substrate data.

Reference: Tsai et al. ACS Catalysis 2019, 9, 10712–10720.
           DOI: 10.1021/acscatal.9b03752

Compound representation
-----------------------
Each compound has:
  chain    : ordered list of moiety IDs forming the PRIMARY BACKBONE,
             from the non-reducing (NR) end to the reducing end, then the tag.
             Branch substituents are NOT included in chain.
  linkages : list of (from_idx, to_idx, linkage_str, donor_pos, acceptor_pos)
             describing backbone edges only.
  branches : dict { backbone_node_index → list of branch-dicts }
             Each branch-dict: {"moiety": id, "linkage": str,
                                "donor_pos": int, "acceptor_pos": int,
                                "side": "above" | "below"}
             e.g. compound 21: {0: [{"moiety": "Fuc", "linkage": "α1,2",
                                      "donor_pos": 1, "acceptor_pos": 2,
                                      "side": "above"}]}

Scoring
-------
Compounds with measured kcat/KM receive a log-scaled catalytic-efficiency score:

  E(x) = ln(1 + x/x0) / ln(1 + xmax/x0)    (softplus normalization)

Synthesis-yield compounds (preparative yield, no kinetics) use a calibrated
pseudo-kcat/KM derived from a single-concentration kinetic model:

  yield = 1 - exp(-kcat/KM × [E] × t)   →   kcat/KM ∝ -ln(1-yield) / t

Calibration point: compound 11 (kcat/KM = 0.04 mM⁻¹s⁻¹, yield = 71%, t = 23 h)
gives calibration constant C = 0.04 / (-ln(0.29) / (23 × 3600)) = 2675.6 s·mM.

  est_kcat_KM = -ln(1 - yield) / (t_h × 3600) × C

The resulting est_kcat_KM feeds the same ke() normalization as kinetic data.
Confidence is recorded as "preparative yield, calibrated pseudo-kcat/KM".

Explicit negatives receive normalized_score = 0.0 by convention.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from esorex.free_energy import rate_to_energy, compute_scaling_factor

# ── Normalization (Table 1, Tsai et al. 2019) ────────────────────────────────
X0   = 1.0    # reference rate, mM⁻¹s⁻¹
XMAX = 1.40   # best substrate (compound 12)
SF   = compute_scaling_factor(XMAX, X0)

def ke(r):
    """Return log-scaled catalytic-efficiency score from kcat/KM (mM⁻¹s⁻¹)."""
    if r is None or r <= 0:
        return None
    return rate_to_energy(r, X0, SF)

# Calibration for synthesis-yield compounds (see module docstring).
# Anchor: compound 11, kcat/KM=0.04, yield=71%, t=23h → C = 2675.6 s·mM
import math as _math
_CALIB_C = 2675.6   # s·mM

def _yield_to_score(yield_pct: float, t_h: float) -> float:
    """Pseudo-kcat/KM from preparative yield and reaction time, then normalized."""
    est = -_math.log(1.0 - yield_pct / 100.0) / (t_h * 3600.0) * _CALIB_C
    return ke(est)


# ── Moiety library ────────────────────────────────────────────────────────────
MOIETIES = {
    "Glc": {
        "full_name": "β-D-Glucopyranose",
        "abbreviation": "Glc",
        "pubchem_cid": "5793",
        "smiles": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        "color": "#2166ac",
        "text_color": "#ffffff",
        "shape": "hexagon",
        "description": "Reducing-end glucose residue. In Lc3 (cmpd 2) and LNnT (cmpd 4), "
                        "FucTIII fucosylates O3 of this residue (Fucα1,3).",
    },
    "Gal": {
        "full_name": "β-D-Galactopyranose",
        "abbreviation": "Gal",
        "pubchem_cid": "6036",
        "smiles": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O",
        "color": "#f4a100",
        "text_color": "#000000",
        "shape": "hexagon",
        "description": "Galactose. C4 epimer of Glc. Forms the core of LacNAc "
                        "(type-2: Galβ1,4GlcNAc) and LNB (type-1: Galβ1,3GlcNAc) chains.",
    },
    "GlcNAc": {
        "full_name": "N-acetyl-β-D-glucosamine",
        "abbreviation": "GlcNAc",
        "pubchem_cid": "439174",
        "smiles": "CC(=O)N[C@@H]1[C@H](O)[C@@H](O)[C@H](O[C@@H]1CO)O",
        "color": "#1a6e2a",
        "text_color": "#ffffff",
        "shape": "hexagon",
        "description": "N-acetylglucosamine. Primary FucTIII acceptor residue. "
                        "FucTIII fucosylates O3 (type-2 chains, Fucα1,3) or "
                        "O4 (type-1 chains, Fucα1,4).",
    },
    "Fuc": {
        "full_name": "α-L-Fucopyranose",
        "abbreviation": "Fuc",
        "pubchem_cid": "442519",
        "smiles": "C[C@@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        "color": "#c0392b",
        "text_color": "#ffffff",
        "shape": "triangle",
        "description": "L-Fucose (6-deoxy-L-galactose). Transferred from GDP-β-L-fucose. "
                        "C6 is a methyl group (no hydroxymethyl).",
    },
    "Sia": {
        "full_name": "N-acetylneuraminic acid (Neu5Ac)",
        "abbreviation": "Neu5Ac",
        "pubchem_cid": "65309",
        "smiles": "CC(=O)N[C@@H]1[C@@H](O)C[C@@](O)(C[C@H]1O)C(=O)O",
        "color": "#7b2d8b",
        "text_color": "#ffffff",
        "shape": "diamond",
        "description": "Sialic acid (Neu5Ac). α2,3-linked to Gal C3 in compound 22. "
                        "Occupying Gal C3 blocks FucTIII recognition of the type-1 chain, "
                        "making GlcNAc O4 inaccessible (explicit negative).",
    },
    "aziHex": {
        "full_name": "6-azidohexyl reducing-end tag",
        "abbreviation": "N₃",
        "pubchem_cid": None,
        "smiles": "OCCCCCCN=[N+]=[N-]",
        "color": "#888888",
        "text_color": "#ffffff",
        "shape": "tag",
        "description": "6-Azidohexyl group at the reducing-end anomeric carbon. "
                        "Used for C18 reversed-phase SPE purification and UV detection at 250 nm. "
                        "Distal to the FucTIII reaction site.",
    },
}


# ── SMILES status vocabulary ─────────────────────────────────────────────────
# verified, SMILES from a confirmed source; molecular formula checked
# constructed, built manually; formula checked
# source_free, only the free (non-azide-tagged) sugar from PubChem is available
# pending, SMILES not yet available; see smiles_pending_reason
# failed, attempted construction failed; reason given

SMILES_STATUS_LABELS = {
    "verified":    "✓ SMILES verified",
    "constructed": "⚒ Constructed, formula checked",
    "source_free": "◑ Free-sugar source only",
    "pending":     "⏳ Pending construction",
    "failed":      "✗ Construction failed",
}


# ── Compound moiety graphs ────────────────────────────────────────────────────
COMPOUNDS = {
    # ── KINETIC SUBSTRATES (Table 1) ─────────────────────────────────────────
    2: dict(
        name="Lc3-6azidohexyl",
        long_name="Lacto-N-triose (Lc3)",
        chain=["GlcNAc", "Gal", "Glc", "aziHex"],
        linkages=[
            (0, 1, "β1,3", 1, 3),
            (1, 2, "β1,4", 1, 4),
            (3, 2, "tag",  None, 1),
        ],
        branches={},
        rxn_node=2, rxn_pos="C3", rxn_link="α1,3",
        rxn_description=(
            "FucTIII fucosylates O3 of the reducing-end Glc, forming Fucα1,3-Glc. "
            "Product: LNFP V-like trisaccharide."
        ),
        kinetics=dict(
            kcat_s=0.29, kcat_err=0.02,
            KM_mM=4.79, KM_err=0.94,
            kcat_KM=0.06, kcat_KM_err=0.02,
            kcat_KM_units="mM⁻¹s⁻¹",
            normalized_score=ke(0.06),
            paper_verbatim="kcat = 0.29 ± 0.02 s⁻¹; KM = 4.79 ± 0.94 mM; "
                           "kcat/KM = 0.06 ± 0.02 mM⁻¹s⁻¹ (Table 1)",
            confidence="measured kinetics",
        ),
        synthesis=dict(
            active=True, yield_pct=74, brsm_pct=None,
            reaction_time_h=None, donor_equiv=None,
            product_name="LNFP V-like trisaccharide",
        ),
        smiles_status="constructed",
        smiles_pending_reason=None,
        pubchem_cid="53477860",
        substrate_smiles=None,
        substrate_formula="C26H46N4O16",
        notes=(
            "Constructed by replacing reducing-end GlcNAc N-acetyl (NC(C)=O) with OH "
            "in compound 12 SMILES. Net: −C2H2N. Formula C26H46N4O16 verified."
        ),
    ),
    11: dict(
        name="LacNAc-6azidohexyl",
        long_name="N-Acetyllactosamine (LacNAc)",
        chain=["Gal", "GlcNAc", "aziHex"],
        linkages=[
            (0, 1, "β1,4", 1, 4),
            (2, 1, "tag",  None, 1),
        ],
        branches={},
        rxn_node=1, rxn_pos="C3", rxn_link="α1,3",
        rxn_description=(
            "FucTIII fucosylates O3 of the reducing-end GlcNAc, forming Fucα1,3. "
            "Product: Le^x trisaccharide (Galβ1,4(Fucα1,3)GlcNAc)."
        ),
        kinetics=dict(
            kcat_s=0.35, kcat_err=0.03,
            KM_mM=9.61, KM_err=1.58,
            kcat_KM=0.04, kcat_KM_err=0.02,
            kcat_KM_units="mM⁻¹s⁻¹",
            normalized_score=ke(0.04),
            paper_verbatim="kcat = 0.35 ± 0.03 s⁻¹; KM = 9.61 ± 1.58 mM; "
                           "kcat/KM = 0.04 ± 0.02 mM⁻¹s⁻¹ (Table 1)",
            confidence="measured kinetics",
        ),
        synthesis=dict(
            active=True, yield_pct=71, brsm_pct=None,
            reaction_time_h=23, donor_equiv=None,
            product_name="Le^x",
        ),
        smiles_status="constructed",
        smiles_pending_reason=None,
        pubchem_cid="439271",
        substrate_smiles=None,
        substrate_formula="C20H36N4O11",
        notes=(
            "Type-2 disaccharide. 39× worse kcat/KM than compound 12. "
            "SMILES constructed from PubChem CID 439271 ring-form; β-azidohexyl appended "
            "at GlcNAc C1. Formula C20H36N4O11 confirmed by RDKit."
        ),
    ),
    12: dict(
        name="GlcNAcb1-3-LacNAc-6azidohexyl",
        long_name="GlcNAcβ1,3-LacNAc (type-2 trisaccharide)",
        chain=["GlcNAc", "Gal", "GlcNAc", "aziHex"],
        linkages=[
            (0, 1, "β1,3", 1, 3),
            (1, 2, "β1,4", 1, 4),
            (3, 2, "tag",  None, 1),
        ],
        branches={},
        rxn_node=2, rxn_pos="C3", rxn_link="α1,3",
        rxn_description=(
            "FucTIII fucosylates O3 of the reducing-end GlcNAc, forming Fucα1,3. "
            "Product: GlcNAcβ1,3-Le^x. BEST SUBSTRATE."
        ),
        kinetics=dict(
            kcat_s=2.40, kcat_err=0.08,
            KM_mM=1.72, KM_err=0.24,
            kcat_KM=1.40, kcat_KM_err=0.32,
            kcat_KM_units="mM⁻¹s⁻¹",
            normalized_score=ke(1.40),
            paper_verbatim="kcat = 2.40 ± 0.08 s⁻¹; KM = 1.72 ± 0.24 mM; "
                           "kcat/KM = 1.40 ± 0.32 mM⁻¹s⁻¹ ← BEST SUBSTRATE (Table 1)",
            confidence="measured kinetics",
        ),
        synthesis=dict(
            active=True, yield_pct=71, brsm_pct=None,
            reaction_time_h=6, donor_equiv=None,
            product_name="GlcNAcβ1,3-Le^x",
        ),
        smiles_status="verified",
        smiles_pending_reason=None,
        pubchem_cid=None,
        substrate_smiles=None,
        substrate_formula="C28H49N5O16",
        notes=(
            "BEST substrate: NR-end GlcNAc improves kcat 7× and lowers KM 6× vs LacNAc. "
            "35× better kcat/KM than type-1 counterpart (compound 20). "
            "SMILES verified, formula C28H49N5O16 confirmed."
        ),
    ),
    19: dict(
        name="LNB-6azidohexyl",
        long_name="Lacto-N-biose (LNB)",
        chain=["Gal", "GlcNAc", "aziHex"],
        linkages=[
            (0, 1, "β1,3", 1, 3),   # type-1: Galβ1,3GlcNAc
            (2, 1, "tag",  None, 1),
        ],
        branches={},
        rxn_node=1, rxn_pos="C4", rxn_link="α1,4",
        rxn_description=(
            "FucTIII fucosylates O4 of the reducing-end GlcNAc, forming Fucα1,4. "
            "Product: Le^a (Galβ1,3(Fucα1,4)GlcNAc). WORST SUBSTRATE."
        ),
        kinetics=dict(
            kcat_s=0.02, kcat_err=0.00,
            KM_mM=18.14, KM_err=2.00,
            kcat_KM=0.00093, kcat_KM_err=0.00048,
            kcat_KM_units="mM⁻¹s⁻¹",
            normalized_score=ke(0.00093),
            paper_verbatim="kcat = 0.02 ± 0.00 s⁻¹; KM = 18.14 ± 2.00 mM; "
                           "kcat/KM = 0.93 ± 0.48 ×10⁻³ mM⁻¹s⁻¹ ← WORST (Table 1 footnote: ×10⁻³)",
            confidence="measured kinetics",
        ),
        synthesis=dict(
            active=True, yield_pct=47, brsm_pct=77,
            reaction_time_h=48, donor_equiv=None,
            product_name="Le^a",
        ),
        smiles_status="constructed",
        smiles_pending_reason=None,
        pubchem_cid="440994",
        substrate_smiles=None,
        substrate_formula="C20H36N4O11",
        notes=(
            "Type-1 disaccharide (Galβ1,3GlcNAc). Worst substrate. 39× lower kcat/KM than LacNAc. "
            "SMILES constructed from PubChem CID 440994 ring-form; β-azidohexyl appended "
            "at GlcNAc C1. Gal at O3 (not O4). Formula C20H36N4O11 confirmed by RDKit."
        ),
    ),
    20: dict(
        name="GlcNAcb1-3-LNB-6azidohexyl",
        long_name="GlcNAcβ1,3-LNB (type-1 trisaccharide)",
        chain=["GlcNAc", "Gal", "GlcNAc", "aziHex"],
        linkages=[
            (0, 1, "β1,3", 1, 3),
            (1, 2, "β1,3", 1, 3),   # type-1 Galβ1,3GlcNAc
            (3, 2, "tag",  None, 1),
        ],
        branches={},
        rxn_node=2, rxn_pos="C4", rxn_link="α1,4",
        rxn_description=(
            "FucTIII fucosylates O4 of the reducing-end GlcNAc, forming Fucα1,4. "
            "Product: GlcNAcβ1,3-Le^a."
        ),
        kinetics=dict(
            kcat_s=0.24, kcat_err=0.01,
            KM_mM=5.65, KM_err=0.68,
            kcat_KM=0.04, kcat_KM_err=0.02,
            kcat_KM_units="mM⁻¹s⁻¹",
            normalized_score=ke(0.04),
            paper_verbatim="kcat = 0.24 ± 0.01 s⁻¹; KM = 5.65 ± 0.68 mM; "
                           "kcat/KM = 0.04 ± 0.02 mM⁻¹s⁻¹ (Table 1)",
            confidence="measured kinetics",
        ),
        synthesis=dict(
            active=True, yield_pct=55, brsm_pct=89,
            reaction_time_h=42, donor_equiv=None,
            product_name="GlcNAcβ1,3-Le^a",
        ),
        smiles_status="constructed",
        smiles_pending_reason=None,
        pubchem_cid=None,
        substrate_smiles=None,
        substrate_formula="C28H49N5O16",
        notes=(
            "Type-1 trisaccharide. 65× better kcat/KM than LNB alone. "
            "35× worse than type-2 counterpart (compound 12)."
        ),
    ),
    # ── SYNTHESIS SUBSTRATES ─────────────────────────────────────────────────
    3: dict(
        name="LNT-6azidohexyl",
        long_name="Lacto-N-tetraose (LNT)",
        chain=["Gal", "GlcNAc", "Gal", "Glc", "aziHex"],
        linkages=[
            (0, 1, "β1,3", 1, 3),
            (1, 2, "β1,3", 1, 3),
            (2, 3, "β1,4", 1, 4),
            (4, 3, "tag",  None, 1),
        ],
        branches={},
        rxn_node=1, rxn_pos="C4", rxn_link="α1,4",
        rxn_description=(
            "FucTIII fucosylates O4 of the inner GlcNAc, forming Fucα1,4. "
            "Main product: LNFP V (70%). Double fucosylation (2.5 equiv) also gives LNDFH II (53%)."
        ),
        kinetics=dict(
            kcat_s=None, kcat_err=None,
            KM_mM=None, KM_err=None,
            kcat_KM=None, kcat_KM_err=None,
            kcat_KM_units=None,
            normalized_score=_yield_to_score(70, 16),
            paper_verbatim="No kinetic parameters. Preparative yield: 70% → LNFP V (Table 2).",
            confidence="preparative yield, calibrated pseudo-kcat/KM",
        ),
        synthesis=dict(
            active=True, yield_pct=70, brsm_pct=None,
            reaction_time_h=16, donor_equiv=2.5,
            product_name="LNFP V",
        ),
        smiles_status="pending",
        smiles_pending_reason=(
            "PubChem CID 440993 SMILES uses open-chain (aldehyde) reducing-end Glc. "
            "Cannot attach 6-azidohexyl to an aldehyde carbon, ring-form pyranose required. "
            "Needs manual 4-residue ring-form SMILES construction."
        ),
        pubchem_cid="440993",
        substrate_smiles=None,
        substrate_formula="C32H56N4O21",
        notes="LNT = Galβ1,3GlcNAcβ1,3-Galβ1,4Glc. No kinetic parameters reported.",
    ),
    4: dict(
        name="LNnT-6azidohexyl",
        long_name="Lacto-N-neotetraose (LNnT)",
        chain=["Gal", "GlcNAc", "Gal", "Glc", "aziHex"],
        linkages=[
            (0, 1, "β1,4", 1, 4),
            (1, 2, "β1,3", 1, 3),
            (2, 3, "β1,4", 1, 4),
            (4, 3, "tag",  None, 1),
        ],
        branches={},
        rxn_node=3, rxn_pos="C3", rxn_link="α1,3",
        rxn_description=(
            "FucTIII fucosylates O3 of the reducing-end Glc, forming Fucα1,3. "
            "Two products in one pot: LNFP VI (main, 70%) and LNnDFH II (minor di-fucosylated, 10%)."
        ),
        kinetics=dict(
            kcat_s=None, kcat_err=None,
            KM_mM=None, KM_err=None,
            kcat_KM=None, kcat_KM_err=None,
            kcat_KM_units=None,
            normalized_score=_yield_to_score(70, 24),
            paper_verbatim="No kinetic parameters. Preparative yields: 70% LNFP VI + 10% LNnDFH II (Table 2).",
            confidence="preparative yield, calibrated pseudo-kcat/KM (t=24h assumed)",
        ),
        synthesis=dict(
            active=True, yield_pct=70, brsm_pct=None,
            reaction_time_h=24, donor_equiv=None,
            product_name="LNFP VI (major, 70%) + LNnDFH II (minor, 10%)",
        ),
        smiles_status="pending",
        smiles_pending_reason=(
            "PubChem CID 121853 SMILES uses open-chain (aldehyde) reducing-end Glc. "
            "Needs manual 4-residue ring-form SMILES construction."
        ),
        pubchem_cid="121853",
        substrate_smiles=None,
        substrate_formula="C32H56N4O21",
        notes=(
            "LNnT = Galβ1,4GlcNAcβ1,3-Galβ1,4Glc. Two products in one reaction. "
            "NMR confirms fucosylation at reducing-end Glc O3 (LNFP VI). "
            "No kinetic parameters reported."
        ),
    ),
    15: dict(
        name="Fuca1-2-LacNAc-6azidohexyl",
        long_name="Fucα1,2-LacNAc (H-2 antigen precursor)",
        # type-2 chain with Fucα1,2 pre-installed on Gal
        chain=["Gal", "GlcNAc", "aziHex"],
        linkages=[
            (0, 1, "β1,4", 1, 4),
            (2, 1, "tag",  None, 1),
        ],
        branches={
            0: [{"moiety": "Fuc", "linkage": "α1,2",
                 "donor_pos": 1, "acceptor_pos": 2, "side": "above"}],
        },
        rxn_node=1, rxn_pos="C3", rxn_link="α1,3",
        rxn_description=(
            "FucTIII fucosylates O3 of GlcNAc (type-2 chain), forming Fucα1,3. "
            "Pre-existing Fucα1,2 on Gal does not block activity. "
            "Product: Le^y antigen (Fucα1,2-Galβ1,4(Fucα1,3)GlcNAc)."
        ),
        kinetics=dict(
            kcat_s=None, kcat_err=None,
            KM_mM=None, KM_err=None,
            kcat_KM=None, kcat_KM_err=None,
            kcat_KM_units=None,
            normalized_score=_yield_to_score(85, 7.5),
            paper_verbatim="No kinetic parameters. Preparative yield: 85% (94% brsm, 7.5 h) → Le^y antigen (Table 3).",
            confidence="preparative yield, calibrated pseudo-kcat/KM",
        ),
        synthesis=dict(
            active=True, yield_pct=85, brsm_pct=94,
            reaction_time_h=7.5, donor_equiv=1.5,
            product_name="Le^y antigen",
        ),
        smiles_status="pending",
        smiles_pending_reason=None,
        pubchem_cid=None,
        substrate_smiles=None,
        substrate_formula="C26H46N4O15",
        notes=(
            "H-2 antigen precursor (type-2 chain). Highest preparative yield and shortest "
            "reaction time in Table 3, consistent with type-2 GlcNAc C3 being the preferred site. "
            "SMILES assembled from moiety graph via glycan_assembler."
        ),
    ),
    17: dict(
        name="Neu5Aca2-3-LacNAc-6azidohexyl",
        long_name="Neu5Acα2,3-LacNAc (SLe^x precursor)",
        # type-2 chain with Neu5Acα2,3 on Gal
        chain=["Gal", "GlcNAc", "aziHex"],
        linkages=[
            (0, 1, "β1,4", 1, 4),
            (2, 1, "tag",  None, 1),
        ],
        branches={
            0: [{"moiety": "Sia", "linkage": "α2,3",
                 "donor_pos": 2, "acceptor_pos": 3, "side": "above"}],
        },
        rxn_node=1, rxn_pos="C3", rxn_link="α1,3",
        rxn_description=(
            "FucTIII fucosylates O3 of GlcNAc (type-2 chain), forming Fucα1,3. "
            "Neu5Acα2,3 on Gal does NOT block activity (contrast with compound 22). "
            "Product: SLe^x (Neu5Acα2,3-Galβ1,4(Fucα1,3)GlcNAc)."
        ),
        kinetics=dict(
            kcat_s=None, kcat_err=None,
            KM_mM=None, KM_err=None,
            kcat_KM=None, kcat_KM_err=None,
            kcat_KM_units=None,
            normalized_score=_yield_to_score(80, 23),
            paper_verbatim="No kinetic parameters. Preparative yield: 80% (92% brsm, 23 h) → SLe^x (Table 3).",
            confidence="preparative yield, calibrated pseudo-kcat/KM",
        ),
        synthesis=dict(
            active=True, yield_pct=80, brsm_pct=92,
            reaction_time_h=23, donor_equiv=1.5,
            product_name="SLe^x (sialyl Lewis x)",
        ),
        smiles_status="pending",
        smiles_pending_reason=None,
        pubchem_cid=None,
        substrate_smiles=None,
        substrate_formula="C31H53N5O19",
        notes=(
            "SLe^x precursor (type-2 chain). Neu5Acα2,3 is at Gal C3, one residue away "
            "from the GlcNAc C3 reaction site. Contrasts with compound 22 (Sia on type-1 "
            "chain, explicit negative): here Sia is on type-2 chain and enzyme retains "
            "activity. SMILES assembled from moiety graph via glycan_assembler."
        ),
    ),
    21: dict(
        name="Fuca1-2-LNB-6azidohexyl",
        long_name="Fucα1,2-LNB (H-1 antigen precursor)",
        # Backbone: Galβ1,3-GlcNAc-N3  (type-1 chain)
        # Branch:   Fucα1,2 from Gal C2  (drawn above Gal)
        chain=["Gal", "GlcNAc", "aziHex"],
        linkages=[
            (0, 1, "β1,3", 1, 3),
            (2, 1, "tag",  None, 1),
        ],
        branches={
            0: [{"moiety": "Fuc", "linkage": "α1,2",
                 "donor_pos": 1, "acceptor_pos": 2, "side": "above"}],
        },
        rxn_node=1, rxn_pos="C4", rxn_link="α1,4",
        rxn_description=(
            "FucTIII fucosylates O4 of the reducing-end GlcNAc, forming Fucα1,4. "
            "Pre-existing Fucα1,2 at Gal does NOT block activity. "
            "Product: Le^b antigen (Fucα1,2-Galβ1,3(Fucα1,4)GlcNAc)."
        ),
        kinetics=dict(
            kcat_s=None, kcat_err=None,
            KM_mM=None, KM_err=None,
            kcat_KM=None, kcat_KM_err=None,
            kcat_KM_units=None,
            normalized_score=_yield_to_score(51, 46),
            paper_verbatim="No kinetic parameters. Preparative yield: 51% (91% brsm, 46 h) → Le^b antigen (Table 3).",
            confidence="preparative yield, calibrated pseudo-kcat/KM",
        ),
        synthesis=dict(
            active=True, yield_pct=51, brsm_pct=91,
            reaction_time_h=46, donor_equiv=1.5,
            product_name="Le^b antigen",
        ),
        smiles_status="pending",
        smiles_pending_reason=(
            "No PubChem CID for the azide-tagged form. H-1 antigen free disaccharide "
            "(Fucα1,2-Galβ1,3GlcNAc) structure is known but branched, requires explicit "
            "construction of the β-GlcNAc C1–O–(CH₂)₆N₃ linkage with Fucα1,2 intact at Gal."
        ),
        pubchem_cid=None,
        substrate_smiles=None,
        substrate_formula=None,
        notes=(
            "H-1 antigen precursor. Confirms pre-existing α1,2-Fuc on Gal does not "
            "block FucTIII. High brsm (91%) confirms clean, selective reaction."
        ),
    ),
    # ── EXPLICIT NEGATIVE ────────────────────────────────────────────────────
    22: dict(
        name="Neu5Aca2-3-LNB-6azidohexyl",
        long_name="Neu5Acα2,3-LNB (sialylated type-1 chain)",
        # Backbone: Galβ1,3-GlcNAc-N3
        # Branch:   Neu5Acα2,3 from Gal C3  (drawn above Gal)
        chain=["Gal", "GlcNAc", "aziHex"],
        linkages=[
            (0, 1, "β1,3", 1, 3),
            (2, 1, "tag",  None, 1),
        ],
        branches={
            0: [{"moiety": "Sia", "linkage": "α2,3",
                 "donor_pos": 2, "acceptor_pos": 3, "side": "above"}],
        },
        rxn_node=None, rxn_pos=None, rxn_link=None,
        rxn_description=(
            "No reaction (explicit negative). Neu5Acα2,3 occupies Gal C3, blocking "
            "FucTIII recognition of the Galβ1,3GlcNAc type-1 chain. "
            "This leaves GlcNAc O4, the intended fucosylation site, inaccessible."
        ),
        kinetics=dict(
            kcat_s=None, kcat_err=None,
            KM_mM=None, KM_err=None,
            kcat_KM=None, kcat_KM_err=None,
            kcat_KM_units=None,
            normalized_score=0.0,
            paper_verbatim="No detectable product. Explicit negative (Table 3).",
            confidence="explicit negative",
        ),
        synthesis=dict(
            active=False, yield_pct=0, brsm_pct=None,
            reaction_time_h=None, donor_equiv=None,
            product_name=None,
        ),
        smiles_status="pending",
        smiles_pending_reason=None,
        pubchem_cid=None,
        substrate_smiles=None,
        substrate_formula="C31H53N5O19",
        notes=(
            "Explicit negative. Neu5Acα2,3 at Gal C3 blocks enzyme recognition, "
            "the blocked site is Gal C3 (recognition), with the consequence that "
            "GlcNAc O4 (fucosylation target) becomes inaccessible. "
            "SMILES assembled from moiety graph via glycan_assembler."
        ),
    ),
    # ── SYNTHESIS PRODUCT (reference) ────────────────────────────────────────
    7: dict(
        name="LNnDFH-II-6azidohexyl",
        long_name="LNnDFH II (di-fucosylated hexasaccharide)",
        # Simplified linear backbone, true topology has two Fuc branches; see notes
        chain=["GlcNAc", "Gal", "GlcNAc", "Gal", "Glc", "aziHex"],
        linkages=[
            (0, 1, "α1,3", 1, 3),
            (1, 2, "β1,4", 1, 4),
            (2, 3, "β1,3", 1, 3),
            (3, 4, "β1,4", 1, 4),
            (5, 4, "tag",  None, 1),
        ],
        branches={},   # branching approximated in linear form for this product
        rxn_node=None, rxn_pos=None, rxn_link=None,
        rxn_description="Synthesis product. No further fucosylation assayed.",
        kinetics=dict(
            kcat_s=None, kcat_err=None,
            KM_mM=None, KM_err=None,
            kcat_KM=None, kcat_KM_err=None,
            kcat_KM_units=None,
            normalized_score=None,
            paper_verbatim="Synthesis product: 65% yield (88% brsm), from LNFP VI, 2.5 equiv GDP-Fuc (Table 2).",
            confidence="synthesis product, rxn_node=None, not a prediction target",
        ),
        synthesis=dict(
            active=None, yield_pct=65, brsm_pct=88,
            reaction_time_h=None, donor_equiv=2.5,
            product_name="LNnDFH II",
        ),
        smiles_status="source_free",
        smiles_pending_reason=(
            "PubChem CID 172638591 has a SMILES for the free glycoside. "
            "Azide-tagged form not independently verified."
        ),
        pubchem_cid="172638591",
        substrate_smiles=(
            "C[C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O[C@H]2[C@@H]([C@H](O[C@H]"
            "([C@@H]2O[C@@H]3[C@H]([C@@H]([C@@H]([C@@H](O3)CO)O)O)O)CO)O[C@@H]4"
            "[C@@H]([C@@H](O[C@@H]([C@H]4O)O[C@H]5[C@@H](O[C@@H]([C@H]([C@@H]5"
            "O[C@@H]6[C@@H]([C@H]([C@H]([C@H](O6)C)O)O)O)O)OCCCCCCN=[N+]=[N-])"
            "CO)CO)O)NC(=O)C)O)O)O"
        ),
        substrate_formula="C44H76N4O29",
        notes=(
            "LNnDFH II = di-fucosylated product from LNnT. True topology has Fuc branches at "
            "two positions; this cartoon is a simplified linear approximation. "
            "Stereo/linkage of both Fuc residues needs confirmation from paper NMR data."
        ),
    ),
}

# Normalization parameters (exported for use in report)
ENERGY_PARAMS = dict(x0=X0, xmax=XMAX, scaling_factor=SF)


# ── Auto-compute substrate SMILES from moiety graphs ─────────────────────────
# Compounds 22 (Sia branch) and 7 (complex synthesis product) keep their
# pre-set substrate_smiles.  All others are assembled from moiety SMILES.
try:
    from reports.glycan_assembler import build_compound as _build_compound
    from reports.glycan_assembler import build_product as _build_product
    from rdkit.Chem import rdMolDescriptors as _rmd
    from rdkit import Chem as _Chem

    for _cn, _c in COMPOUNDS.items():
        if _c.get("substrate_smiles") is not None:
            # Pre-set compounds (7, 22): still compute product if site is known
            _rn, _rp = _c.get("rxn_node"), _c.get("rxn_pos")
            if _rn is not None and _rp is not None and _c.get("product_smiles") is None:
                try:
                    _, _mol_mapped, _ = _build_compound(
                        _c["chain"], _c["linkages"], _c.get("branches", {}))
                    _prod_mol = _build_product(_mol_mapped, _rn, _rp)
                    _c["product_smiles"] = _Chem.MolToSmiles(_prod_mol)
                except Exception:
                    _c["product_smiles"] = None
            continue
        try:
            _smi, _mol_mapped, _ = _build_compound(
                _c["chain"], _c["linkages"], _c.get("branches", {}))
            _formula = _rmd.CalcMolFormula(_Chem.AddHs(_Chem.MolFromSmiles(_smi)))
            _c["substrate_smiles"] = _smi
            _c["substrate_formula"] = _formula
            _c["smiles_status"] = "assembled"
            _c["smiles_pending_reason"] = None
            # Build product for compounds with a known reaction site
            _rn, _rp = _c.get("rxn_node"), _c.get("rxn_pos")
            if _rn is not None and _rp is not None:
                try:
                    _prod_mol = _build_product(_mol_mapped, _rn, _rp)
                    _c["product_smiles"] = _Chem.MolToSmiles(_prod_mol)
                except Exception:
                    _c["product_smiles"] = None
            else:
                _c["product_smiles"] = None
        except Exception as _e:
            _c["smiles_status"] = "pending"
            _c["smiles_pending_reason"] = f"Assembly error: {_e}"
            _c["product_smiles"] = None
except ImportError:
    pass
