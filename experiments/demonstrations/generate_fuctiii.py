"""Generate visual assets for the FucTIII regioselectivity demonstration."""
from __future__ import annotations

import json
import re
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "assets" / "demonstrations" / "fuctiii"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, RWMol
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFMCS
from PIL import Image

# ── palette ───────────────────────────────────────────────────────────────────
BG       = "#f9f9f9"
GREEN    = "#2e7d32"
RED      = "#b71c1c"
BLUE     = "#1565c0"
GREY     = "#777777"
BG_RGB   = (0.976, 0.976, 0.976)

GREEN_RGB  = (0.18, 0.49, 0.20)
RED_RGB    = (0.72, 0.11, 0.11)
BLUE_RGB   = (0.08, 0.40, 0.74)
ORANGE_RGB = (0.85, 0.42, 0.05)

# Sugar unit residue colors, light pastels so black bonds/labels read through
GAL_RGB    = (0.70, 0.93, 0.76)   # light mint, galactose
GLCNAC_RGB = (0.85, 0.72, 0.96)   # light lavender, GlcNAc
FUC_RGB    = (0.99, 0.82, 0.52)   # light peach, fucose

SCREEN_JSON = ROOT / "experiments/background_screen/results/fuctiii_regioselectivity.json"

# Fixed visual bond length in pixels, same value for every molecule panel so
# individual sugar rings appear at the same size regardless of how many sugars
# are in the panel.
BOND_PX = 28


def _median_bond_len_2d(mol: Chem.Mol) -> float:
    """Median bond length in the mol's 2D conformer (coordinate units)."""
    conf = mol.GetConformer()
    lens = []
    for bond in mol.GetBonds():
        p1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        p2 = conf.GetAtomPosition(bond.GetEndAtomIdx())
        lens.append(((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5)
    return sorted(lens)[len(lens) // 2] if lens else 1.5


def _mol_canvas_size(mol: Chem.Mol, bond_px: int = BOND_PX,
                     pad: float = 0.20) -> tuple[int, int]:
    """Pixel canvas (w, h) needed to draw mol at bond_px pixels per bond."""
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    dx = max(max(xs) - min(xs), 0.01)
    dy = max(max(ys) - min(ys), 0.01)
    scale = bond_px / _median_bond_len_2d(mol)
    w = int(dx * scale * (1 + 2 * pad)) + 60
    h = int(dy * scale * (1 + 2 * pad)) + 60
    return max(w, 120), max(h, 80)


# ── SMARTS → regular mol helpers ──────────────────────────────────────────────

def _smarts_template_to_mol(template_mol: Chem.Mol) -> Chem.Mol:
    """Convert a SMARTS reaction template mol to a regular mol.

    Strips atom map numbers and converts query atoms ([#6], [#8] …) to
    regular element atoms.  Keeps valences satisfied via full sanitization.
    """
    rw = RWMol()
    idx_map: dict[int, int] = {}
    for atom in template_mol.GetAtoms():
        a = Chem.Atom(atom.GetAtomicNum())
        a.SetAtomMapNum(0)
        a.SetChiralTag(atom.GetChiralTag())
        idx_map[atom.GetIdx()] = rw.AddAtom(a)
    for bond in template_mol.GetBonds():
        bt = bond.GetBondType()
        if bt in (Chem.BondType.UNSPECIFIED, Chem.BondType.ZERO):
            bt = Chem.BondType.SINGLE
        rw.AddBond(idx_map[bond.GetBeginAtomIdx()],
                   idx_map[bond.GetEndAtomIdx()], bt)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        Chem.SanitizeMol(
            rw,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    AllChem.Compute2DCoords(rw)
    return rw.GetMol()


def _add_explicit_oh(mol: Chem.Mol) -> Chem.Mol:
    """Return mol with explicit H atoms added only to hydroxyl oxygens.

    C-H bonds stay implicit; O-H bonds become explicit atoms in the graph
    so they render visibly in the drawing.
    """
    rw = RWMol(mol)
    # Collect O-atom indices first; modifying the mol invalidates atom iterators
    o_atoms = [(a.GetIdx(), a.GetTotalNumHs())
               for a in rw.GetAtoms()
               if a.GetAtomicNum() == 8 and a.GetTotalNumHs() > 0]
    for o_idx, _ in o_atoms:
        h_idx = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(o_idx, h_idx, Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(o_idx).SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(
            rw,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    except Exception:
        pass
    AllChem.Compute2DCoords(rw)
    return rw.GetMol()


# ── molecule rendering ────────────────────────────────────────────────────────

def _pil_from_mol(mol: Chem.Mol, size=(400, 280),
                  highlight_atoms=None, atom_colors=None,
                  highlight_bonds=None, bond_colors=None,
                  remove_h=True, padding=0.08, bond_px=None) -> Image.Image:
    """Render mol to PIL.

    Uses existing 2-D conformer if present (preserves spatial orientation).
    Strips atom map numbers.
    """
    if remove_h:
        mol = Chem.RemoveHs(mol)

    rw = RWMol(mol)
    for a in rw.GetAtoms():
        a.SetAtomMapNum(0)
    mol = rw.GetMol()

    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)

    if bond_px is not None:
        size = _mol_canvas_size(mol, bond_px=bond_px, pad=padding)

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.padding = padding
    opts.bondLineWidth = 1.8
    if bond_px is not None:
        opts.fixedBondLength = float(bond_px)
    opts.clearBackground = True

    ha = [x for x in (highlight_atoms or []) if x is not None]
    hb = [x for x in (highlight_bonds or []) if x is not None]
    ac = {k: v for k, v in (atom_colors or {}).items() if k is not None}
    bc = {k: v for k, v in (bond_colors or {}).items() if k is not None}

    drawer.DrawMolecule(mol, highlightAtoms=ha, highlightBonds=hb,
                        highlightAtomColors=ac, highlightBondColors=bc)
    drawer.FinishDrawing()
    return _fix_bg(Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA"))


def _pil_from_rxn(rxn, sub_w: int, h: int) -> Image.Image:
    img = Draw.ReactionToImage(rxn, subImgSize=(sub_w, h), useSVG=False)
    return _fix_bg(img.convert("RGBA"))


def _fix_bg(img: Image.Image) -> Image.Image:
    bg = (int(BG_RGB[0]*255), int(BG_RGB[1]*255), int(BG_RGB[2]*255), 255)
    data = np.array(img)
    white = (data[:,:,0] > 245) & (data[:,:,1] > 245) & (data[:,:,2] > 245)
    data[white] = bg
    return Image.fromarray(data)


# ── operator panel B ──────────────────────────────────────────────────────────

def _operator_pil(d_op_smirks: str) -> Image.Image:
    """Render the EVODEX-D operator at BOND_PX scale (matches molecule panels).

    Reactant: H–O–C fragment with O and C highlighted blue.
    Product: fucose ring (C:103 removed, substrate side) with glycosidic O blue.
    """
    from PIL import ImageDraw

    rxn = AllChem.ReactionFromSmarts(d_op_smirks)
    r_template = rxn.GetReactants()[0]
    p_template = rxn.GetProducts()[0]

    def _map_idx(template, mapnum):
        for a in template.GetAtoms():
            if a.GetAtomMapNum() == mapnum:
                return a.GetIdx()
        return None

    r_o_idx = _map_idx(r_template, 113)
    r_c_idx = _map_idx(r_template, 103)
    p_o_idx = _map_idx(p_template, 113)
    p_c_idx = _map_idx(p_template, 103)

    # Build reactant mol
    r_mol = _smarts_template_to_mol(r_template)

    # Build product mol: remove C:103 (substrate side), add explicit OHs
    p_mol_raw = _smarts_template_to_mol(p_template)
    p_mol_rw = RWMol(p_mol_raw)
    if p_c_idx is not None:
        p_mol_rw.RemoveAtom(p_c_idx)
        try:
            Chem.SanitizeMol(p_mol_rw)
        except Exception:
            pass
    p_mol_draw = _add_explicit_oh(p_mol_rw.GetMol())
    # Adjust O index after atom removal
    p_o_draw = (p_o_idx - 1 if (p_c_idx is not None and p_o_idx is not None
                                  and p_o_idx > p_c_idx)
                else p_o_idx)

    # Derive canvas sizes from molecular extent at BOND_PX scale
    r_size = _mol_canvas_size(r_mol, bond_px=BOND_PX, pad=0.10)
    p_size = _mol_canvas_size(p_mol_draw, bond_px=BOND_PX, pad=0.16)
    h = max(r_size[1], p_size[1])
    rw, pw = r_size[0], p_size[0]
    aw = 52  # arrow strip

    # Reactant: O and C highlighted blue
    ha_r = [i for i in [r_o_idx, r_c_idx] if i is not None]
    ac_r = {i: BLUE_RGB for i in ha_r}
    hb_r, bc_r = [], {}
    if r_o_idx is not None and r_c_idx is not None:
        bond = r_mol.GetBondBetweenAtoms(r_o_idx, r_c_idx)
        if bond:
            hb_r.append(bond.GetIdx())
            bc_r[bond.GetIdx()] = BLUE_RGB

    # Product: glycosidic O highlighted blue
    ha_p = [p_o_draw] if p_o_draw is not None else []
    ac_p = {p_o_draw: BLUE_RGB} if p_o_draw is not None else {}

    img_r = _pil_from_mol(r_mol, size=(rw, h), bond_px=BOND_PX,
                          highlight_atoms=ha_r, atom_colors=ac_r,
                          highlight_bonds=hb_r, bond_colors=bc_r,
                          remove_h=False, padding=0.10)
    img_p = _pil_from_mol(p_mol_draw, size=(pw, h), bond_px=BOND_PX,
                          highlight_atoms=ha_p, atom_colors=ac_p,
                          remove_h=False, padding=0.16)

    # Stitch: reactant | arrow | product
    bg_px = (int(BG_RGB[0]*255), int(BG_RGB[1]*255), int(BG_RGB[2]*255), 255)
    combined = Image.new("RGBA", (rw + aw + pw, h), bg_px)
    combined.paste(img_r, (0, 0))
    combined.paste(img_p, (rw + aw, 0))

    draw = ImageDraw.Draw(combined)
    ax1, ax2 = rw + 6, rw + aw - 8
    ay = h // 2
    draw.line([(ax1, ay), (ax2, ay)], fill="#555555", width=2)
    head = 7
    draw.polygon([(ax2 + head, ay), (ax2, ay - 4), (ax2, ay + 4)], fill="#555555")

    return _fix_bg(combined)


# ── sugar unit coloring ───────────────────────────────────────────────────────

def _classify_pyranose(mol: Chem.Mol, ring_set: set) -> str:
    """Return 'GlcNAc', 'Fucose', or 'Gal' for a 6-membered O-containing ring."""
    # GlcNAc: a ring carbon has a nitrogen neighbor (the NHAc group)
    for idx in ring_set:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetAtomicNum() == 7:
                return 'GlcNAc'
    # Fucose: a ring carbon has a methyl branch (C neighbor with no further heavy atoms)
    for idx in ring_set:
        if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6:
            continue
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() in ring_set or nb.GetAtomicNum() != 6:
                continue
            # nb is an exo-cyclic C, is it a methyl (CH3, no further heavy neighbors)?
            heavy_nbs = [n for n in nb.GetNeighbors()
                         if n.GetAtomicNum() > 1 and n.GetIdx() != idx]
            if not heavy_nbs:
                return 'Fucose'
    return 'Gal'


def _sugar_unit_atom_colors(mol: Chem.Mol) -> dict[int, tuple]:
    """Return {atom_idx: (R,G,B)} keyed by sugar residue assignment.

    Uses ring detection: each 6-membered O-containing ring is classified,
    then flood-filled into branches (stopping at other ring atoms).
    Atoms not reachable from any pyranose ring are left uncolored.
    """
    ring_info = mol.GetRingInfo()
    pyranose_rings = [set(r) for r in ring_info.AtomRings()
                      if len(r) == 6 and
                      any(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in r)]
    if not pyranose_rings:
        return {}

    all_ring_atoms = set().union(*pyranose_rings)
    palette = {'Gal': GAL_RGB, 'GlcNAc': GLCNAC_RGB, 'Fucose': FUC_RGB}
    atom_colors: dict[int, tuple] = {}

    for ring_set in pyranose_rings:
        col = palette[_classify_pyranose(mol, ring_set)]
        other_ring_atoms = all_ring_atoms - ring_set
        included: set[int] = set(ring_set)
        queue = list(ring_set)
        while queue:
            idx = queue.pop()
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                ni = nb.GetIdx()
                if ni not in included and ni not in other_ring_atoms:
                    included.add(ni)
                    queue.append(ni)
        for idx in included:
            atom_colors[idx] = col

    return atom_colors


def _apply_sugar_colors(mol: Chem.Mol, override: dict[int, tuple] | None = None
                        ) -> tuple[list, dict, list, dict]:
    """Build highlight args for _pil_from_mol with per-sugar-unit coloring.

    override: {atom_idx: color} to apply on top (for reactive site markers).
    Returns (ha, ac, hb, bc).
    """
    ac = _sugar_unit_atom_colors(mol)
    if override:
        ac.update(override)
    ha = list(ac.keys())

    hb, bc = [], {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        c1, c2 = ac.get(a1), ac.get(a2)
        if c1 is not None and c1 == c2:   # same-residue bond: color it
            hb.append(bond.GetIdx())
            bc[bond.GetIdx()] = c1
    return ha, ac, hb, bc


def _glcnac_fucose_bridge_atoms(mol: Chem.Mol, atom_colors: dict[int, tuple]
                                ) -> tuple[int | None, int | None]:
    """Find the glycosidic O and accepting C between GlcNAc and Fucose domains.

    Returns (glycosidic_O_idx, accepting_C_idx) or (None, None) if not found.
    """
    glcnac = {i for i, c in atom_colors.items() if c == GLCNAC_RGB}
    fucose = {i for i, c in atom_colors.items() if c == FUC_RGB}
    for idx in glcnac | fucose:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 8:
            continue
        nbs = [nb.GetIdx() for nb in atom.GetNeighbors()]
        in_glcnac = [i for i in nbs if i in glcnac]
        in_fucose = [i for i in nbs if i in fucose]
        if in_glcnac and in_fucose:
            accepting_c = next(
                (i for i in in_glcnac if mol.GetAtomWithIdx(i).GetAtomicNum() == 6),
                None)
            return idx, accepting_c
    return None, None


def _prod_pil(sub_noh: Chem.Mol, prod_mol: Chem.Mol, size=(400, 280),
              override: dict[int, tuple] | None = None,
              bond_px: int | None = None) -> Image.Image:
    """Render product aligned to substrate, colored by sugar residue."""
    prod_noh = Chem.RemoveHs(prod_mol)
    rw = RWMol(prod_noh)
    for a in rw.GetAtoms():
        a.SetAtomMapNum(0)
    prod_noh = rw.GetMol()

    AllChem.Compute2DCoords(prod_noh)
    try:
        AllChem.GenerateDepictionMatching2DStructure(prod_noh, sub_noh)
    except Exception:
        pass

    # Find glycosidic bridge atoms for the blue EVODEX-C/O highlight
    base_colors = _sugar_unit_atom_colors(prod_noh)
    bridge_o, bridge_c = _glcnac_fucose_bridge_atoms(prod_noh, base_colors)
    bridge_override = {i: BLUE_RGB for i in [bridge_o, bridge_c] if i is not None}
    if override:
        bridge_override.update(override)

    ha, ac, hb, bc = _apply_sugar_colors(prod_noh, override=bridge_override)
    return _pil_from_mol(prod_noh, size=size, bond_px=bond_px,
                         highlight_atoms=ha, atom_colors=ac,
                         highlight_bonds=hb, bond_colors=bc,
                         remove_h=False)


# ── load training data ────────────────────────────────────────────────────────

def _load_training() -> dict:
    from reports.fuctiii_compound_data import COMPOUNDS
    from experiments.model_selection.datasets.fuctiii_regioselectivity import (
        actual_fucosylation_site, _natural_substrates, _explicit_h_reaction,
    )
    from esorex.reaction_preparation import prepare_reaction
    from esorex.generate_mechanistic_tree import generate_mechanistic_tree
    from esorex.generate_mechanistic_tree import _collect_operators_by_level
    from esorex.label_substrate import label_substrate

    natural = _natural_substrates()
    lacnac_nat = natural["LacNAc"]

    rxn_smi = _explicit_h_reaction(f"{lacnac_nat['sub']}>>{lacnac_nat['prod']}")
    rxn = prepare_reaction(rxn_smi)
    trees = generate_mechanistic_tree([rxn])
    tree = trees[0]
    ops = _collect_operators_by_level(tree)
    d_op_smirks = ops["D"][0]
    d_rxn_op = AllChem.ReactionFromSmarts(d_op_smirks)

    compound_ids = {"LacNAc": 11, "LNB": 19}
    result = {}

    for name, info in natural.items():
        cid = compound_ids[name]
        sub_smi = info["sub"]
        prod_smi = info["prod"]
        pos_map = actual_fucosylation_site(prod_smi)

        sub_mol_h = Chem.AddHs(Chem.MolFromSmiles(sub_smi))
        rw = RWMol(sub_mol_h)
        for atom in rw.GetAtoms():
            atom.SetAtomMapNum(0)
        sub_stripped = rw.GetMol()

        labels = label_substrate(sub_stripped, d_rxn_op)

        sites = []
        for lab in labels:
            o_idx = next(idx for idx, mn in lab["mapped_atoms"] if mn == 113)
            c_idx = next((idx for idx, mn in lab["mapped_atoms"] if mn == 103), None)
            orig_o_map = sub_mol_h.GetAtomWithIdx(o_idx).GetAtomMapNum()
            is_pos = (orig_o_map == pos_map)
            sites.append({
                "o_idx": o_idx, "c_idx": c_idx,
                "orig_o_map": orig_o_map, "is_positive": is_pos,
            })

        hyp_products = d_rxn_op.RunReactants((sub_stripped,))

        # Compute sub_noh coords ONCE, reused across all panels for this compound
        sub_noh = Chem.RemoveHs(sub_mol_h)
        AllChem.Compute2DCoords(sub_noh)
        map_to_noh = {atom.GetAtomMapNum(): atom.GetIdx()
                      for atom in sub_noh.GetAtoms()}

        pos_o_orig = next(s["orig_o_map"] for s in sites if s["is_positive"])
        pos_noh_idx = map_to_noh.get(pos_o_orig)

        pos_site = next(s for s in sites if s["is_positive"])
        pos_c_noh_idx = None
        if pos_site["c_idx"] is not None:
            pos_c_noh_idx = map_to_noh.get(
                sub_mol_h.GetAtomWithIdx(pos_site["c_idx"]).GetAtomMapNum())

        neg_sites = [s for s in sites if not s["is_positive"]]

        # Clean product SMILES
        prod_clean = re.sub(r':\d+', '', prod_smi)
        prod_clean = re.sub(
            r'\[O\]\[CH2\]\[CH2\]\[CH2\]\[CH2\]\[CH2\]\[CH2\]'
            r'\[N\]=\[N\+\]=\[N-\]', 'O', prod_clean)
        prod_mol = Chem.MolFromSmiles(prod_clean)

        result[name] = {
            "sub_noh": sub_noh,
            "prod_mol": prod_mol,
            "sites": sites,
            "pos_noh_idx": pos_noh_idx,
            "pos_c_noh_idx": pos_c_noh_idx,
            "neg_sites": neg_sites,
            "map_to_noh": map_to_noh,
            "sub_mol_h": sub_mol_h,
            "hyp_products": hyp_products,
            "d_op_smirks": d_op_smirks,
            "d_rxn_op": d_rxn_op,
            "tree": tree,
        }

    return result


# ── figure 1: 4-panel sites figure ───────────────────────────────────────────

def write_sites_figure_svg(training: dict, out_path: Path) -> None:
    lac        = training["LacNAc"]
    sub_noh    = lac["sub_noh"]          # has pre-computed 2D coords
    prod_mol   = lac["prod_mol"]
    pos_noh    = lac["pos_noh_idx"]
    pos_c_noh  = lac["pos_c_noh_idx"]
    neg_sites  = lac["neg_sites"]
    map_to_noh = lac["map_to_noh"]
    sub_mol_h  = lac["sub_mol_h"]
    hyp_prods  = lac["hyp_products"]
    sites      = lac["sites"]
    d_op_smirks = lac["d_op_smirks"]

    # Negative site: pick the second one for visual variety
    neg0 = neg_sites[min(1, len(neg_sites) - 1)]
    neg_o_noh = map_to_noh.get(neg0["orig_o_map"])
    neg_c_noh = None
    if neg0["c_idx"] is not None:
        neg_c_noh = map_to_noh.get(
            sub_mol_h.GetAtomWithIdx(neg0["c_idx"]).GetAtomMapNum())

    # Hypothetical product for the chosen negative site
    hyp_mol = None
    for si, s in enumerate(sites):
        if s["orig_o_map"] == neg0["orig_o_map"] and si < len(hyp_prods):
            try:
                p = hyp_prods[si][0]
                Chem.SanitizeMol(p)
                hyp_mol = p
            except Exception:
                pass
            break

    # ── render all panels at BOND_PX fixed scale ──────────────────────────────
    def _site_oc_noh(site):
        o = map_to_noh.get(site["orig_o_map"])
        c = (map_to_noh.get(sub_mol_h.GetAtomWithIdx(site["c_idx"]).GetAtomMapNum())
             if site["c_idx"] is not None else None)
        return [i for i in [o, c] if i is not None]

    all_oc_idxs = []
    for s in sites:
        all_oc_idxs.extend(_site_oc_noh(s))
    all_o_idxs = [i for i in [map_to_noh.get(s["orig_o_map"]) for s in sites]
                  if i is not None]

    # Panel A
    override_a = {i: BLUE_RGB for i in all_oc_idxs}
    ha_a, ac_a, hb_a, bc_a = _apply_sugar_colors(sub_noh, override=override_a)
    img_A = _pil_from_mol(sub_noh, highlight_atoms=ha_a, atom_colors=ac_a,
                          highlight_bonds=hb_a, bond_colors=bc_a, bond_px=BOND_PX)

    # Panel B
    img_B = _operator_pil(d_op_smirks)

    # Panel C substrate
    override_c: dict[int, tuple] = {idx: BLUE_RGB
                                    for idx in [pos_noh, pos_c_noh]
                                    if idx is not None}
    ha_c, ac_c, hb_c, bc_c = _apply_sugar_colors(sub_noh, override=override_c)
    img_Csub = _pil_from_mol(sub_noh, highlight_atoms=ha_c, atom_colors=ac_c,
                             highlight_bonds=hb_c, bond_colors=bc_c, bond_px=BOND_PX)

    # Panel D substrate
    override_d: dict[int, tuple] = {idx: BLUE_RGB
                                    for idx in [neg_o_noh, neg_c_noh]
                                    if idx is not None}
    ha_d, ac_d, hb_d, bc_d = _apply_sugar_colors(sub_noh, override=override_d)
    img_Dsub = _pil_from_mol(sub_noh, highlight_atoms=ha_d, atom_colors=ac_d,
                             highlight_bonds=hb_d, bond_colors=bc_d, bond_px=BOND_PX)

    # Products, bond_px propagated so product rings match substrate scale
    img_Cprod = _prod_pil(sub_noh, prod_mol, bond_px=BOND_PX) if prod_mol else None
    img_Dprod = _prod_pil(sub_noh, hyp_mol,  bond_px=BOND_PX) if hyp_mol  else None

    # ── layout, derive figure size from PIL image dimensions ─────────────────
    # All substrate panels (A, C-sub, D-sub) use the same mol → same PIL size.
    sub_w, sub_h = img_A.size

    prod_pils = [p for p in [img_Cprod, img_Dprod] if p is not None]
    prod_w = max(p.size[0] for p in prod_pils) if prod_pils else sub_w
    prod_h = max(p.size[1] for p in prod_pils) if prod_pils else sub_h

    op_w, op_h = img_B.size

    DPI     = 100
    MARGIN  = 24    # outer margin in pixels
    TITLE_H = 42    # suptitle area
    LBL_H   = 18    # per-panel label row
    GAP     = 14    # gap between rows
    ARR_W   = 70    # arrow column

    # Right column: sized to the wider of operator vs product
    right_w = max(op_w, prod_w)

    # Row content heights
    row0_h  = max(sub_h, op_h)
    row12_h = max(sub_h, prod_h)

    # Y positions from bottom (0 = bottom of figure)
    y2 = MARGIN
    y1 = y2 + row12_h + GAP + LBL_H
    y0 = y1 + row12_h + GAP + LBL_H

    fig_w_px = MARGIN + sub_w + ARR_W + right_w + MARGIN
    fig_h_px = y0 + row0_h + LBL_H + TITLE_H + MARGIN

    fig = plt.figure(figsize=(fig_w_px / DPI, fig_h_px / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)

    # X positions
    x_left  = MARGIN
    x_arr   = MARGIN + sub_w
    x_right = MARGIN + sub_w + ARR_W

    def _rect(x, y, w, h):
        """Pixel coords → [left, bottom, width, height] in figure fraction."""
        return [x / fig_w_px, y / fig_h_px, w / fig_w_px, h / fig_h_px]

    def _center_y(row_y, row_h, img_h):
        return row_y + max(0, (row_h - img_h) // 2)

    def _center_x(col_x, col_w, img_w):
        return col_x + max(0, (col_w - img_w) // 2)

    def _mol_ax(x, y, img, label=""):
        ax = fig.add_axes(_rect(x, y, img.size[0], img.size[1]))
        ax.imshow(np.array(img), interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if label:
            ax.text(0.01, 0.98, label, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", color="#222",
                    va="top", ha="left", fontfamily="Helvetica", zorder=5)

    def _lbl(x, y, text, color="#333"):
        fig.text(x / fig_w_px, y / fig_h_px, text,
                 fontsize=9, color=color, fontweight="bold",
                 va="bottom", ha="left")

    def _arrow_ax(x, y, w, h, color, label=""):
        ax = fig.add_axes(_rect(x, y, w, h))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.annotate("", xy=(0.85, 0.50), xytext=(0.15, 0.50),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.32,head_length=0.22",
                        color=color, lw=2.2))
        if label:
            ax.text(0.50, 0.66, label, ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold",
                    transform=ax.transAxes)

    # ── Row 0: A (substrate, all sites) + B (operator) ────────────────────────
    ay0 = _center_y(y0, row0_h, sub_h)
    _lbl(x_left, y0 + row0_h + 2,
         f"LacNAc, {len(all_o_idxs)} OH sites match the EVODEX-D operator",
         BLUE)
    _mol_ax(x_left, ay0, img_A, label="A")

    bx = _center_x(x_right, right_w, op_w)
    by = _center_y(y0, row0_h, op_h)
    _lbl(bx, y0 + row0_h + 2,
         "EVODEX-D operator, acceptor C–OH → O-linked α-L-fucose", "#555")
    _mol_ax(bx, by, img_B, label="B")

    # ── Row 1: C-sub → C-prod ─────────────────────────────────────────────────
    cy1 = _center_y(y1, row12_h, sub_h)
    _lbl(x_left, y1 + row12_h + 2, "C3-GlcNAc, reactive site (positive)", GREEN)
    _mol_ax(x_left, cy1, img_Csub, label="C")
    _arrow_ax(x_arr, y1, ARR_W, row12_h, GREEN, "observed")
    if img_Cprod:
        cpx = _center_x(x_right, right_w, img_Cprod.size[0])
        cpy = _center_y(y1, row12_h, img_Cprod.size[1])
        _lbl(cpx, y1 + row12_h + 2, "Fucα1,3-LacNAc, actual product", GREEN)
        _mol_ax(cpx, cpy, img_Cprod)

    # ── Row 2: D-sub → D-prod ─────────────────────────────────────────────────
    dy2 = _center_y(y2, row12_h, sub_h)
    _lbl(x_left, y2 + row12_h + 2, "Adjacent OH, inferred negative", RED)
    _mol_ax(x_left, dy2, img_Dsub, label="D")
    _arrow_ax(x_arr, y2, ARR_W, row12_h, RED, "not observed")
    if img_Dprod:
        dpx = _center_x(x_right, right_w, img_Dprod.size[0])
        dpy = _center_y(y2, row12_h, img_Dprod.size[1])
        _lbl(dpx, y2 + row12_h + 2,
             "Hypothetical product, inferred not to occur", RED)
        _mol_ax(dpx, dpy, img_Dprod)

    # ── row dividers ──────────────────────────────────────────────────────────
    for y_px in [(y2 + row12_h + GAP / 2), (y1 + row12_h + GAP / 2)]:
        fig.add_artist(plt.Line2D(
            [MARGIN / fig_w_px, 1 - MARGIN / fig_w_px],
            [y_px / fig_h_px, y_px / fig_h_px],
            transform=fig.transFigure, color="#dddddd", lw=0.9, zorder=0))

    # ── suptitle ──────────────────────────────────────────────────────────────
    fig.text(0.5, (y0 + row0_h + LBL_H + TITLE_H / 2) / fig_h_px,
             "FucTIII, regioselectivity from inferred negatives",
             ha="center", va="center",
             fontsize=12, fontweight="bold", color="#222")

    plt.savefig(out_path, format="svg", bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  wrote {out_path}")


# ── figure 2: KEGG screen ─────────────────────────────────────────────────────

# ── figure: regioselectivity predictions (energetic model) ───────────────────

def write_regioselectivity_svg(out_path: Path) -> None:
    """Train the energetic model on the two natural substrates and, for every held-out
    compound with a known reactive site, draw the molecule with the model's top-predicted
    hydroxyl marked: green if it matches the site the enzyme uses, red if not (with the
    true site in blue)."""
    import io
    import matplotlib.pyplot as plt
    from PIL import Image
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from experiments.model_selection.datasets.fuctiii_regioselectivity import (
        build_energetic, build_test_energetic,
    )
    from esorex.energetic_specificity import EnergeticSpecificityModel

    mols, cores, energies, _ = build_energetic()
    model = EnergeticSpecificityModel()
    model.train(mols, cores, energies=energies)

    panels = []
    for t in build_test_energetic():
        if t["known_site"] is None:
            continue
        o_by_map = {sm: o for o, sm in t["sites"]}
        pred_map = min(((model.predict(t["mol"], {o}).energy, sm) for o, sm in t["sites"]))[1]
        panels.append((t["name"], t["mol"], o_by_map[pred_map],
                       o_by_map[t["known_site"]], pred_map == t["known_site"]))

    n_ok = sum(1 for p in panels if p[4])
    ncol = 5
    nrow = (len(panels) + ncol - 1) // ncol
    fig = plt.figure(figsize=(2.35 * ncol, 2.7 * nrow))
    fig.subplots_adjust(top=0.88, hspace=0.32, wspace=0.04)
    for k, (name, mol, pred_o, known_o, correct) in enumerate(panels):
        m = Chem.RemoveHs(mol)
        AllChem.Compute2DCoords(m)
        hi = [pred_o]
        colors = {pred_o: GREEN_RGB if correct else RED_RGB}
        if not correct:
            hi.append(known_o)
            colors[known_o] = BLUE_RGB
        d = rdMolDraw2D.MolDraw2DCairo(360, 300)
        opt = d.drawOptions()
        opt.highlightRadius = 0.42
        opt.clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=hi, highlightAtomColors=colors)
        d.FinishDrawing()
        img = Image.open(io.BytesIO(d.GetDrawingText()))
        ax = fig.add_subplot(nrow, ncol, k + 1)
        ax.imshow(img); ax.axis("off")
        short = name.split(" ", 1)[1].split("-6azido")[0] if " " in name else name
        ax.set_title(("✓ " if correct else "✗ ") + short, fontsize=9.5,
                     color=GREEN if correct else RED, fontweight="bold")
    fig.suptitle(f"FucTIII regioselectivity: the predicted site is correct on "
                 f"{n_ok} of {len(panels)} held-out compounds", fontsize=13,
                 fontweight="bold", y=0.98)
    fig.text(0.5, 0.925, "green = predicted site (correct)   ·   red = predicted site "
             "(wrong)   ·   blue = the site the enzyme uses", ha="center",
             fontsize=9, color="#555")
    fig.savefig(out_path, facecolor="white")
    plt.close()
    print(f"  wrote {out_path} ({n_ok}/{len(panels)} correct)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading training data...")
    training = _load_training()

    print("Writing sites figure (reaction context)...")
    write_sites_figure_svg(training, OUT / "sites_figure.svg")

    print("Writing regioselectivity predictions figure...")
    write_regioselectivity_svg(OUT / "predictions.svg")

    print(f"\nDone. Assets: {OUT}")


if __name__ == "__main__":
    main()
