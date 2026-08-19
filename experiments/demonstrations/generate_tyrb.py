"""Generate the TyrB energetic-model demonstration and its report.

Outputs (assets/demonstrations/tyrb/):
  operator_levels.svg: Phe annotated at EVODEX levels A-E (mechanistic context)
  ensemble.svg:        the learned ensemble painted onto real substrates (per-atom weights)
  predictions.svg:     predicted vs measured rate for the held-out unnatural analogs
  augmentation.svg:    prediction error before vs after adding the outlier to training
plus the report at experiments/transaminases/tyrb_energetic_report.html.

Usage:
  cd /path/to/ESOREX && source venv/bin/activate
  python experiments/demonstrations/generate_tyrb.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "assets" / "demonstrations" / "tyrb"

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

# ── constants ──────────────────────────────────────────────────────────────────
CURATED    = ROOT / "data/transaminases/ONUFFER_curated.csv"
ENZYME_COL = "eTATase_kf_over_KD_M-1_s-1"
DUPLICATE  = {"Arginine (mu = 1.0)"}
NATURALS   = {
    "Aspartate", "Glutamate", "Phenylalanine", "Tryptophan", "Tyrosine",
    "Alanine", "Leucine", "Valine", "Arginine (mu = 0.2)",
}

# EVODEX level SMIRKS from the TyrB mechanistic tree (N is the mapped atom)
EVODEX_LEVELS = {
    "A": "[*:1]-*>>*-[*:1]",
    "B": "[#7:1]-[#6@@]>>[#1]-[#7:1]",
    "C": "[#1]-[#6@@](-[#7:1](-[#1])-[#1])(-[#6])-[#6]>>[#1]-[#7:1](-[#1])-[#1]",
    "D": "[#1]-[#6@@](-[#7:1](-[#1])-[#1])(-[#6]=[#8])-[#6]>>[#1]-[#7:1](-[#1])-[#1]",
    "E": "[#1]-[#6@@](-[#7:1](-[#1])-[#1])(-[#6](=[#8])-[#8])-[#6]>>[#1]-[#7:1](-[#1])-[#1]",
}

_ELEM = {1:"H",6:"C",7:"N",8:"O",15:"P",16:"S",9:"F",17:"Cl",35:"Br",53:"I"}
def _elem(n): return _ELEM.get(n, f"[{n}]")

def _x(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


# ── data loading ───────────────────────────────────────────────────────────────
def _parse_float(s):
    s = s.strip()
    if not s or s.lower() in ("nd","n/a","na","-"): return None
    try: return float(s)
    except ValueError: return None

# ── EVODEX level filter check ──────────────────────────────────────────────────
# ── SVG: EVODEX level operator visualization ───────────────────────────────────
def write_operator_levels_svg(out_path):
    """5-panel grid showing Phe with each EVODEX level's atoms highlighted."""
    PHE_SMILES = "N[C@@H](Cc1ccccc1)C(=O)O"

    # Determine which atoms are matched at each level
    mol_full = Chem.AddHs(Chem.MolFromSmiles(PHE_SMILES))
    mol_draw = Chem.MolFromSmiles(PHE_SMILES)
    AllChem.Compute2DCoords(mol_draw)

    level_matched = {}
    for lname, smirks in EVODEX_LEVELS.items():
        pat = Chem.MolFromSmarts(smirks.split(">>")[0])
        if pat:
            match = mol_full.GetSubstructMatch(pat)
            # map back to heavy-atom indices in mol_draw
            # mol_full has Hs appended; heavy atoms are indices 0..N-1
            n_heavy = mol_draw.GetNumAtoms()
            heavy_match = [idx for idx in match if idx < n_heavy]
            level_matched[lname] = heavy_match
        else:
            level_matched[lname] = []

    panel_w, panel_h = 230, 170
    label_h = 30
    gap = 12
    margin = 16
    total_w = margin * 2 + 5 * panel_w + 4 * gap
    total_h = margin * 2 + label_h + panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">',
        '<style>',
        '  .bg { fill: #fafafa; }',
        '  .panel { fill: #ffffff; stroke: #e0e0e0; stroke-width: 1; rx: 8; ry: 8; }',
        '  .level-label { font-family: Helvetica, Arial, sans-serif; font-size: 13px; '
        '                 font-weight: 800; fill: #1a1a1a; text-anchor: middle; }',
        '  .level-desc  { font-family: Helvetica, Arial, sans-serif; font-size: 10px; '
        '                 fill: #666; text-anchor: middle; }',
        '</style>',
        f'<rect class="bg" width="{total_w}" height="{total_h}"/>',
    ]

    level_descriptions = {
        "A": "bond topology only",
        "B": "atom types at reactive center",
        "C": "sigma shell",
        "D": "pi shell (C=O context)",
        "E": "full carboxylate",
    }

    # Nice highlight colors for each level
    level_colors_hex = {
        "A": "#aaaaaa",
        "B": "#4a90d9",
        "C": "#7bc67e",
        "D": "#f0a830",
        "E": "#e05a5a",
    }

    for i, (lname, smirks) in enumerate(EVODEX_LEVELS.items()):
        x0 = margin + i * (panel_w + gap)
        y0 = margin

        parts.append(f'<rect class="panel" x="{x0}" y="{y0}" width="{panel_w}" height="{total_h - 2*margin}" rx="8" ry="8"/>')

        # Level label
        cx = x0 + panel_w // 2
        parts.append(f'<text class="level-label" x="{cx}" y="{y0 + 20}"'
                     f' fill="{level_colors_hex[lname]}">EVODEX-{lname}</text>')
        parts.append(f'<text class="level-desc" x="{cx}" y="{y0 + 33}">'
                     f'{level_descriptions[lname]}</text>')

        # Draw molecule with highlights
        matched = level_matched.get(lname, [])
        mol_copy = Chem.RWMol(mol_draw)

        drawer = rdMolDraw2D.MolDraw2DSVG(panel_w - 8, panel_h - 4)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = False
        opts.padding = 0.12

        # Highlight atoms matching this level
        ha = list(set(matched))
        hb = []
        for b in mol_draw.GetBonds():
            if b.GetBeginAtomIdx() in ha and b.GetEndAtomIdx() in ha:
                hb.append(b.GetIdx())

        # Build color dicts
        from rdkit.Chem.Draw import rdMolDraw2D as rdd
        color = rdd.DrawingOptions.elemDict.get if hasattr(rdd, 'DrawingOptions') else None
        # Convert hex color to RGB tuple
        h = level_colors_hex[lname].lstrip("#")
        rgb = tuple(int(h[i:i+2],16)/255 for i in (0,2,4))
        atom_colors = {idx: rgb for idx in ha}
        bond_colors = {idx: rgb for idx in hb}

        drawer.DrawMolecule(mol_draw,
                            highlightAtoms=ha, highlightBonds=hb,
                            highlightAtomColors=atom_colors,
                            highlightBondColors=bond_colors)
        drawer.FinishDrawing()
        mol_svg = drawer.GetDrawingText()
        # Strip XML header and outer SVG tag, embed as group
        mol_svg = re.sub(r"<\?xml[^>]*\?>", "", mol_svg)
        mol_svg = re.sub(r'<svg[^>]*>', "", mol_svg, count=1)
        mol_svg = mol_svg.replace("</svg>", "")
        parts.append(f'<g transform="translate({x0+4},{y0+label_h})">{mol_svg}</g>')

    parts.append("</svg>")
    svg = "\n".join(parts)
    out_path.write_text(svg)
    print(f"  wrote {out_path}")


# ── TreeEdit reactive-site helper + edit-matrix model figure ──────────────────
def _load_substrate_smiles():
    """Load substrate_smiles (clean molecule SMILES) from the curated CSV."""
    smi_map = {}
    with CURATED.open() as f:
        for row in csv.DictReader(f):
            name = row["substrate"].strip()
            smi  = row.get("substrate_smiles","").strip()
            if smi:
                smi_map[name] = smi
    return smi_map


# ── SVG: scatter plot ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
# Energetic model (current ensemble), the TyrB story
# ═══════════════════════════════════════════════════════════════════════════════
# The sections above draw the mechanistic context (the EVODEX operator TyrB shares
# across substrates).  The specificity model itself is now the energetic ensemble:
# rates -> E = -RT ln k, substrates -> hydrogenated-reference physics features ->
# identifiability collapse -> a free-energy constraint system.  Everything below
# trains that model on the natural amino acids, reads out what it has learned about
# the enzyme, and tests it on the unnatural analogs.

_AA_CORE = Chem.MolFromSmarts("[NX3][CX4H][CX3](=O)[OX2]")


def _substrate_core(mol):
    return set(mol.GetSubstructMatch(_AA_CORE))


def _energetic_dataset():
    """Return (naturals, analogs) as [(name, mol, rate)].

    Naturals are the proteinogenic amino acids TyrB evolved with (the experiment's
    NATURALS set); everything else with a measured rate is a held-out unnatural test
    substrate, INCLUDING poor and surprising ones.  The test set is not curated for
    agreement; poor and surprising substrates stay in it, and any outlier is a result
    of the experiment, not a thing to hide."""
    smi = _load_substrate_smiles()
    nat, ana = [], []
    with CURATED.open() as f:
        for row in csv.DictReader(f):
            name = row["substrate"].strip()
            if name in DUPLICATE:
                continue
            rate = _parse_float(row.get(ENZYME_COL, ""))
            mol = Chem.MolFromSmiles(smi.get(name, ""))
            if rate is None or mol is None or not _substrate_core(mol):
                continue
            (nat if name in NATURALS else ana).append((name, mol, rate))
    return nat, ana


def train_energetic_model():
    from esorex.energetic_specificity import EnergeticSpecificityModel
    nat, ana = _energetic_dataset()
    model = EnergeticSpecificityModel()
    model.train([m for _, m, _ in nat], [_substrate_core(m) for _, m, _ in nat],
                rates=[r for _, _, r in nat])
    return model, nat, ana


# substrates chosen to span the learned active-site logic, on real training structures
_PANEL = [
    ("Tryptophan",          "bulk and aromaticity win",  "every ring carbon lowers the barrier"),
    ("Tyrosine",            "where, not only what",      "the same ring, but the para-O is penalized"),
    ("Arginine (mu = 0.2)", "distance flips the sign",   "chain carbons favor, the distal guanidinium penalizes"),
    ("Aspartate",           "polar is not enough",       "the carboxylate reads mildly unfavorable"),
]
_PANEL_SHORT = {"Arginine (mu = 0.2)": "Arginine"}


def _paint_contrib(model, mol, core):
    """Per-atom shading values for the painted figure: each carbon from
    model.atom_contributions, each passenger heteroatom taking the mean of the carbons it
    hangs on.  Core (reaction-center) atoms are left out (drawn gray by the renderer)."""
    core = set(core)
    carbon = model.atom_contributions(mol, core)
    contrib = dict(carbon)
    for a in mol.GetAtoms():
        j = a.GetIdx()
        if j in core or a.GetAtomicNum() in (1, 6):
            continue
        vals = [carbon[nb.GetIdx()] for nb in a.GetNeighbors() if nb.GetIdx() in carbon]
        if vals:
            contrib[j] = sum(vals) / len(vals)
    return contrib


def write_ensemble_svg(model, nat, out_path):
    """Paint what the model learned onto its real substrates: each atom is shaded by the
    free-energy contribution the model attributes to it (green lowers the activation energy,
    red raises it).  Geometry is the actual 2D structure; the shading is computed from the
    constraint system's weights via model.atom_contributions, not drawn by hand."""
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from PIL import Image
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    cmap = LinearSegmentedColormap.from_list(
        "esorex", ["#1f7a4d", "#7dc39a", "#f3f4f6", "#e3a486", "#b8492a"])
    vmax = 1.8
    norm = Normalize(-vmax, vmax)
    by_name = {n: (m, r) for n, m, r in nat}

    def render_mol(mol, contrib, core, w_px=620, h_px=400):
        AllChem.Compute2DCoords(mol)
        hi_atoms, hi_colors = [], {}
        for i in range(mol.GetNumAtoms()):
            if i in core:
                hi_atoms.append(i); hi_colors[i] = (0.93, 0.94, 0.95)   # fixed backbone
            elif i in contrib:
                hi_atoms.append(i)
                r, gc, b, _ = cmap(norm(contrib[i]))
                hi_colors[i] = (r, gc, b)
        d = rdMolDraw2D.MolDraw2DCairo(w_px, h_px)
        o = d.drawOptions()
        o.highlightRadius = 0.40
        o.highlightBondWidthMultiplier = 8
        o.padding = 0.10
        o.clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hi_atoms,
                                           highlightAtomColors=hi_colors, highlightBonds=[])
        d.FinishDrawing()
        return Image.open(io.BytesIO(d.GetDrawingText()))

    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.03, top=0.85, bottom=0.14,
                          left=0.03, right=0.97)
    for k, (name, headline, sub) in enumerate(_PANEL):
        mol, rate = by_name[name]
        core = _substrate_core(mol)
        contrib = _paint_contrib(model, mol, core)
        img = render_mol(mol, contrib, core)
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.imshow(img); ax.axis("off")
        disp = _PANEL_SHORT.get(name, name)
        ax.set_title(f"{disp}   ·   measured k = {rate:.2g}", fontsize=11,
                     fontweight="bold", pad=6)
        ax.text(0.5, -0.02, headline, transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, color="#1f2328")
        ax.text(0.5, -0.09, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.8, color="#6b7280", style="italic")

    fig.suptitle("What TyrB learned, painted on its substrates", fontsize=15,
                 fontweight="bold", y=0.965)
    fig.text(0.5, 0.905, "each atom is shaded by the free-energy contribution the model "
             "attributes to it  ·  the α-amino-acid backbone (gray) is the fixed "
             "reaction center", ha="center", fontsize=9.5, color="#59636e")
    cax = fig.add_axes([0.30, 0.055, 0.40, 0.022])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([-vmax, 0, vmax])
    cb.set_ticklabels(["lowers ΔE‡  (faster)", "0", "raises ΔE‡  (slower)"])
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)
    fig.savefig(out_path, facecolor="white")
    plt.close()
    print(f"  wrote {out_path}")


def write_energetic_predictions_svg(model, nat, ana, out_path):
    """Measured vs predicted rate for the held-out analogs, marking determined
    (fully constrained) predictions and coloring by side-chain class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy import stats
    import numpy as np

    SHORT = {"4-Methylphenylalanine": "4Me-Phe", "4-Methoxyphenylalanine": "4MeO-Phe",
             "4-Aminophenylalanine": "4NH2-Phe", "4-Nitrophenylalanine": "4NO2-Phe",
             "Metatyrosine": "m-Tyr", "Orthotyrosine": "o-Tyr", "Phosphotyrosine": "PO4-Tyr",
             "2-Aminooctanoate": "C8-AA", "Norleucine": "Nle", "Norvaline": "Nva",
             "2-Aminobutyrate": "Abu", "CHA (cyclohexylalanine)": "CHA", "DOPA": "DOPA"}

    def klass(name):
        n = name.lower()
        if any(x in n for x in ("phe", "tyr", "trp", "dopa")): return "#4C72B0", "aromatic"
        if "cha" in n or "cyclo" in n: return "#55A868", "cyclic"
        return "#DD8452", "alkyl"

    pts = []
    for name, mol, rate in ana:
        p = model.predict(mol, _substrate_core(mol))
        pts.append((name, rate, p.rate, p.determined))
    meas = np.array([p[1] for p in pts]); pred = np.array([p[2] for p in pts])
    rho, pval = stats.spearmanr(meas, pred)
    err = np.abs(np.log10(np.maximum(pred, 1e-30)) - np.log10(np.maximum(meas, 1e-30)))

    fig, ax = plt.subplots(figsize=(6.9, 6.9))
    lo, hi = min(meas.min(), pred.min()) * 0.3, max(meas.max(), pred.max()) * 3
    ax.plot([lo, hi], [lo, hi], "--", color="#999", lw=1, zorder=1)
    for fct in (0.1, 10):
        ax.plot([lo, hi], [lo * fct, hi * fct], ":", color="#ccc", lw=0.8, zorder=1)
    for name, m, pr, det in pts:
        col, _ = klass(name)
        ax.scatter(m, pr, s=90, zorder=3, edgecolors=col, linewidths=1.7,
                   facecolors=col if det else "white")
        ax.annotate(SHORT.get(name, name[:7]), (m, pr), textcoords="offset points",
                    xytext=(6, 3), fontsize=7.5, color="#444")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("measured  kf/KD  (M⁻¹s⁻¹)")
    ax.set_ylabel("predicted  kf/KD  (M⁻¹s⁻¹)")
    ax.set_title(f"TyrB, train {len(nat)} naturals, predict {len(pts)} analogs\n"
                 f"Spearman ρ = {rho:.3f}   median |log₁₀ err| = {np.median(err):.2f}",
                 fontsize=10)
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#4C72B0', markersize=9, label='aromatic'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor='#DD8452', markersize=9, label='alkyl'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor='#55A868', markersize=9, label='cyclic'),
           Line2D([0], [0], marker='o', color='#333', markerfacecolor='#333', markersize=9, label='determined'),
           Line2D([0], [0], marker='o', color='#333', markerfacecolor='white', markersize=9, label='undetermined')]
    ax.legend(handles=leg, fontsize=8, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close()
    print(f"  wrote {out_path}")
    return dict(rho=round(float(rho), 3), pval=round(float(pval), 4),
                median_err=round(float(np.median(err)), 2),
                mean_err=round(float(err.mean()), 2),
                determined=int(sum(p[3] for p in pts)),
                points=[dict(name=n, meas=m, pred=pr, determined=d) for n, m, pr, d in pts])


def _logerr(pred, meas):
    import numpy as np
    return float(np.log10(max(pred, 1e-30)) - np.log10(max(meas, 1e-30)))


def _worst_outlier(model, ana):
    """The single held-out substrate the model misreads most (largest |log rate error|)."""
    worst, we = None, -1.0
    for name, mol, rate in ana:
        e = abs(_logerr(model.predict(mol, _substrate_core(mol)).rate, rate))
        if e > we:
            we, worst = e, name
    return worst


def run_augmentation(nat, ana, outlier_name):
    """Add the outlier back into training, retrain, and measure the effect on every
    other test prediction: does inconsistent new data erode the model or refine it?"""
    outlier = next(t for t in ana if outlier_name in t[0])
    base = _fit([(m, _substrate_core(m), r) for _, m, r in nat])
    aug = _fit([(m, _substrate_core(m), r) for _, m, r in nat + [outlier]])
    before, after = {}, {}
    for name, mol, rate in ana:
        before[name] = _logerr(base.predict(mol, _substrate_core(mol)).rate, rate)
        after[name] = _logerr(aug.predict(mol, _substrate_core(mol)).rate, rate)
    return dict(outlier=outlier[0], base=base, aug=aug, before=before, after=after,
                aug_exact=aug.info["exact"], aug_coop=aug.info["cooperative_terms_added"])


def _fit(triples):
    from esorex.energetic_specificity import EnergeticSpecificityModel
    m = EnergeticSpecificityModel()
    m.train([t[0] for t in triples], [t[1] for t in triples], rates=[t[2] for t in triples])
    return m


def write_augmentation_svg(aug_result, out_path):
    """|error| before vs after adding the outlier to training.  Points on the diagonal
    are unchanged; the outlier drops to zero (now reproduced exactly); the question is
    whether the others rise (erosion) or stay put (graceful integration)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(aug_result["before"])
    bef = np.array([abs(aug_result["before"][n]) for n in names])
    aft = np.array([abs(aug_result["after"][n]) for n in names])
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    hi = max(bef.max(), aft.max()) * 1.28
    ax.plot([0, hi], [0, hi], "--", color="#999", lw=1)
    ax.fill_between([0, hi], [0, hi], [hi, hi], color="#c0603a", alpha=0.05)   # erosion zone
    for n, b, a in zip(names, bef, aft):
        is_out = aug_result["outlier"] in n
        ax.scatter(b, a, s=90 if is_out else 55,
                   color="#c0603a" if is_out else "#2e5a8f",
                   zorder=3, edgecolors="white", linewidths=0.8)
        if is_out or abs(b - a) > 0.3:
            ax.annotate(n.split()[0][:10], (b, a), textcoords="offset points",
                        xytext=(6, 3), fontsize=7.5, color="#444")
    ax.set_xlim(-0.1, hi); ax.set_ylim(-0.1, hi)
    ax.set_xlabel("|log₁₀ error|  before  (trained on naturals only)")
    ax.set_ylabel("|log₁₀ error|  after  (+ outlier in training)")
    ax.set_title("Does adding the outlier erode the model?\n"
                 "points above the diagonal = worse;  on it = unchanged", fontsize=10)
    ax.text(hi * 0.55, hi * 0.9, "erosion zone", color="#c0603a", fontsize=8.5, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close()
    print(f"  wrote {out_path}")


_REPORT_TEMPLATE = r"""<meta charset="utf-8">
<title>TyrB, reading an aminotransferase as free energy</title>
<style>
:root{--paper:#f6f8fb;--card:#fff;--ink:#161d29;--muted:#5b6779;--line:#e2e7ef;
 --accent:#2e5a8f;--accent-soft:#e8f0f9;--good:#2f8a5b;--warn:#c0603a;
 --serif:Georgia,'Iowan Old Style','Times New Roman',serif;
 --sans:-apple-system,BlinkMacSystemFont,system-ui,'Segoe UI',sans-serif;
 --mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;}
@media(prefers-color-scheme:dark){:root{--paper:#0f141b;--card:#161d27;--ink:#e6ebf3;
 --muted:#94a1b5;--line:#26303d;--accent:#7aa6d8;--accent-soft:#1b2735;}}
:root[data-theme="dark"]{--paper:#0f141b;--card:#161d27;--ink:#e6ebf3;--muted:#94a1b5;
 --line:#26303d;--accent:#7aa6d8;--accent-soft:#1b2735;}
:root[data-theme="light"]{--paper:#f6f8fb;--card:#fff;--ink:#161d29;--muted:#5b6779;
 --line:#e2e7ef;--accent:#2e5a8f;--accent-soft:#e8f0f9;}
*{box-sizing:border-box}.wrap{margin:0}
.wrap{background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.62;
 padding:clamp(20px,4vw,64px) clamp(16px,4vw,40px)}
.doc{max-width:840px;margin:0 auto}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--accent);margin:0 0 12px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(28px,5vw,44px);line-height:1.1;
 letter-spacing:-.01em;text-wrap:balance;margin:0 0 16px}
.lede{font-size:clamp(17px,2.4vw,20px);color:var(--muted);max-width:62ch;margin:0 0 8px}
h2{font-family:var(--serif);font-weight:600;font-size:24px;margin:54px 0 6px;padding-top:22px;
 border-top:1px solid var(--line)}
h2 .n{font-family:var(--mono);font-size:13px;color:var(--accent);margin-right:12px;font-weight:400}
p{max-width:65ch}
figure{margin:26px 0}figure svg{width:100%;height:auto;max-width:100%}
.card-fig{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
figcaption{font-size:13.5px;color:var(--muted);margin-top:12px;max-width:66ch}
.scoreboard{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
 background:var(--line);border:1px solid var(--line);border-radius:14px;overflow:hidden;margin:28px 0}
.metric{background:var(--card);padding:20px 22px}
.metric .k{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--muted);margin-bottom:8px}
.metric .v{font-family:var(--serif);font-size:32px;font-weight:600;line-height:1;font-variant-numeric:tabular-nums}
.metric .d{font-size:13px;color:var(--muted);margin-top:7px}
.tablewrap{overflow-x:auto;margin:20px 0;border:1px solid var(--line);border-radius:12px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
 font-weight:500;background:var(--card)}
tr:last-child td{border-bottom:none}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
td.num.big{color:var(--warn);font-weight:600}
.pill{font-family:var(--mono);font-size:11px;padding:3px 9px;border-radius:20px}
.pill.det{background:var(--accent-soft);color:var(--accent)}
.pill.und{border:1px solid var(--line);color:var(--muted)}
.note{background:var(--card);border-left:3px solid var(--accent);border-radius:0 10px 10px 0;
 padding:16px 20px;margin:22px 0;font-size:14.5px}
code{font-family:var(--mono);font-size:.9em;background:var(--accent-soft);color:var(--accent);
 padding:1px 5px;border-radius:4px}
.foot{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);
 font-family:var(--mono);font-size:12px;color:var(--muted)}
</style>
<div class="wrap"><div class="doc">
 <p class="eyebrow">ESOREX · transaminase specificity</p>
 <h1>Reading TyrB as free energy</h1>
 <p class="lede">TyrB is an engineered tyrosine aminotransferase. Given only its rates on
  the natural amino acids, can a free-energy model recover what its active site cares
  about, and predict the unnatural substrates it has never seen?</p>

 <h2><span class="n">01</span>The enzyme and its reaction</h2>
 <p>Every substrate TyrB accepts undergoes the same transamination at the α-carbon; the
  substrates differ only in the side chain the active site must accommodate. That shared
  mechanistic core is the EVODEX reaction operator, the fixed chemistry against which
  side-chain differences become the specificity signal.</p>
 <figure class="card-fig">%OPERATOR_SVG%
  <figcaption>The reaction operator at successively wider electronic scopes (A→E), drawn
   on phenylalanine. The core is common to all substrates; specificity lives outside it.</figcaption>
 </figure>

 <h2><span class="n">02</span>How the model represents a substrate</h2>
 <p>Each substrate's rate becomes an activation energy, <code>E = &minus;RT ln k</code>.
  Its structure is reduced to a fully hydrogenated carbon skeleton, and the erased
  chemistry returns as <em>physical</em> deviations from that reference, a nucleus's
  proton count, its oxidation state and formal charge, its available lone pairs, and the
  π systems that define each subdomain. No functional groups are named; a hydroxyl or a
  nitro is an emergent pattern of these quantities. Features are addressed at three
  positional scales (atom, subdomain, passenger), and an identifiability collapse keeps
  only the distinctions the ten measurements can actually support.</p>
 <div class="note">The model is a hard constraint, not a fit: it reproduces all %NNAT%
  training rates exactly. Of %RAW% candidate physical features it retains %COLLAPSED%
  after collapse, leaving a %NULL%-dimensional space of energetic decompositions the data
  do not resolve, which the model keeps rather than resolving arbitrarily.</div>

 <h2><span class="n">03</span>What it learned about the active site</h2>
 <p>Because the energy is additive, <code>E_S = E_0 + &Sigma; w_i x_i</code>, every learned weight
  can be attributed back to the exact atoms its feature addresses. Painted onto a substrate, each
  atom carries the free-energy contribution the model assigns to it, and summed over the atoms it
  reproduces that substrate's prediction exactly. The result is a portrait of what TyrB's active
  site rewards and penalizes, recovered from nine rates with no chemical hypothesis supplied in
  advance, and read directly off real structures.</p>
 <figure class="card-fig">%ENSEMBLE_SVG%
  <figcaption>Four training substrates, each atom shaded by the free-energy contribution the model
   attributes to it (green lowers the activation energy, red raises it); the &alpha;-amino-acid
   backbone in gray is the fixed reaction center. Geometry is the real 2D structure; the shading is
   computed from the constraint system's weights, not drawn by hand.</figcaption>
 </figure>
 <p>The dominant reward is bulk and aromaticity: every carbon of an aromatic ring lowers the
  activation energy, which is why Trp, Tyr, and Phe are the best substrates. The most telling
  distinction is <em>positional</em>. On tyrosine the ring is green while its para-oxygen is the
  lone red atom: the same ring is rewarded, that one position penalized. Distance matters too,
  arginine's near-core chain carbons paint green while its distal guanidinium paints strongly red,
  the same nitrogen chemistry rewarded near the reaction center and penalized far from it. And
  polarity by itself is not enough, aspartate's carboxylate reads mildly unfavorable rather than
  rewarded. Where a group sits, not only what it is, sets its energetic effect, and the model read
  that off the rates.</p>
 <div class="note">The per-atom split is honest but not unique. The weights come from a min-norm
  solution of an underdetermined system, so each atom's share is one representative of the solution
  space the model keeps; the summed prediction is what the data pin down. Only the combinations the
  data determine are claimed.</div>

 <h2><span class="n">04</span>Predicting the unnatural substrates</h2>
 <p>Trained on the naturals alone, the model predicts each of the %NANA% held-out
  unnatural substrates, and ranks them well (Spearman ρ = %RHO%). Most land within a
  few-fold. The test set is not curated for agreement, and one substrate,
  <b>%OUTLIER%</b>, comes out badly wrong (%OUT_ERR% in log rate): it is the largest
  single miss, and setting it aside the rank correlation is ρ = %RHO_WO%. The robust
  summary is a median error of %MEDIAN% (≈ %FOLD%× in rate).</p>
 <div class="scoreboard">
  <div class="metric"><div class="k">Median |log₁₀ err|</div><div class="v">%MEDIAN%</div>
   <div class="d">≈ %FOLD%× in rate</div></div>
  <div class="metric"><div class="k">Spearman ρ</div><div class="v">%RHO%</div>
   <div class="d">%RHO_WO% without the one outlier</div></div>
  <div class="metric"><div class="k">Training</div><div class="v">exact</div>
   <div class="d">all %NNAT% naturals reproduced</div></div>
  <div class="metric"><div class="k">Determined</div><div class="v">%DETERMINED%/%NANA%</div>
   <div class="d">fully constrained</div></div>
 </div>
 <figure class="card-fig">%PREDICTIONS_SVG%
  <figcaption>Filled points are <b>determined</b>; open points are extrapolations the
   model flags as unresolved but still estimates. %OUTLIER% is the open point far %SIDE%
   the diagonal, flagged, and wrong.</figcaption>
 </figure>
 <div class="tablewrap"><table>
  <thead><tr><th>Substrate</th><th>Measured</th><th>Predicted</th><th>log₁₀ err</th><th>Status</th></tr></thead>
  <tbody>%TABLE_ROWS%</tbody>
 </table></div>

 <h2><span class="n">05</span>Why it gets %OUTLIER% wrong</h2>
 <p>%OUTLIER% is the model's one real miss, predicted %OUT_ERR% in log rate (too %TOO%).
  Its side chain is large but has no aromatic ring. In the training naturals size and
  aromaticity always travel together, every large natural (Phe, Tyr, Trp) is aromatic, so
  the model cannot separate "large" from "aromatic": it only ever saw the bulk reward in
  the company of a ring.</p>
 <p>Faced with a side chain that is large but not aromatic, the additive model withholds
  the bulk reward it never learned to grant on its own, and predicts too %TOO%. It flags
  the prediction unresolved, because the combination is unprecedented, but the estimate
  still lands orders of magnitude off. This is the honest limit of an additive model
  trained where size, π, and aromaticity all happen to co-vary, the training set holds no
  large non-aromatic acceptor to teach the two apart.</p>

 <h2><span class="n">06</span>Feeding the outlier back in</h2>
 <p>So we did the second experiment: put %OUTLIER% and its true rate into the training set
  and retrain. Does one inconsistent, surprising measurement erode what the model already
  knew, or does it simply make it more complete?</p>
 <figure class="card-fig">%AUGMENTATION_SVG%
  <figcaption>Absolute error on every test substrate, before vs after adding %OUTLIER%
   to training. On the diagonal = unchanged; above it = eroded. %OUTLIER% drops to zero
   (now reproduced exactly); the rest stay on the diagonal.</figcaption>
 </figure>
 <p>It integrates gracefully. %OUTLIER% is reproduced exactly (with %AUG_COOP% cooperative
  terms, the constraint system simply absorbs it), and the other predictions barely move: median error %AUG_MED_B% → %AUG_MED_A%, with %AUG_IMPROVED%
  improved, %AUG_DEGRADED% worse, and %AUG_SAME% unchanged. The surprising point does not
  overwrite what the naturals established; each existing substrate is still anchored by
  its own measurement. New data, even data that contradicts the model's prior
  extrapolation, makes it more nuanced without eroding it. That is the behavior a hard,
  constraint-based model is built to have: it never trades away a known fact to fit a new
  one, it widens the represented world to hold both.</p>

 <div class="foot">TyrB (engineered tyrosine aminotransferase; Onuffer &amp; Kirsch, Protein Sci. 1995)
  · eTATase kf/KD · energetic ESOREX · train %NNAT% naturals → predict %NANA% analogs</div>
</div></div>
"""


def _inline_svg(path):
    """Read an .svg file and return its <svg>… markup for inline embedding."""
    s = Path(path).read_text()
    i = s.find("<svg")
    return s[i:] if i >= 0 else s


def write_report(model, nat, ana, stats, aug, out_path):
    """Assemble the TyrB story: the enzyme, how the model represents it, what it
    learned, how well it predicts, why it fails on one substrate, and what happens
    when that substrate is fed back in."""
    import numpy as np
    info = model.info

    # honest metric read: the outlier dominates ρ; report it and its effect openly.
    outlier = aug["outlier"]
    others = [p for p in stats["points"] if outlier not in p["name"]]
    from scipy import stats as _st
    rho_wo, _ = _st.spearmanr([p["meas"] for p in others], [p["pred"] for p in others])
    out_err = next(_logerr(p["pred"], p["meas"]) for p in stats["points"] if outlier in p["name"])
    too = "slow" if out_err < 0 else "fast"
    side = "below" if out_err < 0 else "above"

    # augmentation summary on the OTHER test substrates
    ok = [n for n in aug["before"] if outlier not in n]
    b = np.array([abs(aug["before"][n]) for n in ok])
    a = np.array([abs(aug["after"][n]) for n in ok])
    improved = int((a < b - 0.1).sum()); degraded = int((a > b + 0.1).sum())
    same = len(ok) - improved - degraded
    SHORT = {"CHA (cyclohexylalanine)": "cyclohexyl-Ala"}
    rows = []
    for p in sorted(stats["points"], key=lambda p: -p["meas"]):
        e = round(_np_log(p["pred"]) - _np_log(p["meas"]), 2)
        badge = ('<span class="pill det">determined</span>' if p["determined"]
                 else '<span class="pill und">extrapolation</span>')
        big = ' class="num big"' if abs(e) >= 1 else ' class="num"'
        rows.append(f"<tr><td>{SHORT.get(p['name'], p['name'])}</td>"
                    f"<td class='num'>{p['meas']:.1e}</td><td class='num'>{p['pred']:.1e}</td>"
                    f"<td{big}>{e:+.2f}</td><td>{badge}</td></tr>")
    repl = {
        "%RHO%": str(stats["rho"]), "%MEDIAN%": str(stats["median_err"]),
        "%NANA%": str(len(ana)), "%NNAT%": str(len(nat)),
        "%DETERMINED%": str(stats["determined"]), "%RAW%": str(info["n_raw_features"]),
        "%FOLD%": str(round(10 ** stats["median_err"], 1)),
        "%COLLAPSED%": str(info["n_collapsed_features"]), "%NULL%": str(info["null_dim"]),
        "%COOP%": str(info["cooperative_terms_added"]),
        "%OPERATOR_SVG%": _inline_svg(OUT / "operator_levels.svg"),
        "%ENSEMBLE_SVG%": _inline_svg(OUT / "ensemble.svg"),
        "%PREDICTIONS_SVG%": _inline_svg(OUT / "predictions.svg"),
        "%AUGMENTATION_SVG%": _inline_svg(OUT / "augmentation.svg"),
        "%TABLE_ROWS%": "".join(rows),
        "%OUTLIER%": outlier, "%OUT_ERR%": f"{out_err:+.1f}",
        "%TOO%": too, "%SIDE%": side,
        "%RHO_WO%": f"{rho_wo:.3f}",
        "%AUG_MED_B%": f"{np.median(b):.2f}", "%AUG_MED_A%": f"{np.median(a):.2f}",
        "%AUG_IMPROVED%": str(improved), "%AUG_DEGRADED%": str(degraded),
        "%AUG_SAME%": str(same), "%AUG_COOP%": str(aug["aug_coop"]),
    }
    html = _REPORT_TEMPLATE
    for k, v in repl.items():
        html = html.replace(k, v)
    Path(out_path).write_text(html)
    print(f"  wrote {out_path}")


def _np_log(x):
    import numpy as np
    return float(np.log10(max(x, 1e-30)))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Writing EVODEX operator levels SVG (mechanistic context)...")
    write_operator_levels_svg(OUT / "operator_levels.svg")

    print("Training energetic model on the natural amino acids...")
    model, nat, ana = train_energetic_model()
    print("  ", {k: model.info[k] for k in
                 ("n_substrates", "n_raw_features", "n_collapsed_features",
                  "rank", "null_dim", "cooperative_terms_added", "exact")})

    print("Painting the learned ensemble onto real substrates...")
    write_ensemble_svg(model, nat, OUT / "ensemble.svg")

    print("Writing energetic predictions SVG...")
    stats = write_energetic_predictions_svg(model, nat, ana, OUT / "predictions.svg")
    print(f"  Spearman ρ = {stats['rho']}, median |log err| = {stats['median_err']}, "
          f"determined {stats['determined']}/{len(ana)}")

    print("Running augmentation experiment (add the outlier back)...")
    aug = run_augmentation(nat, ana, _worst_outlier(model, ana))
    write_augmentation_svg(aug, OUT / "augmentation.svg")
    print(f"  outlier {aug['outlier']}: {aug['before'][aug['outlier']]:+.2f} -> "
          f"{aug['after'][aug['outlier']]:+.2f} (now training); aug exact={aug['aug_exact']}")

    print("Writing TyrB story report...")
    report_path = ROOT / "experiments" / "transaminases" / "tyrb_energetic_report.html"
    write_report(model, nat, ana, stats, aug, report_path)

    print("\nAll assets written to:", OUT)
    print("Report:", report_path)
    return model, nat, ana, stats


if __name__ == "__main__":
    main()
