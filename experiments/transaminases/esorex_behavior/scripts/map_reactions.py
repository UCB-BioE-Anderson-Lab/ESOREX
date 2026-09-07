"""
map_reactions.py — Generate fully atom-mapped reaction SMILES for Onuffer 1995 substrates.

Reads:
  data/transaminases/ONUFFER_curated.csv

Writes:
  experiments/transaminases/esorex_behavior/results/onuffer_mapped_reactions.csv
      Per-substrate full mapping results and prepare_reaction status.

  experiments/transaminases/esorex_behavior/results/onuffer_mapping_report.html
      Human-readable experiment report with prose and tables.

Run from the repository root:
    python experiments/transaminases/esorex_behavior/scripts/map_reactions.py

Chemistry
---------
Each L-alpha-amino acid substrate undergoes transamination:

    amino acid + H2O  →  alpha-keto acid + NH3   (partial reaction, PLP/PMP omitted)

Atom-map convention:
  :1  alpha-N
  :2  alpha-C
  :3  carboxyl-C
  :4  carboxyl =O
  :5  carboxyl -OH
  :6  water O  (becomes the product ketone =O)
  :7–:9  reserved
  :10 beta-position atom (first atom of side_chain_smiles)
  :11+ gamma and beyond

Reactive-center template (SC = side_chain_smiles fragment):
  Reactant: [N:1][C@@H:2]({SC})[C:3](=[O:4])[OH:5].[OH2:6]
  Product:  [N:1].[C:2](=[O:6])({SC})[C:3](=[O:4])[OH:5]

The side_chain_smiles column of ONUFFER_curated.csv supplies {SC} per substrate.
Concatenating gives a reaction SMILES where every heavy atom carries a unique
map number — the precondition for prepare_reaction.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

DATA_IN   = REPO_ROOT / "data" / "transaminases" / "ONUFFER_curated.csv"
RESULTS   = REPO_ROOT / "experiments" / "transaminases" / "esorex_behavior" / "results"
CSV_OUT   = RESULTS / "onuffer_mapped_reactions.csv"
HTML_OUT  = RESULTS / "onuffer_mapping_report.html"

# Reactive-center template.  {SC} is replaced by the side_chain_smiles fragment.
REACTANT_TMPL = "[N:1][C@@H:2]({SC})[C:3](=[O:4])[OH:5].[OH2:6]"
PRODUCT_TMPL  = "[N:1].[C:2](=[O:6])({SC})[C:3](=[O:4])[OH:5]"


def build_reaction_smiles(side_chain_smiles: str) -> str:
    """Substitute side_chain_smiles into the reactive-center template."""
    reactant = REACTANT_TMPL.replace("{SC}", side_chain_smiles)
    product  = PRODUCT_TMPL.replace("{SC}", side_chain_smiles)
    return f"{reactant}>>{product}"


def attempt_prepare_reaction(rxn_smiles: str):
    """
    Try prepare_reaction.  Returns (ok: bool, error_msg: str, rxn_obj or None).
    """
    try:
        from esorex.reaction_preparation import prepare_reaction
        rxn = prepare_reaction(rxn_smiles)
        return True, "", rxn
    except Exception as e:
        return False, str(e), None


def mol_to_svg(smiles: str, width: int = 200, height: int = 150) -> str:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "<span style='color:#c0392b'>parse error</span>"
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index("<svg"):]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    with open(DATA_IN, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    n_ok = 0
    n_fail = 0

    for row in rows:
        name = row["substrate"]
        sc   = row["side_chain_smiles"].strip()

        if not sc:
            results.append({
                "substrate":             name,
                "side_chain_smiles":     sc,
                "reaction_smiles_full":  "",
                "prepare_reaction_ok":   "skipped",
                "error":                 "no side_chain_smiles",
            })
            n_fail += 1
            continue

        rxn_smiles = build_reaction_smiles(sc)
        ok, err, _ = attempt_prepare_reaction(rxn_smiles)

        results.append({
            "substrate":             name,
            "side_chain_smiles":     sc,
            "reaction_smiles_full":  rxn_smiles,
            "prepare_reaction_ok":   "ok" if ok else "fail",
            "error":                 err,
        })
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    # ── Write CSV ──────────────────────────────────────────────────────────────
    csv_fields = [
        "substrate", "side_chain_smiles", "reaction_smiles_full",
        "prepare_reaction_ok", "error",
    ]
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows → {CSV_OUT}")

    # ── Build HTML report ──────────────────────────────────────────────────────
    # Index curated rows by substrate name for kinetic data
    curated = {r["substrate"]: r for r in rows}

    # Results table rows
    table_rows_html = []
    for i, res in enumerate(results, 1):
        name = res["substrate"]
        sc   = res["side_chain_smiles"]
        ok   = res["prepare_reaction_ok"]
        err  = res["error"]
        rxn  = res["reaction_smiles_full"]

        if ok == "ok":
            status_cell = "<td class='ok'>ok</td>"
        elif ok == "skipped":
            status_cell = "<td class='warn'>skipped</td>"
        else:
            status_cell = "<td class='fail'>fail</td>"

        err_cell = f"<td class='err'>{err}</td>" if err else "<td></td>"

        cr = curated.get(name, {})
        svg = mol_to_svg(cr.get("substrate_smiles", "")) if cr.get("substrate_smiles") else ""

        table_rows_html.append(f"""
  <tr>
    <td class="num">{i}</td>
    <td class="name">{name}</td>
    <td class="sc"><code>{sc}</code></td>
    <td class="struct">{svg}</td>
    {status_cell}
    {err_cell}
  </tr>""")

    # Summary prose
    n_total = len(results)
    n_skipped = sum(1 for r in results if r["prepare_reaction_ok"] == "skipped")

    summary_prose = f"""
<p>
  This experiment constructs a fully atom-mapped reaction SMILES for each of the
  {n_total} L-alpha-amino acid substrates in the Onuffer 1995 dataset and verifies
  that ESOREX's <code>prepare_reaction</code> can parse them.  Passing this check is
  the prerequisite for downstream model building with <code>build_ensemble</code>.
</p>
<p>
  The mapping uses a fixed <em>reactive-center template</em> encoding the
  transamination chemistry, with a per-substrate <em>side-chain fragment</em>
  (the <code>side_chain_smiles</code> column of <code>ONUFFER_curated.csv</code>)
  substituted in.  Every heavy atom carries a unique map number, satisfying
  <code>prepare_reaction</code>'s requirement.
</p>

<h2>Reactive-center template</h2>
<p>
  The transamination is written as a partial reaction (EVODEX sense): the amino
  acid and water are reactants; the alpha-keto acid and ammonia are products.
  PLP/PMP co-substrates are omitted because they carry no tracked atoms.
</p>
<table class="tmpl">
  <tr>
    <th>Role</th><th>Template fragment</th>
  </tr>
  <tr>
    <td>Reactants</td>
    <td><code>[N:1][C@@H:2]({{SC}})[C:3](=[O:4])[OH:5].[OH2:6]</code></td>
  </tr>
  <tr>
    <td>Products</td>
    <td><code>[N:1].[C:2](=[O:6])({{SC}})[C:3](=[O:4])[OH:5]</code></td>
  </tr>
</table>
<p>
  Atom-map legend:
  <code>:1</code> alpha-N &nbsp;·&nbsp;
  <code>:2</code> alpha-C &nbsp;·&nbsp;
  <code>:3</code> carboxyl-C &nbsp;·&nbsp;
  <code>:4</code> carboxyl =O &nbsp;·&nbsp;
  <code>:5</code> carboxyl -OH &nbsp;·&nbsp;
  <code>:6</code> water O (becomes product ketone =O) &nbsp;·&nbsp;
  <code>:10+</code> side-chain atoms (beta-carbon first).
</p>

<h2>Results summary</h2>
<ul>
  <li>Total substrates: <strong>{n_total}</strong></li>
  <li><span class="ok">prepare_reaction passed</span>: <strong>{n_ok}</strong></li>
  <li><span class="fail">prepare_reaction failed</span>: <strong>{n_fail - n_skipped}</strong></li>
  <li>Skipped (no side_chain_smiles): <strong>{n_skipped}</strong></li>
</ul>
"""

    rxn_table_rows = "\n".join(
        f'    <tr>\n'
        f'      <td class="num">{i}</td>\n'
        f'      <td>{r["substrate"]}</td>\n'
        f'      <td><code style="font-size:10px;word-break:break-all">'
        f'{r["reaction_smiles_full"]}</code></td>\n'
        f'    </tr>'
        for i, r in enumerate(results, 1)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Onuffer 1995 — Full Reaction Mapping</title>
<style>
  body  {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           font-size: 13px; color: #222; margin: 2rem; max-width: 1200px; }}
  h1   {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 0.1rem; }}
  h2   {{ font-size: 1.1rem; font-weight: 500; margin-top: 2rem; }}
  p.meta {{ color: #888; font-size: 11px; margin-top: 0; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  table.tmpl {{ width: auto; }}
  th   {{ text-align: left; font-weight: 500; font-size: 12px;
           color: #444; border-bottom: 2px solid #ddd; padding: 6px 10px; }}
  td   {{ border-bottom: 1px solid #eee; padding: 6px 10px; vertical-align: middle; }}
  tr:hover td {{ background: #f9f9f9; }}
  td.num  {{ color: #999; width: 28px; text-align: right; }}
  td.name {{ width: 190px; }}
  td.sc   {{ width: 260px; word-break: break-all; }}
  td.struct {{ width: 210px; text-align: center; }}
  td.err  {{ color: #c0392b; font-size: 11px; word-break: break-all; }}
  td.ok, span.ok   {{ color: #27ae60; font-weight: 600; }}
  td.fail, span.fail {{ color: #c0392b; font-weight: 600; }}
  td.warn {{ color: #e67e22; font-weight: 600; }}
  code  {{ font-family: "SF Mono", "Fira Mono", monospace; font-size: 11px;
            background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
  svg   {{ display: block; margin: auto; }}
  ul li {{ margin-bottom: 0.3rem; }}
</style>
</head>
<body>
<h1>Onuffer 1995 — Full Reaction Mapping</h1>
<p class="meta">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} by
  <code>map_reactions.py</code> from <code>ONUFFER_curated.csv</code>
</p>

{summary_prose}

<h2>Per-substrate results</h2>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Substrate</th>
      <th>side_chain_smiles</th>
      <th>Structure</th>
      <th>prepare_reaction</th>
      <th>Error</th>
    </tr>
  </thead>
  <tbody>
{''.join(table_rows_html)}
  </tbody>
</table>

<h2>Full reaction SMILES</h2>
<p>
  The table below shows the assembled reaction SMILES for each substrate.
  These are the strings passed to <code>prepare_reaction</code>.
</p>
<table>
  <thead>
    <tr><th>#</th><th>Substrate</th><th>reaction_smiles_full</th></tr>
  </thead>
  <tbody>
{rxn_table_rows}
  </tbody>
</table>

</body>
</html>
"""

    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"Wrote HTML report → {HTML_OUT}")
    print(f"\nSummary: {n_ok}/{n_total} passed prepare_reaction")
    if n_fail - n_skipped:
        print("Failures:")
        for r in results:
            if r["prepare_reaction_ok"] == "fail":
                print(f"  {r['substrate']}: {r['error'][:120]}")


if __name__ == "__main__":
    main()
