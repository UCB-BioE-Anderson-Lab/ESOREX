"""
glycan_assembler.py, Build glycan SMILES from moiety graphs using RDKit RWMol surgery.

Design
------
Each moiety is defined by its PubChem free-sugar SMILES.  Ring atoms are identified
by topology (not hard-coded atom indices), then numbered with standard IUPAC sugar
positions via RDKit atom maps:

    C1=1  C2=2  C3=3  C4=4  C5=5  C6=6   (ring carbons; C6 = exocyclic CH2OH or CH3)
    O5=50  (ring oxygen)
    O1=11  O2=12  O3=13  O4=14  O6=16    (exocyclic hydroxyls)
    N2=22  (NHAc nitrogen on GlcNAc C2)

When moieties are combined, map numbers are offset by node_index × 100, so every
atom in the assembled molecule has a unique map that encodes its moiety and position.

Glycosidic bond formation (condensation):
    1. Remove O1 (the anomeric OH, both O atom and its implicit H) from the donor.
    2. Remove the implicit H from the acceptor Ox (deprotonate).
    3. Form a single bond: donor C1 → acceptor Ox.

aziHex tag attachment (treated as an acceptor at its own O):
    1. Remove O1 from the reducing-end sugar.
    2. Deprotonate aziHex O.
    3. Form bond: sugar C1 → aziHex O.

Entry point
-----------
    smiles, atom_map = build_compound(chain, linkages, branches={})

where chain/linkages/branches match the COMPOUNDS dict in fuctiii_compound_data.py.
"""

from __future__ import annotations
from collections import deque
from rdkit import Chem
from rdkit.Chem import RWMol, AllChem
from rdkit.Chem.rdchem import BondType


# ── Moiety free-sugar SMILES from PubChem ────────────────────────────────────
# These are the ONLY structures taken from an external source.
# All compound SMILES are computed from these building blocks.

MOIETY_SMILES: dict[str, str] = {
    "Glc":    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",   # CID 5793  β-D-Glc
    "Gal":    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O",    # CID 6036  β-D-Gal
    "GlcNAc": "OC[C@H]1O[C@@H](O)[C@@H](NC(C)=O)[C@H](O)[C@@H]1O",  # CID 439174 β-D-GlcNAc, C5-first
    "Fuc":    "C[C@@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",    # CID 442519 α-L-Fuc
    # Neu5Ac pyranose (CID 445063 α-D-Neu5Ac): ring O is between C2 (anomeric) and C6.
    # C2 is identified as the anomeric carbon by its exocyclic carboxylate (C=O).
    "Sia":    "CC(=O)N[C@@H]1[C@@H](O)C[C@@](O)(C(=O)O)O[C@H]1[C@H](O)[C@H](O)CO",
}

AZIHEX_SMILES = "OCCCCCCN=[N+]=[N-]"

# Atom maps for aziHex atoms (block 700–720, fixed so substrate/product maps are stable)
# OCCCCCCN=[N+]=[N-]: atom indices 0=O, 1-6=CH2 chain, 7-9=N3
_AZIHEX_MAPS = {0: 700, 1: 701, 2: 702, 3: 703, 4: 704, 5: 705, 6: 706,
                7: 710, 8: 711, 9: 712}

# Position label → intra-moiety atom map offset
_POS_OFFSET: dict[str, int] = {
    "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6,
    "O5": 50,
    "O1": 11, "O2": 12, "O3": 13, "O4": 14, "O6": 16,
    "N2": 22,
}


# ── Pyranose ring finder ──────────────────────────────────────────────────────

def find_pyranose_positions(mol: Chem.Mol) -> dict[str, int | None] | None:
    """
    Identify ring-atom positions of a pyranose by topology alone.

    Strategy:
    - Find the unique ring oxygen (O5) in a 6-membered ring.
    - O5 has two ring-carbon neighbors: C1 (anomeric, has exocyclic O but no exocyclic C)
      and C5 (has exocyclic C = C6; may or may not have exocyclic O).
    - Walk C1→C2→C3→C4 through the remaining ring carbons.
    - Locate exocyclic atoms (O1-O4, O6, N2) hanging off each ring carbon.

    Returns a dict keyed by position label ('C1','O1', ...) with atom indices,
    or None if no suitable pyranose ring is found.
    """
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) != 6:
            continue
        ring_set = set(ring)

        # Must have exactly one ring oxygen
        ring_oxygens = [i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]
        if len(ring_oxygens) != 1:
            continue
        o5_idx = ring_oxygens[0]

        # O5's two ring-carbon neighbors
        o5_ring_neighbors = [
            n.GetIdx() for n in mol.GetAtomWithIdx(o5_idx).GetNeighbors()
            if n.GetIdx() in ring_set
        ]
        if len(o5_ring_neighbors) != 2:
            continue

        def exo_atoms(cidx: int):
            """Return (exo_oxygens, exo_carbons, exo_nitrogens) atom indices."""
            catom = mol.GetAtomWithIdx(cidx)
            exo_o, exo_c, exo_n = [], [], []
            for n in catom.GetNeighbors():
                nidx = n.GetIdx()
                if nidx in ring_set:
                    continue
                anum = n.GetAtomicNum()
                if anum == 8:
                    exo_o.append(nidx)
                elif anum == 6:
                    exo_c.append(nidx)
                elif anum == 7:
                    exo_n.append(nidx)
            return exo_o, exo_c, exo_n

        # Identify C1 and C5
        c1_idx, c5_idx = None, None
        for cidx in o5_ring_neighbors:
            eo, ec, en = exo_atoms(cidx)
            if ec:
                c5_idx = cidx   # C5 always has exocyclic C (C6)
            else:
                c1_idx = cidx   # C1 has exocyclic O (anomeric) but no exocyclic C

        if c1_idx is None or c5_idx is None:
            continue

        # Walk C1→C2→C3→C4 (away from O5, stopping before C5)
        prev, curr = o5_idx, c1_idx
        chain_c = [c1_idx]
        for _ in range(3):
            catom = mol.GetAtomWithIdx(curr)
            nxt = None
            for n in catom.GetNeighbors():
                nidx = n.GetIdx()
                if nidx in ring_set and nidx != prev and nidx != o5_idx and nidx != c5_idx:
                    nxt = nidx
                    break
            if nxt is None:
                break
            chain_c.append(nxt)
            prev, curr = curr, nxt

        if len(chain_c) != 4:
            continue
        c1, c2, c3, c4 = chain_c
        c5 = c5_idx

        # Verify C4–C5 are bonded
        if c5 not in {n.GetIdx() for n in mol.GetAtomWithIdx(c4).GetNeighbors()}:
            continue

        # C6: exocyclic carbon on C5
        eo5, ec5, en5 = exo_atoms(c5)
        c6_idx = ec5[0] if ec5 else None

        def _first_exo_o(cidx):
            eo, _, _ = exo_atoms(cidx)
            return eo[0] if eo else None

        def _first_exo_n(cidx):
            _, _, en = exo_atoms(cidx)
            return en[0] if en else None

        def _first_exo_o_on(cidx):
            # On C6: exocyclic O
            if cidx is None:
                return None
            catom = mol.GetAtomWithIdx(cidx)
            for n in catom.GetNeighbors():
                if n.GetAtomicNum() == 8:
                    return n.GetIdx()
            return None

        return {
            "C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6_idx,
            "O5": o5_idx,
            "O1": _first_exo_o(c1),
            "O2": _first_exo_o(c2),
            "O3": _first_exo_o(c3),
            "O4": _first_exo_o(c4),
            "O6": _first_exo_o_on(c6_idx),
            "N2": _first_exo_n(c2),
        }

    return None


# ── Sia-specific ring finder ──────────────────────────────────────────────────

def find_sia_positions(mol: Chem.Mol) -> dict[str, int | None] | None:
    """
    Identify ring-atom positions of Neu5Ac (sialic acid) by topology.

    Sia's 6-membered ring (O6–C2–C3–C4–C5–C6) differs from standard hexoses:
    - C2 (anomeric) has an exocyclic carboxylate (COOH), making it distinct from C6
    - Both ring-O neighbors have exocyclic carbons, so the standard hexose finder fails

    Strategy: the ring-O neighbor bearing an exocyclic carboxyl carbon (C with C=O) is
    C2 (our "C1"); the other neighbor (C6, with a plain alkyl side-chain) is our "C5".

    Position mapping (IUPAC sugar positions → Sia atom):
        C1 = Sia C2 (anomeric)     O1 = anomeric OH on Sia C2
        C2 = Sia C3 (methylene, so O2 = None)
        C3 = Sia C4                O3 = OH on Sia C4
        C4 = Sia C5 (NHAc)        N2 = NHAc-N on Sia C5
        C5 = Sia C6                C6 = Sia C7 (first of side chain)
        O5 = ring O (Sia O6)       O6 = OH on Sia C7
    """
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) != 6:
            continue
        ring_set = set(ring)

        ring_oxygens = [i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]
        if len(ring_oxygens) != 1:
            continue
        o5_idx = ring_oxygens[0]

        o5_ring_neighbors = [
            n.GetIdx() for n in mol.GetAtomWithIdx(o5_idx).GetNeighbors()
            if n.GetIdx() in ring_set
        ]
        if len(o5_ring_neighbors) != 2:
            continue

        def has_carboxyl_neighbor(cidx: int) -> bool:
            """True if cidx has an exocyclic C that bears a C=O double bond."""
            catom = mol.GetAtomWithIdx(cidx)
            for n in catom.GetNeighbors():
                if n.GetIdx() in ring_set or n.GetAtomicNum() != 6:
                    continue
                for bond in n.GetBonds():
                    if bond.GetBondTypeAsDouble() == 2.0:
                        other = bond.GetOtherAtom(n)
                        if other.GetAtomicNum() == 8:
                            return True
            return False

        c1_idx, c5_idx = None, None
        for cidx in o5_ring_neighbors:
            if has_carboxyl_neighbor(cidx):
                c1_idx = cidx   # Sia C2 → our C1
            else:
                c5_idx = cidx   # Sia C6 → our C5

        if c1_idx is None or c5_idx is None:
            continue

        prev, curr = o5_idx, c1_idx
        chain_c = [c1_idx]
        for _ in range(3):
            catom = mol.GetAtomWithIdx(curr)
            nxt = None
            for n in catom.GetNeighbors():
                nidx = n.GetIdx()
                if nidx in ring_set and nidx != prev and nidx != o5_idx and nidx != c5_idx:
                    nxt = nidx
                    break
            if nxt is None:
                break
            chain_c.append(nxt)
            prev, curr = curr, nxt

        if len(chain_c) != 4:
            continue
        c1, c2, c3, c4 = chain_c
        c5 = c5_idx

        if c5 not in {n.GetIdx() for n in mol.GetAtomWithIdx(c4).GetNeighbors()}:
            continue

        def exo_atoms(cidx: int):
            catom = mol.GetAtomWithIdx(cidx)
            exo_o, exo_c, exo_n = [], [], []
            for n in catom.GetNeighbors():
                nidx = n.GetIdx()
                if nidx in ring_set:
                    continue
                anum = n.GetAtomicNum()
                if anum == 8:   exo_o.append(nidx)
                elif anum == 6: exo_c.append(nidx)
                elif anum == 7: exo_n.append(nidx)
            return exo_o, exo_c, exo_n

        def _first_exo_o(cidx):
            eo, _, _ = exo_atoms(cidx)
            return eo[0] if eo else None

        def _first_exo_n(cidx):
            _, _, en = exo_atoms(cidx)
            return en[0] if en else None

        def _first_exo_c(cidx):
            _, ec, _ = exo_atoms(cidx)
            return ec[0] if ec else None

        def _first_exo_o_on(cidx):
            if cidx is None:
                return None
            catom = mol.GetAtomWithIdx(cidx)
            for n in catom.GetNeighbors():
                if n.GetAtomicNum() == 8:
                    return n.GetIdx()
            return None

        c6_idx = _first_exo_c(c5)   # Sia C7, first atom of C7-C8-C9 side chain

        return {
            "C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6_idx,
            "O5": o5_idx,
            "O1": _first_exo_o(c1),   # anomeric OH on Sia C2
            "O2": _first_exo_o(c2),   # None (Sia C3 is CH2)
            "O3": _first_exo_o(c3),   # OH on Sia C4
            "O4": _first_exo_o(c4),   # None (Sia C5 has N, not direct O)
            "O6": _first_exo_o_on(c6_idx),
            "N2": _first_exo_n(c4),   # NHAc-N on Sia C5
        }

    return None


# ── Moiety mol factory ────────────────────────────────────────────────────────

def make_moiety_mol(moiety_id: str, base: int = 0) -> tuple[Chem.Mol, dict]:
    """
    Parse moiety free-sugar SMILES, identify ring positions, assign atom maps.

    Maps are base + _POS_OFFSET[position].  E.g. base=100 → C1 gets map 101,
    O3 gets map 113, etc.

    Returns (mol_with_maps, positions_dict).
    positions_dict keys: 'C1'..'C6', 'O5', 'O1'..'O6', 'N2'; values: atom indices.
    """
    smi = MOIETY_SMILES[moiety_id]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES for {moiety_id}: {smi}")

    if moiety_id == "Sia":
        pos = find_sia_positions(mol)
        if pos is None:
            raise ValueError(f"Could not find Sia ring in {moiety_id} ({smi})")
    else:
        pos = find_pyranose_positions(mol)
        if pos is None:
            raise ValueError(f"Could not find pyranose ring in {moiety_id} ({smi})")

    rwmol = RWMol(mol)
    for pos_name, atom_idx in pos.items():
        if atom_idx is not None and pos_name in _POS_OFFSET:
            rwmol.GetAtomWithIdx(atom_idx).SetAtomMapNum(base + _POS_OFFSET[pos_name])

    Chem.SanitizeMol(rwmol)
    return rwmol.GetMol(), pos


# ── Atom lookup by map ────────────────────────────────────────────────────────

def _atom_by_map(mol: Chem.Mol, map_num: int) -> int | None:
    """Return atom index with the given atom map, or None."""
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom.GetIdx()
    return None


# ── Valence helpers ───────────────────────────────────────────────────────────

def _remove_atom(rwmol: RWMol, idx: int) -> RWMol:
    """Remove atom (and all its bonds) from rwmol; return same rwmol."""
    rwmol.RemoveAtom(idx)
    return rwmol


def _deprotonate_oxygen(rwmol: RWMol, o_idx: int) -> None:
    """
    Remove the implicit H from an -OH at o_idx, making it ready to bond.
    After this the O has valence 1 (to its existing C) and can accept one more bond.
    """
    o_atom = rwmol.GetAtomWithIdx(o_idx)
    total_h = o_atom.GetTotalNumHs()
    if total_h > 0:
        o_atom.SetNumExplicitHs(0)
        o_atom.SetNoImplicit(True)


# ── Core bond-forming operations ──────────────────────────────────────────────

def attach_azihex(chain_mol: Chem.Mol, c1_map: int, o1_map: int) -> Chem.Mol:
    """
    Replace the anomeric OH on the reducing-end sugar with an aziHex aglycon.

    Steps:
    1. Remove O1 (the anomeric -OH, atom and implicit H) from chain_mol.
    2. Append aziHex; its terminal O becomes the glycosidic oxygen.
    3. Form bond: sugar-C1 → aziHex-O.
    """
    rwmol = RWMol(chain_mol)

    o1_idx = _atom_by_map(rwmol, o1_map)
    if o1_idx is None:
        raise ValueError(f"O1 (map={o1_map}) not found in reducing-end mol")
    _remove_atom(rwmol, o1_idx)

    # aziHex: first atom (index 0) is the O that bonds to C1; assign fixed maps
    azihex_mol = Chem.MolFromSmiles(AZIHEX_SMILES)
    azihex_rw = RWMol(azihex_mol)
    for aidx, amap in _AZIHEX_MAPS.items():
        azihex_rw.GetAtomWithIdx(aidx).SetAtomMapNum(amap)
    azihex_mol = azihex_rw.GetMol()
    azihex_o_idx = 0  # O in OCCCCCCN=[N+]=[N-]

    combo = RWMol(Chem.CombineMols(rwmol.GetMol(), azihex_mol))
    n_sugar = rwmol.GetNumAtoms()

    c1_in_combo = _atom_by_map(combo, c1_map)
    az_o_in_combo = n_sugar + azihex_o_idx

    if c1_in_combo is None:
        raise ValueError(f"C1 (map={c1_map}) not found after O1 removal")

    _deprotonate_oxygen(combo, az_o_in_combo)
    combo.AddBond(c1_in_combo, az_o_in_combo, BondType.SINGLE)
    Chem.SanitizeMol(combo)
    return combo.GetMol()


def attach_donor(chain_mol: Chem.Mol,
                 acceptor_o_map: int,
                 donor_mol: Chem.Mol,
                 donor_c1_map: int,
                 donor_o1_map: int) -> Chem.Mol:
    """
    Form a glycosidic bond: donor C1 → acceptor Ox (already in chain).

    Steps:
    1. Remove donor O1 (anomeric OH).
    2. Combine donor + chain into one mol.
    3. Deprotonate acceptor Ox (remove its implicit H).
    4. Bond donor C1 → acceptor Ox.
    """
    donor_rw = RWMol(donor_mol)
    o1_idx = _atom_by_map(donor_rw, donor_o1_map)
    if o1_idx is None:
        raise ValueError(f"Donor O1 (map={donor_o1_map}) not found")
    _remove_atom(donor_rw, o1_idx)

    # Combine: chain first, then (trimmed) donor
    combo = RWMol(Chem.CombineMols(chain_mol, donor_rw.GetMol()))

    acceptor_o_in_combo = _atom_by_map(combo, acceptor_o_map)
    donor_c1_in_combo  = _atom_by_map(combo, donor_c1_map)

    if acceptor_o_in_combo is None:
        raise ValueError(f"Acceptor O (map={acceptor_o_map}) not found in chain")
    if donor_c1_in_combo is None:
        raise ValueError(f"Donor C1 (map={donor_c1_map}) not found")

    _deprotonate_oxygen(combo, acceptor_o_in_combo)
    combo.AddBond(donor_c1_in_combo, acceptor_o_in_combo, BondType.SINGLE)
    Chem.SanitizeMol(combo)
    return combo.GetMol()


# ── Top-level assembler ───────────────────────────────────────────────────────

def build_compound(chain: list[str],
                   linkages: list[tuple],
                   branches: dict | None = None) -> tuple[str, dict]:
    """
    Assemble a glycan SMILES from its moiety graph.

    Parameters
    ----------
    chain     : moiety IDs from NR end → reducing end → "aziHex".
                e.g. ["Gal", "GlcNAc", "aziHex"]
    linkages  : list of (from_node, to_node, linkage_str, donor_pos, acceptor_pos).
                from_node is always the NR-end (donor) node; to_node is the acceptor.
                "tag" linkages connect aziHex to the reducing-end sugar.
                donor_pos=1 means C1 of from_node; acceptor_pos=N means CN of to_node.
    branches  : {parent_node_idx: [{"moiety": id, "acceptor_pos": int, ...}]}
                Branch nodes (e.g. Fuc on Gal) attached outside the backbone chain.

    Returns
    -------
    (smiles, atom_map_legend) where atom_map_legend maps map_number → "MoietyID:position".
    """
    if branches is None:
        branches = {}

    # ── Find aziHex node and its attached sugar ──────────────────────────────
    azihex_node: int | None = None
    reducing_end_node: int | None = None
    for lk in linkages:
        fi, ti, lkstr, dp, ap = lk
        if lkstr == "tag":
            # convention in COMPOUNDS: (azihex_node, sugar_node, "tag", None, 1)
            azihex_node = fi
            reducing_end_node = ti
            break

    if azihex_node is None:
        raise ValueError("No 'tag' linkage found, cannot identify reducing end")

    # ── Assign base atom-map offsets per node (skip aziHex) ─────────────────
    # Use node_index * 100 so each moiety occupies a distinct map block.
    node_bases = {
        i: i * 100
        for i in range(len(chain))
        if chain[i] != "aziHex"
    }

    # Build a legend: map_num → human label
    atom_map_legend: dict[int, str] = {}
    for node_idx, base in node_bases.items():
        mid = chain[node_idx]
        for pos, offset in _POS_OFFSET.items():
            atom_map_legend[base + offset] = f"{mid}[node{node_idx}]:{pos}"

    # ── Create atom-mapped mol for each sugar node ───────────────────────────
    moiety_mols: dict[int, Chem.Mol] = {}
    for node_idx in node_bases:
        mol, _ = make_moiety_mol(chain[node_idx], base=node_bases[node_idx])
        moiety_mols[node_idx] = mol

    # ── Start chain with reducing-end sugar ──────────────────────────────────
    re_base = node_bases[reducing_end_node]
    chain_mol = moiety_mols[reducing_end_node]

    # ── Attach aziHex at the reducing-end C1 ─────────────────────────────────
    chain_mol = attach_azihex(chain_mol,
                              c1_map=re_base + _POS_OFFSET["C1"],
                              o1_map=re_base + _POS_OFFSET["O1"])

    # ── Walk backbone toward NR end, attaching each donor ───────────────────
    # Build adjacency for non-tag linkages
    adj: dict[int, list[tuple[int, tuple]]] = {}
    for lk in linkages:
        fi, ti, lkstr, dp, ap = lk
        if lkstr == "tag":
            continue
        adj.setdefault(fi, []).append((ti, lk))
        adj.setdefault(ti, []).append((fi, lk))

    visited = {reducing_end_node, azihex_node}
    queue = deque([reducing_end_node])
    while queue:
        node = queue.popleft()
        for neighbor, lk in adj.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)

            # neighbor is the new donor (not yet in chain)
            # lk = (fi, ti, lkstr, donor_pos, acceptor_pos)
            fi, ti, lkstr, dp, ap = lk
            donor_node = neighbor
            acceptor_node = node

            # In COMPOUNDS, fi = NR-end (donor), ti = more-reducing (acceptor)
            if fi == donor_node:
                acceptor_c_pos = ap   # position on acceptor (ti)
            else:
                # linkage is encoded with ti as donor, use dp as acceptor position
                acceptor_c_pos = dp

            acceptor_o_map = node_bases[acceptor_node] + _POS_OFFSET[f"O{acceptor_c_pos}"]
            donor_c1_map   = node_bases[donor_node]   + _POS_OFFSET["C1"]
            donor_o1_map   = node_bases[donor_node]   + _POS_OFFSET["O1"]

            chain_mol = attach_donor(chain_mol, acceptor_o_map,
                                     moiety_mols[donor_node],
                                     donor_c1_map, donor_o1_map)

    # ── Attach branch substituents ────────────────────────────────────────────
    next_branch_base = (max(node_bases.values()) + 100)
    for parent_node_idx, branch_list in branches.items():
        for b in branch_list:
            bm = b["moiety"]
            bap = b.get("acceptor_pos")
            if bap is None or bm not in MOIETY_SMILES:
                continue  # Sia branches handled separately; skip unknowns

            b_mol, _ = make_moiety_mol(bm, base=next_branch_base)
            acceptor_o_map = node_bases[parent_node_idx] + _POS_OFFSET[f"O{bap}"]
            b_c1_map = next_branch_base + _POS_OFFSET["C1"]
            b_o1_map = next_branch_base + _POS_OFFSET["O1"]

            # Add branch legend entries
            for pos, offset in _POS_OFFSET.items():
                atom_map_legend[next_branch_base + offset] = f"{bm}[branch@node{parent_node_idx}]:{pos}"

            chain_mol = attach_donor(chain_mol, acceptor_o_map, b_mol, b_c1_map, b_o1_map)
            next_branch_base += 100

    # ── Assign sequential maps to any remaining unmapped heavy atoms ─────────
    # Covers NHAc acetyl groups, carboxylates, etc. that weren't assigned above.
    chain_mol = _fill_remaining_maps(chain_mol, start=800)

    return Chem.MolToSmiles(chain_mol), chain_mol, atom_map_legend


# ── Fill unmapped atoms ───────────────────────────────────────────────────────

def _fill_remaining_maps(mol: Chem.Mol, start: int = 800) -> Chem.Mol:
    """Assign sequential atom maps to every heavy atom that still has map=0."""
    rw = RWMol(mol)
    counter = start
    for atom in rw.GetAtoms():
        if atom.GetAtomMapNum() == 0 and atom.GetAtomicNum() != 1:
            atom.SetAtomMapNum(counter)
            counter += 1
    Chem.SanitizeMol(rw)
    return rw.GetMol()


# ── Product builder ───────────────────────────────────────────────────────────

def build_product(substrate_mol: Chem.Mol,
                  rxn_node_idx: int,
                  rxn_pos: str) -> Chem.Mol:
    """
    Attach Fuc to the substrate at the FucTIII reaction site.

    substrate_mol : fully atom-mapped substrate (from build_compound).
    rxn_node_idx  : chain node index of the acceptor residue (e.g. 1 for node 1).
    rxn_pos       : IUPAC position string, e.g. "C3" or "C4".

    The Fuc atoms in the product carry NO atom maps (map=0), so a viewer can
    immediately see which atoms originated from the substrate vs. which came
    from GDP-fucose.

    Returns the product Chem.Mol (atom maps on substrate portion, 0 on Fuc).
    """
    pos_num = int(rxn_pos[1])  # "C3" → 3
    acceptor_o_map = rxn_node_idx * 100 + _POS_OFFSET[f"O{pos_num}"]

    # Fuc with temporary high-base maps so attach_donor can find C1 / O1
    _FUC_TMP_BASE = 9000
    fuc_mol, _ = make_moiety_mol("Fuc", base=_FUC_TMP_BASE)

    product = attach_donor(substrate_mol, acceptor_o_map, fuc_mol,
                           _FUC_TMP_BASE + _POS_OFFSET["C1"],
                           _FUC_TMP_BASE + _POS_OFFSET["O1"])

    # Strip Fuc maps → map 0 (cofactor-derived, not tracked).
    # prepare_reaction accepts map-0 product atoms when the reactant uses explicit [H].
    rw = RWMol(product)
    for atom in rw.GetAtoms():
        if atom.GetAtomMapNum() >= _FUC_TMP_BASE:
            atom.SetAtomMapNum(0)
    Chem.SanitizeMol(rw)
    return rw.GetMol()


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rdkit.Chem import rdMolDescriptors

    tests = [
        # (label, chain, linkages, branches, expected_formula)
        ("LacNAc-azide (cmpd 11)",
         ["Gal", "GlcNAc", "aziHex"],
         [(0, 1, "β1,4", 1, 4), (2, 1, "tag", None, 1)],
         {},
         "C20H36N4O11"),

        ("LNB-azide (cmpd 19)",
         ["Gal", "GlcNAc", "aziHex"],
         [(0, 1, "β1,3", 1, 3), (2, 1, "tag", None, 1)],
         {},
         "C20H36N4O11"),

        ("Lc3-azide (cmpd 2)",
         ["GlcNAc", "Gal", "Glc", "aziHex"],
         [(0, 1, "β1,3", 1, 3), (1, 2, "β1,4", 1, 4), (3, 2, "tag", None, 1)],
         {},
         "C26H46N4O16"),

        ("GlcNAcβ1,3-LacNAc-azide (cmpd 12)",
         ["GlcNAc", "Gal", "GlcNAc", "aziHex"],
         [(0, 1, "β1,3", 1, 3), (1, 2, "β1,4", 1, 4), (3, 2, "tag", None, 1)],
         {},
         "C28H49N5O16"),

        ("LNT-azide (cmpd 3)",
         ["Gal", "GlcNAc", "Gal", "Glc", "aziHex"],
         [(0, 1, "β1,3", 1, 3), (1, 2, "β1,3", 1, 3), (2, 3, "β1,4", 1, 4),
          (4, 3, "tag", None, 1)],
         {},
         "C32H56N4O21"),

        ("LNnT-azide (cmpd 4)",
         ["Gal", "GlcNAc", "Gal", "Glc", "aziHex"],
         [(0, 1, "β1,4", 1, 4), (1, 2, "β1,3", 1, 3), (2, 3, "β1,4", 1, 4),
          (4, 3, "tag", None, 1)],
         {},
         "C32H56N4O21"),

        ("Fucα1,2-LNB-azide (cmpd 21)",
         ["Gal", "GlcNAc", "aziHex"],
         [(0, 1, "β1,3", 1, 3), (2, 1, "tag", None, 1)],
         {0: [{"moiety": "Fuc", "acceptor_pos": 2}]},
         "C26H46N4O15"),

        ("Neu5Acα2,3-LNB-azide (cmpd 22)",
         ["Gal", "GlcNAc", "aziHex"],
         [(0, 1, "β1,3", 1, 3), (2, 1, "tag", None, 1)],
         {0: [{"moiety": "Sia", "acceptor_pos": 3}]},
         "C31H53N5O19"),

        ("Neu5Acα2,3-LacNAc-azide (cmpd 17)",
         ["Gal", "GlcNAc", "aziHex"],
         [(0, 1, "β1,4", 1, 4), (2, 1, "tag", None, 1)],
         {0: [{"moiety": "Sia", "acceptor_pos": 3}]},
         "C31H53N5O19"),

        ("Fucα1,2-LacNAc-azide (cmpd 15)",
         ["Gal", "GlcNAc", "aziHex"],
         [(0, 1, "β1,4", 1, 4), (2, 1, "tag", None, 1)],
         {0: [{"moiety": "Fuc", "acceptor_pos": 2}]},
         "C26H46N4O15"),
    ]

    print(f"{'Label':<40} {'Result':<10} {'Formula'}")
    print("-" * 75)
    for label, chain, linkages, branches, expected_formula in tests:
        try:
            smi, mol, _ = build_compound(chain, linkages, branches)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"{label:<40} INVALID")
                continue
            formula = rdMolDescriptors.CalcMolFormula(Chem.AddHs(mol))
            ok = "✓" if formula == expected_formula else "✗"
            print(f"{label:<40} {ok}         {formula}  (expected {expected_formula})")
        except Exception as e:
            print(f"{label:<40} ERROR     {e}")
