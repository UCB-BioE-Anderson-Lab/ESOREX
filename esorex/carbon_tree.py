"""Carbon-only ensemble tree: the specificity model's representation.

A substrate, minus its reactive core, leaves passenger fragments.  Each passenger's
carbon skeleton is the tree; heteroatoms, bond orders, and oxidation become deltas hung
on the carbons.  Heteroatoms are grouped into connected clusters: a cluster touching one
carbon is a SUBSTITUENT delta, a cluster touching two or more carbons is a BRIDGE that
keeps the carbon skeleton connected across the heteroatom (so, e.g., arginine's
guanidinium carbon is not lost when the Nepsilon separates it from the chain).

The per-substrate tree is addressed in a molecule-local canonical (seed) order.  To
compare positions across substrates the trees are aligned into one shared frame by the
alignment that minimizes electron edit distance (permuting sibling order at each branch
point, that is, rotating bonds), grown most-active-first.  This module owns that physics
and that alignment; `substrate_tree` remains a separate structural identity/hash object.
"""
from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

# Pauling electronegativities for oxidation-state assignment (self-contained physics).
_EN = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 11: 0.93, 15: 2.19,
       16: 2.58, 17: 3.16, 34: 2.55, 35: 2.96, 53: 2.66}
_PT = Chem.GetPeriodicTable()


def _oxidation_state(atom) -> int:
    """Oxidation number: each bond's electrons to the more electronegative partner
    (Pauling), plus formal charge; implicit H is the less electronegative partner."""
    en = _EN.get(atom.GetAtomicNum(), 2.5)
    ox = float(atom.GetFormalCharge())
    for b in atom.GetBonds():
        other = b.GetOtherAtom(atom)
        eno = _EN.get(other.GetAtomicNum(), 2.5)
        order = b.GetBondTypeAsDouble()
        if eno > en:
            ox += order
        elif eno < en:
            ox -= order
    nH = atom.GetTotalNumHs()
    ox += -nH if en > _EN[1] else nH
    return int(round(ox))


def _lone_pairs(atom) -> int:
    """Orbital availability: nonbonding pairs = (valence electrons - bonds - charge)/2."""
    v = _PT.GetNOuterElecs(atom.GetAtomicNum())
    nonbonding = v - int(atom.GetTotalValence()) - atom.GetFormalCharge()
    return max(0, nonbonding // 2)


# ---------------------------------------------------------------------------
# Per-substrate carbon forest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeteroCluster:
    """A connected group of heteroatoms, as a substituent (one carbon) or bridge (>=2)."""
    atoms: tuple            # sorted tuple[(atomic_num, formal_charge, n_h, lone_pairs, aromatic)]
    carbon_bond_orders: tuple
    n_carbons: int

    @property
    def is_bridge(self) -> bool:
        return self.n_carbons >= 2


@dataclass
class CNode:
    """A node in a substrate's carbon tree: a carbon with its deltas, or a structural
    root.  `kind` is 'virtual' (whole substrate), 'passenger' (one fragment root holder),
    or 'carbon'."""
    kind: str
    sidx: int = -1                       # substrate atom index for carbons
    oxidation_state: int = 0
    pi_orders: tuple = ()
    ring_closures: int = 0
    substituents: tuple = ()             # tuple[HeteroCluster]
    in_bridge: Optional[HeteroCluster] = None   # bridge cluster joining this node to its parent
    kids: Dict[int, "CNode"] = field(default_factory=dict)

    def delta_key(self) -> tuple:
        """Hashable summary of this carbon's chemistry (for alignment cost + equality)."""
        return (self.oxidation_state, self.pi_orders, self.ring_closures,
                self.substituents, self.in_bridge)


def _passenger_components(mol, core):
    core = set(core)
    seen, comps = set(), []
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i in core or i in seen or a.GetAtomicNum() == 1:
            continue
        stack, comp = [i], set()
        while stack:
            j = stack.pop()
            if j in comp or j in core:
                continue
            comp.add(j); seen.add(j)
            for nb in mol.GetAtomWithIdx(j).GetNeighbors():
                if nb.GetIdx() not in core and nb.GetAtomicNum() != 1:
                    stack.append(nb.GetIdx())
        comps.append(comp)
    return comps


def _hetero_clusters(mol, hets):
    seen, out = set(), []
    for h in hets:
        if h in seen:
            continue
        comp, dq = set(), [h]
        while dq:
            x = dq.pop()
            if x in comp:
                continue
            comp.add(x); seen.add(x)
            for nb in mol.GetAtomWithIdx(x).GetNeighbors():
                if nb.GetIdx() in hets and nb.GetIdx() not in comp:
                    dq.append(nb.GetIdx())
        out.append(comp)
    return out


def _cluster_descriptor(mol, cluster, touched) -> HeteroCluster:
    atoms = []
    for i in cluster:
        a = mol.GetAtomWithIdx(i)
        atoms.append((a.GetAtomicNum(), a.GetFormalCharge(), a.GetTotalNumHs(),
                      _lone_pairs(a), a.GetIsAromatic()))
    orders = []
    for i in cluster:
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() in touched:
                orders.append(mol.GetBondBetweenAtoms(i, nb.GetIdx()).GetBondTypeAsDouble())
    return HeteroCluster(atoms=tuple(sorted(atoms)),
                         carbon_bond_orders=tuple(sorted(orders)), n_carbons=len(touched))


def build_carbon_tree(mol: Chem.Mol, core) -> CNode:
    """Build the substrate's carbon forest as a virtual-root CNode whose children are the
    passenger fragment trees.  Child order is a molecule-local canonical seed; alignment
    refines it."""
    core = set(core)
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    virtual = CNode(kind="virtual")
    pslot = 0
    for comp in sorted(_passenger_components(mol, core),
                       key=lambda comp: min(rank[i] for i in comp)):
        carbons = {i for i in comp if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
        hets = {i for i in comp if mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6)}
        if not carbons:
            continue

        cadj = defaultdict(set)
        for c in carbons:
            for nb in mol.GetAtomWithIdx(c).GetNeighbors():
                if nb.GetIdx() in carbons:
                    cadj[c].add(nb.GetIdx())
        subst = defaultdict(list)
        bridge_of = {}
        for cl in _hetero_clusters(mol, hets):
            touched = {nb.GetIdx() for h in cl for nb in mol.GetAtomWithIdx(h).GetNeighbors()
                       if nb.GetIdx() in carbons}
            desc = _cluster_descriptor(mol, cl, touched)
            if len(touched) <= 1:
                for c in touched:
                    subst[c].append(desc)
            else:
                tl = sorted(touched)
                for i in range(len(tl)):
                    for j in range(i + 1, len(tl)):
                        cadj[tl[i]].add(tl[j]); cadj[tl[j]].add(tl[i])
                        bridge_of[frozenset((tl[i], tl[j]))] = desc

        rootsc = [c for c in carbons
                  if any(nb.GetIdx() in core for nb in mol.GetAtomWithIdx(c).GetNeighbors())]
        root_c = min(rootsc or carbons, key=lambda i: rank[i])

        # a passenger holder node, so multi-passenger substrates align passenger-to-passenger
        holder = CNode(kind="passenger")
        virtual.kids[pslot] = holder
        pslot += 1

        built: Dict[int, CNode] = {}
        visited = {root_c}
        q = deque([(root_c, None, holder)])
        while q:
            c, parent, pnode = q.popleft()
            pi = tuple(sorted(
                mol.GetBondBetweenAtoms(c, nb.GetIdx()).GetBondTypeAsDouble()
                for nb in mol.GetAtomWithIdx(c).GetNeighbors()
                if nb.GetIdx() in carbons
                and mol.GetBondBetweenAtoms(c, nb.GetIdx()).GetBondTypeAsDouble() != 1.0))
            node = CNode(
                kind="carbon", sidx=c,
                oxidation_state=_oxidation_state(mol.GetAtomWithIdx(c)),
                pi_orders=pi,
                ring_closures=sum(1 for v in cadj[c] if v in visited and v != parent),
                substituents=tuple(subst.get(c, ())),
                in_bridge=bridge_of.get(frozenset((c, parent))) if parent is not None else None)
            built[c] = node
            kids = sorted((v for v in cadj[c] if v not in visited), key=lambda i: rank[i])
            for k, ch in enumerate(kids):
                visited.add(ch)
                q.append((ch, c, node))
        # attach children in seed order under their parents
        # rebuild parent->child links using BFS order captured above
        # (redo the walk to set kids dicts deterministically)
        _link_children(mol, cadj, rank, root_c, built, holder)
    return virtual


def _link_children(mol, cadj, rank, root_c, built, holder):
    visited = {root_c}
    holder.kids[0] = built[root_c]
    q = deque([root_c])
    while q:
        c = q.popleft()
        kids = sorted((v for v in cadj[c] if v not in visited), key=lambda i: rank[i])
        for k, ch in enumerate(kids):
            visited.add(ch)
            built[c].kids[k] = built[ch]
            q.append(ch)


# ---------------------------------------------------------------------------
# Alignment: match trees, minimize electron edit distance, grow a shared frame
# ---------------------------------------------------------------------------

_UNMATCHED = 4.0        # cost of leaving one carbon node unmatched (per node, plus subtree)
_MAX_FANOUT = 7


def _node_diff(a: Optional[CNode], b: Optional[CNode]) -> float:
    if a is None or b is None:
        return 2.0
    ka, kb = a.delta_key(), b.delta_key()
    return float(sum(1 for x, y in zip(ka, kb) if x != y))


def _size(n: Optional[CNode]) -> float:
    if n is None:
        return 0.0
    return _UNMATCHED + sum(_size(c) for c in n.kids.values())


def _cost(ra: Optional[CNode], qa: Optional[CNode]) -> float:
    if ra is None or qa is None:
        return _size(ra if ra is not None else qa)
    return _node_diff(ra, qa) + _match(ra.kids, qa.kids)[0]


def _match(ref_kids: Dict[int, CNode], qry_kids: Dict[int, CNode]):
    """Order-independent optimal matching of query children to reference children.
    Returns (cost, pairs) where each pair is (ref_index|None, qry_index|None)."""
    ref_idx, qry_idx = list(ref_kids), list(qry_kids)
    if not ref_idx and not qry_idx:
        return 0.0, []
    if max(len(ref_idx), len(qry_idx)) > _MAX_FANOUT:
        return _match_greedy(ref_kids, qry_kids)
    best = [float("inf"), []]
    used = [False] * len(qry_idx)

    def rec(i, cost, pairs):
        if cost >= best[0]:
            return
        if i == len(ref_idx):
            total, extra = cost, []
            for k, qj in enumerate(qry_idx):
                if not used[k]:
                    total += _cost(None, qry_kids[qj]); extra.append((None, qj))
            if total < best[0]:
                best[0], best[1] = total, pairs + extra
            return
        ri = ref_idx[i]
        for k, qj in enumerate(qry_idx):
            if not used[k]:
                used[k] = True
                rec(i + 1, cost + _cost(ref_kids[ri], qry_kids[qj]), pairs + [(ri, qj)])
                used[k] = False
        rec(i + 1, cost + _cost(ref_kids[ri], None), pairs + [(ri, None)])

    rec(0, 0.0, [])
    return best[0], best[1]


def _match_greedy(ref_kids, qry_kids):
    order = sorted(((ri, qj) for ri in ref_kids for qj in qry_kids),
                   key=lambda rq: _cost(ref_kids[rq[0]], qry_kids[rq[1]]))
    ur, uq, pairs, cost = set(ref_kids), set(qry_kids), [], 0.0
    for ri, qj in order:
        if ri in ur and qj in uq:
            ur.discard(ri); uq.discard(qj)
            cost += _cost(ref_kids[ri], qry_kids[qj]); pairs.append((ri, qj))
    for ri in ur:
        cost += _cost(ref_kids[ri], None); pairs.append((ri, None))
    for qj in uq:
        cost += _cost(None, qry_kids[qj]); pairs.append((None, qj))
    return cost, pairs


def _next_free(kids) -> int:
    return max(kids, default=-1) + 1


def _realign(ref: Optional[CNode], qry: CNode, path: tuple,
             coord: Dict[int, tuple], grow: bool):
    """Address query node `qry` at `path` in the frame `ref`, recursing with the optimal
    child matching.  Unmatched query nodes get fresh indices beyond the frame's range; if
    `grow`, they are also folded into the frame (training), otherwise they are novel
    positions the collapse will zero (prediction)."""
    if qry.kind == "carbon":
        coord[qry.sidx] = path
    ref_kids = ref.kids if ref is not None else {}
    _, pairs = _match(ref_kids, qry.kids)
    free = _next_free(ref_kids)
    for ref_i, qry_j in pairs:
        if qry_j is None:
            continue
        qchild = qry.kids[qry_j]
        if ref_i is None:
            ni = free; free += 1
            if grow and ref is not None:
                child = _clone_shallow(qchild)
                ref.kids[ni] = child
                _realign(child, qchild, path + (ni,), coord, grow)
            else:
                _realign(None, qchild, path + (ni,), coord, grow)
        else:
            _realign(ref.kids[ref_i], qchild, path + (ref_i,), coord, grow)


def _clone_shallow(n: CNode) -> CNode:
    return CNode(kind=n.kind, sidx=n.sidx, oxidation_state=n.oxidation_state,
                 pi_orders=n.pi_orders, ring_closures=n.ring_closures,
                 substituents=n.substituents, in_bridge=n.in_bridge)


@dataclass
class AlignedEnsemble:
    """The shared frame plus each substrate's carbon coordinate map (sidx -> path)."""
    frame: CNode
    coords: List[Dict[int, tuple]]
    trees: List[CNode]


def build_ensemble(trees: List[CNode], energies) -> AlignedEnsemble:
    """Grow one shared frame most-active-first; return each tree's coordinate map.
    A coordinate is the tuple path from the virtual root (passenger slot, then carbon
    sibling indices) in the shared frame.  Lower energy = more active = seeds first."""
    order = sorted(range(len(trees)), key=lambda i: energies[i])
    best = order[0]
    frame = _clone_tree(trees[best])                      # seed = the most active substrate
    coords: List[Optional[Dict[int, tuple]]] = [None] * len(trees)
    for ti in order:
        if ti == best:
            cmap: Dict[int, tuple] = {}
            _fill_coords(trees[ti], (), cmap)             # seed's own paths are the frame's
            coords[ti] = cmap
        else:
            cmap = {}
            _realign(frame, trees[ti], (), cmap, grow=True)
            coords[ti] = cmap
    return AlignedEnsemble(frame=frame, coords=coords, trees=trees)


def _clone_tree(n: CNode) -> CNode:
    c = _clone_shallow(n)
    for k, ch in n.kids.items():
        c.kids[k] = _clone_tree(ch)
    return c


def _fill_coords(n: CNode, path: tuple, coord: Dict[int, tuple]):
    if n.kind == "carbon":
        coord[n.sidx] = path
    for k, ch in n.kids.items():
        _fill_coords(ch, path + (k,), coord)


def align_to_frame(frame: CNode, tree: CNode) -> Dict[int, tuple]:
    """Address a single (query) tree onto a fixed shared frame; returns sidx -> path.
    Does not mutate the frame: novel query positions get fresh coordinates the collapse
    treats as unsupported."""
    cmap: Dict[int, tuple] = {}
    _realign(frame, tree, (), cmap, grow=False)
    return cmap


def carbon_nodes(tree: CNode) -> Dict[int, CNode]:
    """Map each carbon's substrate atom index to its CNode."""
    out: Dict[int, CNode] = {}

    def walk(n: CNode):
        if n.kind == "carbon":
            out[n.sidx] = n
        for c in n.kids.values():
            walk(c)

    walk(tree)
    return out


# ---------------------------------------------------------------------------
# Rigid-body (sigma/pi) subdomain layer: the middle positional abstraction
# ---------------------------------------------------------------------------
# The passenger's carbons partition deterministically at sigma/pi boundaries into
# RIGID subdomains (rings, conjugated/aromatic pi systems, merged when they overlap)
# and FLEXIBLE linkers (maximal sp3 runs).  A subdomain is addressed by its hop count
# from the core in units of subdomains, so length INSIDE a subdomain is abstracted away
# (a 3-methylene and a 4-methylene linker are the same subdomain).  This is the middle
# rung of the atom -> subdomain -> passenger abstraction ladder; it exists so the model
# can learn "wants a hydrophobic linker here" without inventing a carbon-count threshold.

@dataclass(frozen=True)
class Subdomain:
    hop: int                 # subdomains from the core (length-independent)
    kind: str                # 'linker' | 'ring' | 'aromatic_ring' | 'conjugated'
    aromatic: bool
    in_ring: bool
    hydrophobic: bool        # no heteroatom deltas on any of its carbons

    @property
    def addr(self) -> str:
        return f"h{self.hop}:{self.kind}"


def _connected_carbons(mol, atoms):
    seen, comps = set(), []
    for s in atoms:
        if s in seen:
            continue
        comp, stack = set(), [s]
        while stack:
            x = stack.pop()
            if x in comp or x not in atoms:
                continue
            comp.add(x); seen.add(x)
            for nb in mol.GetAtomWithIdx(x).GetNeighbors():
                if nb.GetIdx() in atoms and nb.GetIdx() not in seen:
                    stack.append(nb.GetIdx())
        comps.append(comp)
    return comps


def _merge_overlapping(sets):
    out = list(sets)
    changed = True
    while changed:
        changed = False
        res, used = [], [False] * len(out)
        for i in range(len(out)):
            if used[i]:
                continue
            cur = set(out[i])
            for j in range(i + 1, len(out)):
                if not used[j] and cur & out[j]:
                    cur |= out[j]; used[j] = True; changed = True
            res.append(cur)
        out = res
    return out


def _partition_subdomains(mol, carbons):
    """(subdomains {sid: set of carbon idx}, atom->sid).  Rigid rings/pi merged; the
    remaining sp3 carbons are flexible linkers."""
    ri = mol.GetRingInfo()
    rigid = [set(r) & carbons for r in ri.AtomRings() if set(r) & carbons]
    pi = set()
    for c in carbons:
        a = mol.GetAtomWithIdx(c)
        if a.GetIsAromatic():
            pi.add(c)
        elif a.GetHybridization() == Chem.HybridizationType.SP2:
            for b in a.GetBonds():
                if b.GetBondTypeAsDouble() >= 2.0 and b.GetOtherAtomIdx(c) in carbons:
                    pi.add(c); pi.add(b.GetOtherAtomIdx(c))
    rigid += _connected_carbons(mol, pi & carbons)
    rigid = _merge_overlapping(rigid)
    a2s, sds, sid = {}, {}, 0
    for s in rigid:
        sds[sid] = s
        for a in s:
            a2s[a] = sid
        sid += 1
    for comp in _connected_carbons(mol, carbons - set(a2s)):
        sds[sid] = comp
        for a in comp:
            a2s[a] = sid
        sid += 1
    return sds, a2s


def subdomains(mol, core) -> Dict[int, Subdomain]:
    """Map each passenger carbon to the Subdomain it belongs to (with hop-from-core,
    kind, and descriptor)."""
    core = set(core)
    out: Dict[int, Subdomain] = {}
    for comp in _passenger_components(mol, core):
        carbons = {i for i in comp if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
        hets = {i for i in comp if mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6)}
        if not carbons:
            continue
        sds, a2s = _partition_subdomains(mol, carbons)

        adj = {s: set() for s in sds}
        for c in carbons:                                    # direct C-C adjacency
            for nb in mol.GetAtomWithIdx(c).GetNeighbors():
                if nb.GetIdx() in carbons and a2s[nb.GetIdx()] != a2s[c]:
                    adj[a2s[c]].add(a2s[nb.GetIdx()])
        for cl in _hetero_clusters(mol, hets):               # heteroatom-bridge adjacency
            touched = sorted({a2s[nb.GetIdx()] for h in cl
                              for nb in mol.GetAtomWithIdx(h).GetNeighbors()
                              if nb.GetIdx() in carbons})
            for i in range(len(touched)):
                for j in range(i + 1, len(touched)):
                    adj[touched[i]].add(touched[j]); adj[touched[j]].add(touched[i])

        root_sd = next((a2s[c] for c in sorted(carbons)
                        if any(nb.GetIdx() in core
                               for nb in mol.GetAtomWithIdx(c).GetNeighbors())), next(iter(sds)))
        hop = {root_sd: 0}
        q = deque([root_sd])
        while q:
            s = q.popleft()
            for t in adj[s]:
                if t not in hop:
                    hop[t] = hop[s] + 1
                    q.append(t)

        for sid, atoms in sds.items():
            arom = any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in atoms)
            inring = any(mol.GetAtomWithIdx(a).IsInRing() for a in atoms)
            haspi = any(b.GetBondTypeAsDouble() >= 2.0
                        for a in atoms for b in mol.GetAtomWithIdx(a).GetBonds()
                        if b.GetOtherAtomIdx(a) in atoms)
            kind = ("aromatic_ring" if arom else "ring" if inring
                    else "conjugated" if haspi else "linker")
            hetero = any(nb.GetAtomicNum() not in (1, 6) and nb.GetIdx() not in core
                         for a in atoms for nb in mol.GetAtomWithIdx(a).GetNeighbors())
            sd = Subdomain(hop=hop.get(sid, -1), kind=kind, aromatic=arom,
                           in_ring=inring, hydrophobic=not hetero)
            for a in atoms:
                out[a] = sd
    return out
