"""
Mechanistic tree construction.

Builds one or more MechanisticTree objects from a bag of mapped reactions for a
single enzyme. Each input reaction is first decomposed into its 1:1 partial
reactions (one substrate ↔ one product). Partials are then assigned to A-trees
in two phases:

Phase 1, bipartite assignment:
  Reactions are processed one at a time. For each reaction's partial set, every
  partial is scored against every existing A-tree slot by the deepest level
  (E > D > C > B > A) at which the partial's matched operator agrees with any
  operator already in that slot. A max-weight greedy matching assigns each
  partial to its best available slot (constraint: at most one partial per reaction
  per slot, so partials from the same full reaction always land in distinct trees).
  Unmatched partials open new slots.

Phase 2, Am-collapse:
  After all reactions are assigned, pairs of slots whose all-carbon Am keys agree
  are merged provided no single reaction has partials in both (which would violate
  the distinctness invariant). This collapses e.g. Br- and I-leaving variants of
  the same substrate partial into one A-tree while leaving I:5 and C:4 partials
  of the same methylation reaction in separate trees.

Within each slot the first partial added is the atom-map reference; all later
partials are remapped to its atom map numbers via graph isomorphism on the
A-level operator.

The index on each root node is built by the generator and passed directly to
the MechanisticTree constructor, the model itself has no construction logic.
"""

from __future__ import annotations
import logging
import re
from collections import defaultdict
from dataclasses import replace as dc_replace

import networkx as nx
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionToSmiles
from CGRtools import smiles as cgr_smiles

from evodex.operators import extract_operator_by_abstraction
from evodex.splitting import split_reaction
from esorex.mechanistic_tree import MechanisticTree, LEVELS

# ── SMARTS → SMILES conversion for CGRtools ───────────────────────────────────

_ATOMIC_NUM_TO_SYMBOL = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
}

def _smarts_to_smiles(smarts: str) -> str:
    """Replace [#N:map] SMARTS atom notation with [Symbol:map] for CGRtools.

    Also strips SMARTS-only qualifiers (&Hn hydrogen count, &[+-] charge notation)
    that CGRtools cannot parse.  The resulting string retains element identity and
    charge but drops hydrogen-count constraints, which is sufficient for the
    canonical grouping key.
    """
    def replace_atom(m):
        num = int(m.group(1))
        rest = m.group(2)
        symbol = _ATOMIC_NUM_TO_SYMBOL.get(num, f"#{num}")
        return f"[{symbol}{rest}]"
    result = re.sub(r'\[#(\d+)([^\]]*)\]', replace_atom, smarts)
    result = re.sub(r'&H\d*', '', result)       # drop &H, &H1, &H2, … hydrogen counts
    result = re.sub(r'&([+-]\d*)', r'\1', result)  # normalize &- / &+ charge notation
    return result


# ── Canonical key for grouping equivalent operators ───────────────────────────

def _canonical_key(operator_smarts: str) -> str:
    """Return a canonical string for an operator (B–E levels), independent of atom map numbers."""
    converted = _smarts_to_smiles(operator_smarts)
    parsed = cgr_smiles(converted)
    return str(parsed)


def _a_canonical_key(b_operator_smarts: str) -> str:
    """Derive an A-level canonical key from a B-level operator.

    CGRtools cannot parse [*:N] wildcard notation, so we derive A from B by
    replacing every atom with carbon before canonicalizing. Two B operators
    with the same bond-change topology but different atom types will reduce
    to the same all-carbon operator and thus the same canonical string.
    """
    converted = _smarts_to_smiles(b_operator_smarts)
    all_carbon = re.sub(r'\[([A-Z][a-z]?)([^\]]*)\]', lambda m: f'[C{m.group(2)}]', converted)
    parsed = cgr_smiles(all_carbon)
    return str(parsed)


def _bm_central_atomic_num(b_operator_smarts: str) -> int:
    """Return the atomic number of the mapped (central) atom in a Bm operator.

    Used as a tiebreaker when two slots share the same Am key: the partial is
    assigned to the slot whose central atom is closest by atomic number, so e.g.
    a thiol (S, 16) prefers the alcohol-oxygen slot (O, 8; diff 8) over the
    iodide slot (I, 53; diff 37).
    """
    m = re.search(r'\[#(\d+):\d+\]', b_operator_smarts)
    return int(m.group(1)) if m else 6  # default to carbon if pattern absent


# ── Atom map correspondence via graph isomorphism ─────────────────────────────

def _build_op_graph(operator_smarts: str) -> nx.Graph:
    """Build a labeled graph from the reactant side of an operator.
    Nodes are atom map numbers; edges carry bond type."""
    rxn = AllChem.ReactionFromSmarts(operator_smarts)
    G = nx.Graph()
    for mol in rxn.GetReactants():
        for atom in mol.GetAtoms():
            m = atom.GetAtomMapNum()
            if m:
                G.add_node(m, atomic_num=atom.GetAtomicNum())
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetAtomMapNum()
            a2 = bond.GetEndAtom().GetAtomMapNum()
            if a1 and a2:
                G.add_edge(a1, a2, bond_type=str(bond.GetBondTypeAsDouble()))
    return G


def _find_correspondence(ref_smarts: str, other_smarts: str) -> dict[int, int]:
    """Return {other_atom_map → ref_atom_map} via graph isomorphism on operators.

    At A-level all atoms are wildcards (atomic_num=0), so node matching succeeds
    for topologically equivalent operators regardless of element identity (e.g.
    O-ester vs thioester share the same A-root).  Returns {} if no isomorphism
    exists; callers should warn rather than silently continue with stale maps.
    """
    ref_G = _build_op_graph(ref_smarts)
    other_G = _build_op_graph(other_smarts)
    gm = nx.algorithms.isomorphism.GraphMatcher(
        ref_G, other_G,
        node_match=lambda n1, n2: n1['atomic_num'] == n2['atomic_num'],
        edge_match=lambda e1, e2: e1['bond_type'] == e2['bond_type'],
    )
    for iso in gm.isomorphisms_iter():
        return {v: k for k, v in iso.items()}  # other_map → ref_map
    return {}


# ── Apply atom map remapping to operator SMARTS ───────────────────────────────

def _apply_remap(operator_smarts: str, remap: dict[int, int]) -> str:
    """Replace :N atom map numbers inside [...] brackets per remap dict."""
    def replace_bracket(m):
        interior = m.group(1)
        new_interior = re.sub(
            r':(\d+)',
            lambda x: f':{remap.get(int(x.group(1)), int(x.group(1)))}',
            interior,
        )
        return f'[{new_interior}]'
    return re.sub(r'\[([^\]]*)\]', replace_bracket, operator_smarts)


# ── Complete operators and fragments from partial reactions ───────────────────

def _one_to_one_partials(smirks: str) -> list[str]:
    """Return only 1:1 partials where both sides have at least one mapped atom."""
    try:
        partials = split_reaction(smirks)
    except Exception:
        return []
    result = []
    for p in partials:
        lhs, rhs = p.split('>>')
        if '.' in lhs or '.' in rhs:
            continue
        rxn = AllChem.ReactionFromSmarts(p)
        if not rxn:
            continue
        # Both sides must have at least one atom-mapped atom
        lhs_mapped = any(a.GetAtomMapNum() > 0
                         for mol in rxn.GetReactants() for a in mol.GetAtoms())
        rhs_mapped = any(a.GetAtomMapNum() > 0
                         for mol in rxn.GetProducts() for a in mol.GetAtoms())
        if lhs_mapped and rhs_mapped:
            result.append(p)
    return result


def _compute_complete_operators(group: list[dict], level: str) -> list[str]:
    """Deduplicated bag of complete operator SMIRKS at this level.

    The group contains 1:1 partial reactions, so operators are extracted
    directly from each partial's SMIRKS.
    """
    seen: set[str] = set()
    result: list[str] = []
    for data in group:
        try:
            op = extract_operator_by_abstraction(data['smirks'], level, matched=False)
        except Exception:
            continue
        if op and op != '>>' and re.search(r':\d+', op) and op not in seen:
            seen.add(op)
            result.append(op)
    return result


def _operator_map_numbers(operator_smarts: str) -> frozenset[int]:
    """Return the set of atom map numbers present in an operator SMARTS."""
    return frozenset(int(m) for m in re.findall(r':(\d+)', operator_smarts))


def _collect_operators_by_level(tree) -> dict:
    """Collect a mechanistic tree's complete operators grouped by EVODEX level.

    Walks the tree and returns {level: [operator_smirks, ...]}.  Lives in the mechanistic
    layer because it is EVODEX/operator plumbing used to enumerate candidate reaction
    sites, independent of any specificity model."""
    ops: dict = {}

    def _walk(node):
        if node.complete_operators:
            ops.setdefault(node.level, []).extend(node.complete_operators)
        for child in getattr(node, "children", []):
            _walk(child)

    _walk(tree)
    return ops


def _compute_fragments(group: list[dict], a_op_smarts: str) -> list[dict]:
    """For each training reaction, identify the passenger atoms, reactant atoms
    whose map numbers are not covered by the A-level operator.

    Each entry: {'mol': Chem.Mol, 'operator_indices': list[int], 'passenger_indices': list[int]}
    Only molecules that have both operator-covered atoms AND passenger atoms are included,
    since pure-operator molecules (e.g. water) have no passenger fragment to model.
    """
    op_maps = _operator_map_numbers(a_op_smarts)
    result: list[dict] = []
    for data in group:
        for mol in data['rxn'].GetReactants():
            op_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() in op_maps]
            passenger_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() not in op_maps]
            if op_indices and passenger_indices:
                result.append({
                    'mol': mol,
                    'operator_indices': op_indices,
                    'passenger_indices': passenger_indices,
                })
    return result


# ── Tree construction ─────────────────────────────────────────────────────────

def _build_node(
    level: str,
    group: list[dict],
    remapped_ops: list[dict[str, str]],
) -> MechanisticTree:
    """Recursively build one MechanisticTree node and its children.

    group: list of reaction data dicts (keys: rxn, smirks, ops)
    remapped_ops: per-reaction operator dicts with atom maps aligned to reference
    """
    level_idx = LEVELS.index(level)
    op_smarts = remapped_ops[0][level]

    complete_ops = _compute_complete_operators(group, level)
    fragments = _compute_fragments(group, op_smarts) if level == 'A' else None

    children = []
    if level_idx < len(LEVELS) - 1:
        next_level = LEVELS[level_idx + 1]

        sub_groups: dict[str, list[int]] = defaultdict(list)
        for i, ops in enumerate(remapped_ops):
            key = _canonical_key(ops[next_level])
            if key is not None:
                sub_groups[key].append(i)

        for indices in sub_groups.values():
            child_group = [group[i] for i in indices]
            child_ops = [remapped_ops[i] for i in indices]
            children.append(_build_node(next_level, child_group, child_ops))

    return MechanisticTree(
        smirks=op_smarts,
        operator=AllChem.ReactionFromSmarts(op_smarts),
        level=level,
        reactions=[d['rxn'] for d in group],
        children=children,
        complete_operators=complete_ops,
        fragments=fragments,
        index=None,
    )


def _build_index(root: MechanisticTree) -> dict[str, list[MechanisticTree]]:
    """Walk the subtree and collect all nodes by level."""
    idx: dict[str, list[MechanisticTree]] = {level: [] for level in LEVELS}
    def walk(node: MechanisticTree) -> None:
        idx[node.level].append(node)
        for child in node.children:
            walk(child)
    walk(root)
    return idx


# ── A-tree slot: accumulates partials with a shared transformation ────────────

class _ATreeSlot:
    """Accumulates 1:1 partials that share an A-level transformation.

    The first partial added becomes the atom-map reference; all later partials
    are remapped to its atom map numbers via graph isomorphism on the A-operator.
    """

    def __init__(self, partial: dict, rxn_id: int) -> None:
        self.ref: dict = partial
        self.members: list[tuple[dict, int]] = [(partial, rxn_id)]
        self.remapped_ops: list[dict[str, str]] = [dict(partial['ops'])]
        self.source_rxn_ids: set[int] = {rxn_id}
        self.am_key: str = _a_canonical_key(partial['ops']['B'])
        # Atomic number of the mapped (central) atom in the reference Bm operator.
        # Used as a tiebreaker when two slots share the same Am key.
        self._ref_bm_central: int = _bm_central_atomic_num(partial['ops']['B'])
        # Canonical keys at each level for match scoring (element-sensitive at B–E,
        # all-carbon at A).  Map-number-independent so remapped and original agree.
        self._level_keys: dict[str, set[str]] = {
            level: {_canonical_key(partial['ops'][level])} for level in LEVELS
        }

    def match_score(self, partial: dict) -> float:
        """Return score for how well partial fits this slot.

        Primary score: 5(E) … 2(B), exact Bm-level match at the given level.
        When only Am topology matches (score 1), a fractional tiebreaker is added:
        1 + 1/(1 + |Δatomic_num|) / 10, so that a thiol (S=16) prefers an
        alcohol slot (O=8, Δ=8) over an iodide slot (I=53, Δ=37).  The secondary
        term is always < 0.1, so it never overrides a real B-level match.
        """
        for primary, level in [(5, 'E'), (4, 'D'), (3, 'C'), (2, 'B')]:
            k = _canonical_key(partial['ops'][level])
            if k in self._level_keys[level]:
                return float(primary)
        if _a_canonical_key(partial['ops']['B']) == self.am_key:
            diff = abs(_bm_central_atomic_num(partial['ops']['B']) - self._ref_bm_central)
            return 1.0 + (1.0 / (1.0 + diff)) / 10.0
        return 0.0

    def add(self, partial: dict, rxn_id: int) -> None:
        remap = _find_correspondence(self.ref['ops']['A'], partial['ops']['A'])
        if not remap and self.ref['ops']['A'] != partial['ops']['A']:
            logging.warning(
                f"Atom-map remapping failed between A-operators "
                f"{self.ref['ops']['A']!r} and {partial['ops']['A']!r}; "
                f"tree structure may be incorrect for this partial."
            )
        remapped = {level: _apply_remap(partial['ops'][level], remap) for level in LEVELS}
        self.members.append((partial, rxn_id))
        self.remapped_ops.append(remapped)
        self.source_rxn_ids.add(rxn_id)
        for level in LEVELS:
            self._level_keys[level].add(_canonical_key(remapped[level]))

    @property
    def group(self) -> list[dict]:
        return [p for p, _ in self.members]


# ── Bipartite assignment of a partial set to existing slots ──────────────────

def _assign_partials(
    partial_set: list[dict],
    slots: list[_ATreeSlot],
    rxn_id: int,
) -> dict[int, int | None]:
    """Return {partial_idx: slot_idx or None}, best injective assignment.

    Scores each (partial, slot) pair at the deepest level they agree (E=5 …
    Am=1; blocked by same-reaction provenance → -1).  A greedy max-weight
    pass assigns each partial to its best available slot; unmatched partials
    receive None (caller opens a new slot for them).
    """
    n, m = len(partial_set), len(slots)
    scores = [
        [
            -1 if rxn_id in slots[j].source_rxn_ids else slots[j].match_score(partial_set[i])
            for j in range(m)
        ]
        for i in range(n)
    ]

    # Sort candidate (partial, slot) pairs descending by score, skip blocked pairs
    candidates = sorted(
        ((scores[i][j], i, j) for i in range(n) for j in range(m) if scores[i][j] > 0),
        reverse=True,
    )

    assignment: dict[int, int | None] = {i: None for i in range(n)}
    used_partials: set[int] = set()
    used_slots: set[int] = set()
    for _, i, j in candidates:
        if i not in used_partials and j not in used_slots:
            assignment[i] = j
            used_partials.add(i)
            used_slots.add(j)

    return assignment


# ── Public API ────────────────────────────────────────────────────────────────

def generate_mechanistic_tree(reactions: list[ChemicalReaction]) -> list[MechanisticTree]:
    """Build mechanistic trees from a bag of positive reactions for one enzyme.

    Returns one MechanisticTree root per distinct A-level operator.
    Each root's index maps 'A'–'E' to all nodes at that level in its subtree.
    See module docstring for the two-phase algorithm.
    """
    # ── Step 1: parse each reaction into its 1:1 partial set ─────────────────
    reaction_partial_sets: list[list[dict]] = []
    for rxn in reactions:
        # ReactionToSmiles avoids [C&H3:n] SMARTS notation that evodex rejects.
        smirks = ReactionToSmiles(rxn)
        partials = _one_to_one_partials(smirks)
        if not partials:
            lhs, _, rhs = smirks.partition('>>')
            if '.' not in lhs and '.' not in rhs:
                partials = [smirks]
            else:
                logging.warning(
                    f"Could not split multi-reactant reaction into 1:1 partials, "
                    f"skipping: {smirks[:80]}"
                )
                continue
        partial_set: list[dict] = []
        for p_smirks in partials:
            try:
                ops = {
                    level: extract_operator_by_abstraction(p_smirks, level, matched=True)
                    for level in LEVELS
                }
            except Exception as e:
                logging.error(f"Failed to process partial: {p_smirks}, Error: {e}")
                continue
            if not re.search(r':\d+', ops.get('A', '')):
                continue  # degenerate, no mapped atoms at A
            p_rxn = AllChem.ReactionFromSmarts(p_smirks)
            if p_rxn is None:
                continue
            partial_set.append({'rxn': p_rxn, 'smirks': p_smirks, 'ops': ops})
        if partial_set:
            reaction_partial_sets.append(partial_set)

    # ── Step 2: assign each reaction's partials to A-tree slots ──────────────
    # Invariant: each slot holds at most one partial from any single reaction,
    # so partials from the same full reaction always occupy distinct trees.
    slots: list[_ATreeSlot] = []
    for rxn_id, partial_set in enumerate(reaction_partial_sets):
        assignment = _assign_partials(partial_set, slots, rxn_id)
        for i, j in assignment.items():
            if j is None:
                slots.append(_ATreeSlot(partial_set[i], rxn_id))
            else:
                slots[j].add(partial_set[i], rxn_id)

    # ── Step 3: Am-collapse ───────────────────────────────────────────────────
    # Merge pairs of slots whose all-carbon Am keys agree, provided no single
    # reaction contributed partials to both (which would violate the invariant).
    changed = True
    while changed:
        changed = False
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                si, sj = slots[i], slots[j]
                if si.am_key == sj.am_key and not (si.source_rxn_ids & sj.source_rxn_ids):
                    for partial, rxn_id in sj.members:
                        si.add(partial, rxn_id)
                    slots.pop(j)
                    changed = True
                    break
            if changed:
                break

    # ── Step 4: build MechanisticTree objects ─────────────────────────────────
    roots = []
    for slot in slots:
        root = _build_node('A', slot.group, slot.remapped_ops)
        root = dc_replace(root, index=_build_index(root))
        roots.append(root)
    return roots


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Three reactions to exercise the tree builder:
    #   rxn1: alkyl ester hydrolysis   (CH3-CO-O-CH2CH3 + H2O)
    #   rxn2: aryl ester hydrolysis    (CH3-CO-O-Ph + H2O), same A as rxn1
    #   rxn3: phosphate hydrolysis     (CH3-O-PO3H2 + H2O), possibly different A
    #
    # Expected: esterases share an A operator; diverge at D/E (alkyl vs aryl).
    # Phosphatase may share A (same bond-change topology) or not, print reveals.
    from esorex.reaction_preparation import prepare_reaction

    test_reaction_smiles = [
        # Alkyl ester hydrolysis: ethyl acetate + H2O → acetic acid + ethanol
        # CH3(1,14-16)-C(2)(=O3)-O4-CH2(5,17-18)-CH3(6,19-21) + O7(H8)(H9)
        # → CH3-C(=O)-O7(H8)  +  O4(H9)-CH2-CH3
        "[C:1]([H:14])([H:15])([H:16])-[C:2](=[O:3])-[O:4]-[C:5]([H:17])([H:18])-[C:6]([H:19])([H:20])([H:21])"
        ".[O:7]([H:8])([H:9])"
        ">>[C:1]([H:14])([H:15])([H:16])-[C:2](=[O:3])-[O:7]([H:8])"
        ".[O:4]([H:9])-[C:5]([H:17])([H:18])-[C:6]([H:19])([H:20])([H:21])",

        # Aryl ester hydrolysis: phenyl acetate + H2O → acetic acid + phenol
        # CH3(1,14-16)-C(2)(=O3)-O4-Ph(5,9-13,22-26) + O7(H8)(H27)
        # → CH3-C(=O)-O7(H8)  +  O4(H27)-Ph
        # Note: water H uses :27 (not :9) to avoid conflict with ring carbon :9
        "[C:1]([H:14])([H:15])([H:16])-[C:2](=[O:3])-[O:4]-[c:5]1:[c:9]([H:22]):[c:10]([H:23]):[c:11]([H:24]):[c:12]([H:25]):[c:13]([H:26]):1"
        ".[O:7]([H:8])([H:27])"
        ">>[C:1]([H:14])([H:15])([H:16])-[C:2](=[O:3])-[O:7]([H:8])"
        ".[O:4]([H:27])-[c:5]1:[c:9]([H:22]):[c:10]([H:23]):[c:11]([H:24]):[c:12]([H:25]):[c:13]([H:26]):1",

        # Phosphate monoester hydrolysis: methyl phosphate + H2O → methanol + phosphoric acid
        # CH3(1,10-12)-O2-P3(=O4)(-O5H13)(-O6H14) + O7(H8)(H9)
        # → CH3-O2-H9  +  H8-O7-P3(=O4)(-O5H13)(-O6H14)
        # O2 (bridging) stays with carbon; O7 (water) attacks phosphorus
        "[C:1]([H:10])([H:11])([H:12])-[O:2]-[P:3](=[O:4])(-[O:5][H:13])-[O:6][H:14]"
        ".[O:7]([H:8])([H:9])"
        ">>[C:1]([H:10])([H:11])([H:12])-[O:2]([H:9])"
        ".[H:8]-[O:7]-[P:3](=[O:4])(-[O:5][H:13])-[O:6][H:14]",
    ]

    reactions = [prepare_reaction(s) for s in test_reaction_smiles]

    trees = generate_mechanistic_tree(reactions)

    print(f"Mechanistic trees: {len(trees)}\n")
    for i, tree in enumerate(trees):
        print(f"── Tree {i} ──────────────────────────────────────────────────────")
        print(f"  A smirks:  {tree.smirks}")
        print(f"  reactions: {len(tree.reactions)}")
        for level in LEVELS:
            nodes = tree.index[level]
            if nodes:
                print(f"  {level}: {len(nodes)} operator(s)")
                for node in nodes:
                    print(f"    [{len(node.reactions)} rxn(s)] {node.smirks}")
        print(f"  complete_operators ({len(tree.complete_operators)}):")
        for op in tree.complete_operators:
            print(f"    {op}")
        n_frags = len(tree.fragments) if tree.fragments else 0
        print(f"  fragments: {n_frags} passenger entries")
        for frag in (tree.fragments or []):
            pax = [frag['mol'].GetAtomWithIdx(i).GetAtomMapNum() for i in frag['passenger_indices']]
            print(f"    passenger map nums: {pax}")
        print()
