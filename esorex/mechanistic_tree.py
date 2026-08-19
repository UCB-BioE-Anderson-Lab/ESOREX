"""
MechanisticTree class.

Represents the A-rooted hierarchy of EVODEX abstractions for a single enzyme.
Each node corresponds to one operator at levels A through E. Reactions that share
the same operator collapse to the same node; where they diverge, children branch.

An enzyme may produce multiple MechanisticTree roots if reactions differ at A.

The index maps 'A'–'E' to all nodes at that level in the subtree, enabling O(1)
retrieval of all operators at a given abstraction level. It is populated by the
generator or deserializer before construction, the model does not build it.

complete_operators: flat bag of deduplicated 1:1 partial complete operator SMIRKS
at this abstraction level, derived from split_reaction on training reactions. Used
at prediction time, apply one of these to a query substrate to test for a match.

fragments: populated only on A-level root nodes. Each entry is the output dict of
label_reaction() applied to a 1:1 training partial with the matching Am complete
operator. The 'unmatched_atoms' field in each dict identifies the passenger atoms
(substrate atoms not covered by the operator), which are the training inputs for
the specificity model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from rdkit.Chem.rdChemReactions import ChemicalReaction

LEVELS = ('A', 'B', 'C', 'D', 'E')


@dataclass(frozen=True, eq=False)
class MechanisticTree:
    smirks: str                                       # operator SMIRKS (debug + serialization)
    operator: ChemicalReaction                        # RDKit object for matching
    level: str                                        # 'A' | 'B' | 'C' | 'D' | 'E'
    reactions: list[ChemicalReaction]                 # full training reactions at this node
    children: list[MechanisticTree] = field(default_factory=list)
    index: dict[str, list[MechanisticTree]] | None = None   # A-root only; level → nodes
    complete_operators: list[str] = field(default_factory=list)  # 1:1 partial complete op SMIRKS
    fragments: list[dict] | None = None               # A-root only; label_reaction output dicts

    def __repr__(self) -> str:
        n_cops = len(self.complete_operators)
        n_frags = len(self.fragments) if self.fragments is not None else 0
        return (
            f"MechanisticTree(level={self.level!r}, "
            f"reactions={len(self.reactions)}, children={len(self.children)}, "
            f"complete_operators={n_cops}, fragments={n_frags}, "
            f"smirks={self.smirks!r})"
        )
