"""Hydrogenated-reference representation (design locked 2026).

The reference is the *saturated carbon-only graph* of a substrate:

    H(S) = keep carbon atoms and C-C connectivity, delete every heteroatom and its
           incident bonds, make all remaining bonds single, fill carbon valence with H.

    benzene -> cyclohexane      pyridine -> pentane      anisole -> cyclohexane + methane

H(S) is a real (possibly fragmented) graph, disconnected fragments are exactly what
the carbon-only definition implies (anisole's methyl detaches when its ether O is
deleted).  Everything erased is recorded as an *addressed transformation* relative to
retained carbon loci, so the pair carries no less information than S:

    S = H(S) + { addressed electronic / topological transformations }

The transformations are of two separable kinds (design decision 3):

  * heteroatom transformations, element, charge, H-count, and the carbon/heteroatom
    loci the heteroatom connects (with bond orders).  A heteroatom feature is never
    just "an O with two lone pairs"; it also says *which loci it joins* (decision 1),
    which is what lets the reference be reconstructed and what distinguishes, e.g., a
    glycosidic O by its linkage position;
  * carbon-framework transformations, bond-order / oxidation-state changes on the
    retained carbon skeleton (a C=C or the sp2 carbon of a carbonyl), kept separable
    from the heteroatom state because C-OH and C=O differ in the carbon even though
    both involve oxygen.

Reconstruction invariant (the correctness test):

    reconstruct(build_reference(S))  ==  S      (constitutionally + electronically;
                                                 stereochemistry deferred, decision 5).

Stereochemistry is not destroyed, it is retained in the mapping for a later decision
about which stereo distinctions become features, but it is not part of this round-trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

_SIDX = "sidx"          # atom property: original substrate atom index, survives edits


@dataclass(frozen=True)
class HeteroTransformation:
    """A deleted heteroatom, addressed by the loci it connected (plan decision 1)."""
    het_id: int                         # local id within this reference
    atomic_num: int
    formal_charge: int
    num_hs: int
    is_aromatic: bool
    # connections: (kind, target, bond_order) where kind is 'C' (target = carbon S-idx)
    # or 'H' (target = another het_id); bond_order is a float (1, 1.5, 2, 3).
    connections: tuple = ()


@dataclass(frozen=True)
class CarbonBondTransformation:
    """A retained C-C bond whose order in S is not the reference single bond."""
    sidx_a: int                         # substrate atom indices of the two carbons
    sidx_b: int
    order: float                        # 2, 3, or 1.5 (aromatic, pre-kekulization)


@dataclass
class HydrogenatedReference:
    """H(S) as an RDKit graph plus the transformations that rebuild S."""
    reference_mol: Chem.Mol             # carbon-only saturated graph (may be fragmented)
    carbon_sidx: tuple                  # reference atom idx -> substrate atom idx
    hetero: tuple                       # tuple[HeteroTransformation]
    carbon_bonds: tuple                 # tuple[CarbonBondTransformation]
    # stereo parked here for a later feature decision (decision 5), not used in round-trip
    stereo: dict = field(default_factory=dict)

    @property
    def reference_smiles(self) -> str:
        return Chem.MolToSmiles(self.reference_mol)


def _bond_order(bt: Chem.BondType) -> float:
    return {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0,
            Chem.BondType.TRIPLE: 3.0, Chem.BondType.AROMATIC: 1.5}.get(bt, 1.0)


def _order_to_bondtype(order: float) -> Chem.BondType:
    return {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE,
            3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}[order]


def build_reference(mol: Chem.Mol) -> HydrogenatedReference:
    """Construct H(S) and the transformation record for substrate S."""
    work = Chem.Mol(mol)
    Chem.Kekulize(work, clearAromaticFlags=True)   # encode aromaticity as explicit doubles
    for atom in work.GetAtoms():
        atom.SetIntProp(_SIDX, atom.GetIdx())

    # -- record transformations from the kekulized substrate ------------------
    het_by_sidx = {a.GetIdx(): i for i, a in enumerate(work.GetAtoms())
                   if a.GetAtomicNum() != 6}
    hetero = []
    for sidx, het_id in het_by_sidx.items():
        a = work.GetAtomWithIdx(sidx)
        conns = []
        for b in a.GetBonds():
            other = b.GetOtherAtom(a)
            oi = other.GetIdx()
            order = _bond_order(b.GetBondType())
            if oi in het_by_sidx:
                if het_by_sidx[oi] > het_id:            # record each het-het bond once
                    conns.append(("H", het_by_sidx[oi], order))
            else:
                conns.append(("C", oi, order))
        hetero.append(HeteroTransformation(
            het_id=het_id, atomic_num=a.GetAtomicNum(),
            formal_charge=a.GetFormalCharge(), num_hs=a.GetTotalNumHs(),
            is_aromatic=a.GetIsAromatic(), connections=tuple(conns)))

    carbon_bonds = []
    for b in work.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6:
            order = _bond_order(b.GetBondType())
            if order != 1.0:
                carbon_bonds.append(CarbonBondTransformation(
                    a1.GetIdx(), a2.GetIdx(), order))

    # -- build H(S): delete heteroatoms, single-bond the carbon skeleton ------
    rw = Chem.RWMol(work)
    for sidx in sorted(het_by_sidx, reverse=True):
        rw.RemoveAtom(sidx)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
        a.SetFormalCharge(0)
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    ref = rw.GetMol()
    Chem.SanitizeMol(ref)
    carbon_sidx = tuple(a.GetIntProp(_SIDX) for a in ref.GetAtoms())

    return HydrogenatedReference(
        reference_mol=ref, carbon_sidx=carbon_sidx,
        hetero=tuple(hetero), carbon_bonds=tuple(carbon_bonds))


def reconstruct(ref: HydrogenatedReference) -> Chem.Mol:
    """Rebuild S from H(S) + transformations, WITHOUT reference to the original.

    Proves the decomposition is lossless at the constitutional/electronic level."""
    rw = Chem.RWMol()
    sidx_to_new: dict[int, int] = {}
    # carbons of the reference skeleton
    for sidx in ref.carbon_sidx:
        sidx_to_new[sidx] = rw.AddAtom(Chem.Atom(6))
    # carbon-carbon skeleton (single) from the reference graph
    for b in ref.reference_mol.GetBonds():
        sa = ref.carbon_sidx[b.GetBeginAtomIdx()]
        sb = ref.carbon_sidx[b.GetEndAtomIdx()]
        rw.AddBond(sidx_to_new[sa], sidx_to_new[sb], Chem.BondType.SINGLE)
    # restore non-single carbon-carbon orders
    for cb in ref.carbon_bonds:
        rw.GetBondBetweenAtoms(sidx_to_new[cb.sidx_a], sidx_to_new[cb.sidx_b]) \
          .SetBondType(_order_to_bondtype(cb.order))
    # re-insert heteroatoms with their electronic state
    het_to_new: dict[int, int] = {}
    for h in ref.hetero:
        atom = Chem.Atom(h.atomic_num)
        atom.SetFormalCharge(h.formal_charge)
        atom.SetNumExplicitHs(h.num_hs)
        atom.SetNoImplicit(True)
        het_to_new[h.het_id] = rw.AddAtom(atom)
    # heteroatom connectivity (to carbon loci and to other heteroatoms)
    for h in ref.hetero:
        for kind, target, order in h.connections:
            dst = sidx_to_new[target] if kind == "C" else het_to_new[target]
            rw.AddBond(het_to_new[h.het_id], dst, _order_to_bondtype(order))
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _canonical(mol: Chem.Mol) -> str:
    """Stereo-stripped canonical SMILES for the reconstruction round-trip test."""
    m = Chem.Mol(mol)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(Chem.RemoveHs(m))


def verify_reconstruction(mol: Chem.Mol) -> tuple[bool, str, str]:
    """Return (ok, original_canonical, reconstructed_canonical)."""
    ref = build_reference(mol)
    rebuilt = reconstruct(ref)
    a, b = _canonical(mol), _canonical(rebuilt)
    return a == b, a, b
