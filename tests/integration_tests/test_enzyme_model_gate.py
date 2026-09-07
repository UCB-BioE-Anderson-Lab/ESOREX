"""The feasibility gate, and the two silent failures that made it necessary.

EnzymeModel.predict is two stages: the complete operators at the requested level decide
whether the enzyme's reaction applies at all, and only a molecule that passes reaches the
specificity model.  No match is rate 0, not a small rate.
"""
import pytest
from rdkit import Chem

from esorex.energetic_specificity import EnergeticSpecificityModel
from esorex.enzyme_model import EnzymeModel
from esorex.generate_mechanistic_tree import generate_mechanistic_tree
from esorex.reaction_preparation import prepare_reaction

# Three TyrB substrates in the mapped form the pipeline ingests: the amine leaves as
# ammonia and water supplies the ketone oxygen, so every heavy atom is paired.
REACTIONS = {
    "Phenylalanine":
        "[N:1][C@@H:2]([C:10][c:11]1[c:12][c:13][c:14][c:15][c:16]1)[C:3](=[O:4])[OH:5]"
        ".[OH2:6]>>[N:1].[C:2](=[O:6])([C:10][c:11]1[c:12][c:13][c:14][c:15][c:16]1)"
        "[C:3](=[O:4])[OH:5]",
    "Alanine":
        "[N:1][C@@H:2]([C:10])[C:3](=[O:4])[OH:5].[OH2:6]>>"
        "[N:1].[C:2](=[O:6])([C:10])[C:3](=[O:4])[OH:5]",
    "Tyrosine":
        "[N:1][C@@H:2]([C:10][c:11]1[c:12][c:13][c:14]([O:17])[c:15][c:16]1)"
        "[C:3](=[O:4])[OH:5].[OH2:6]>>[N:1].[C:2](=[O:6])([C:10][c:11]1[c:12][c:13]"
        "[c:14]([O:17])[c:15][c:16]1)[C:3](=[O:4])[OH:5]",
}
SUBSTRATES = {
    "Phenylalanine": "N[C@@H](Cc1ccccc1)C(=O)O",
    "Alanine": "C[C@H](N)C(=O)O",
    "Tyrosine": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
}
RATES = {"Phenylalanine": 1.6e6, "Alanine": 2.0, "Tyrosine": 3.3e5}

# one edit from phenylalanine, and none of them can be transaminated
NON_SUBSTRATES = {
    "phenylacetate":      "OC(=O)Cc1ccccc1",        # no amine
    "3-phenylpropionate": "OC(=O)CCc1ccccc1",       # amine deleted
    "phenyllactate":      "OC(=O)C(O)Cc1ccccc1",    # amine -> hydroxyl
    "N-methyl-Phe":       "CNC(Cc1ccccc1)C(=O)O",   # amine is secondary
    "alpha-methyl-Phe":   "CC(N)(Cc1ccccc1)C(=O)O", # no alpha-H
    "beta-Phe":           "OC(=O)CC(N)c1ccccc1",    # amine on the wrong carbon
}
NAMES = list(REACTIONS)


@pytest.fixture(scope="module")
def model():
    m = EnzymeModel()
    m.train([prepare_reaction(REACTIONS[n]) for n in NAMES],
            rates=[RATES[n] for n in NAMES])
    return m


# ── the gate ────────────────────────────────────────────────────────────────────
def test_substrates_are_feasible_and_priced(model):
    for name in NAMES:
        p = model.predict(Chem.MolFromSmiles(SUBSTRATES[name]))
        assert p.feasible is True
        assert p.rate > 0
        assert p.core


@pytest.mark.parametrize("name,smiles", sorted(NON_SUBSTRATES.items()))
def test_non_substrates_are_zero_not_slow(model, name, smiles):
    p = model.predict(Chem.MolFromSmiles(smiles))
    assert p.feasible is False
    assert p.rate == 0.0
    assert p.energy is None and p.determined is None and p.core is None


def test_gate_holds_at_every_level(model):
    for level in "BCDE":
        assert model.predict(Chem.MolFromSmiles("CNC(Cc1ccccc1)C(=O)O"),
                             level=level).rate == 0.0
        assert model.predict(Chem.MolFromSmiles(SUBSTRATES["Phenylalanine"]),
                             level=level).feasible is True


def test_unknown_level_is_rejected(model):
    with pytest.raises(ValueError, match="unknown abstraction level"):
        model.predict(Chem.MolFromSmiles(SUBSTRATES["Alanine"]), level="Z")


# ── training inputs must survive tree construction intact ───────────────────────
def test_aromaticity_survives_tree_construction():
    """Fragments train the model and queries are built from SMILES, so a ring has to look
    the same on both sides.  split_reaction returns SMARTS, whose parse has aromatic bonds
    but no aromatic atoms; unrepaired, phenylalanine trains as though it had no ring."""
    trees = generate_mechanistic_tree([prepare_reaction(REACTIONS["Phenylalanine"])])
    checked = 0
    for tree in trees:
        for fragment in tree.fragments or []:
            mol = fragment["mol"]
            if any(b.GetBondType() == Chem.BondType.AROMATIC for b in mol.GetBonds()):
                assert any(a.GetIsAromatic() for a in mol.GetAtoms())
                Chem.CanonicalRankAtoms(mol, breakTies=True)  # unusable mols raise
                checked += 1
    assert checked, "no aromatic fragment was produced, so nothing was verified"


def test_fragments_name_their_source_reaction():
    """Rates are paired to fragments through this index, so it must be present and cover
    the reactions one-to-one."""
    reactions = [prepare_reaction(REACTIONS[n]) for n in NAMES]
    model = EnzymeModel()
    model.train(reactions, rates=[RATES[n] for n in NAMES])
    order = [f["reaction_index"] for f in model.tree.fragments]
    assert sorted(order) == list(range(len(NAMES)))


def test_rates_follow_the_fragment_they_belong_to(monkeypatch):
    """Each fragment must get ITS OWN reaction's rate.

    Fragment order comes from slot assignment inside tree construction; today it happens
    to match the order the reactions were passed, so simply reordering the inputs does not
    discriminate.  Force the two apart: hand train() a tree whose fragments are shuffled
    relative to the reactions.  Pairing by position silently mislabels every substrate;
    pairing by reaction_index is unaffected."""
    import dataclasses
    import esorex.enzyme_model as enzyme_model

    reactions = [prepare_reaction(REACTIONS[n]) for n in NAMES]
    rates = [RATES[n] for n in NAMES]

    reference = EnzymeModel()
    reference.train(reactions, rates=rates)

    real_generate = enzyme_model.generate_mechanistic_tree

    def shuffled(rxns):
        trees = real_generate(rxns)
        out = []
        for tree in trees:
            frags = list(tree.fragments or [])
            out.append(dataclasses.replace(tree, fragments=list(reversed(frags)))
                       if len(frags) > 1 else tree)
        return out

    monkeypatch.setattr(enzyme_model, "generate_mechanistic_tree", shuffled)
    shuffled_model = EnzymeModel()
    shuffled_model.train(reactions, rates=rates)

    for name in NAMES:
        mol = Chem.MolFromSmiles(SUBSTRATES[name])
        assert shuffled_model.predict(mol).rate == pytest.approx(
            reference.predict(mol).rate, rel=1e-6), (
            f"{name} changed when the fragment list was reordered: rates are being "
            f"paired to fragments by position, not by reaction_index")


# ── stereochemistry ─────────────────────────────────────────────────────────────
def test_the_mirror_image_is_not_a_substrate(model):
    """TyrB is an L-aminotransferase, and the operators say so: EVODEX extracts the
    reacting centre's configuration, giving [#6@@:2] at every level.  RDKit ignores
    chirality in substructure matching unless asked, which admitted D-amino acids and
    priced them as their L counterparts."""
    for name, l_smiles, d_smiles in [
        ("phenylalanine", "N[C@@H](Cc1ccccc1)C(=O)O", "N[C@H](Cc1ccccc1)C(=O)O"),
        ("alanine",       "C[C@H](N)C(=O)O",          "C[C@@H](N)C(=O)O"),
    ]:
        natural = model.predict(Chem.MolFromSmiles(l_smiles))
        mirror = model.predict(Chem.MolFromSmiles(d_smiles))
        assert natural.feasible is True, f"L-{name} should be a substrate"
        assert natural.rate > 0
        assert mirror.feasible is False, f"D-{name} matched an L-only operator"
        assert mirror.rate == 0.0


def test_an_unspecified_stereocentre_is_not_assumed_reactive(model):
    """A substrate drawn without its configuration cannot be claimed as the reactive one,
    so it does not match a stereospecific operator."""
    flat = model.predict(Chem.MolFromSmiles("NC(Cc1ccccc1)C(=O)O"))
    assert flat.feasible is False
    assert flat.rate == 0.0


def test_operators_without_stereochemistry_are_unaffected():
    """Enforcing chirality must not narrow an operator that never specified it: a query
    template with no configuration still matches either enantiomer."""
    from rdkit.Chem import rdChemReactions
    from esorex.label_substrate import label_substrate
    op = rdChemReactions.ReactionFromSmarts("[C:1][O:2][H]>>[C:1][O:2]")
    for smiles in ("C[C@H](N)CO", "C[C@@H](N)CO"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert label_substrate(mol, op), f"{smiles} stopped matching an achiral operator"


# ── the specificity model alone cannot recognise a non-substrate ────────────────
def test_specificity_model_refuses_an_empty_core(model):
    """Before this guard it priced the whole molecule against a phantom centre and
    returned a number for anything at all."""
    with pytest.raises(ValueError, match="empty reactive core"):
        model.specificity.predict(Chem.MolFromSmiles("OC(=O)Cc1ccccc1"), set())
