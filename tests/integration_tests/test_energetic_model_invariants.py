"""Model-level invariant tests for esorex.energetic_specificity.EnergeticSpecificityModel.

Scope note: these assert only *solver-level* guarantees of the assembled pipeline,
exact interpolation, determinism, provenance, and the energy<->rate round trip. They
deliberately assert NOTHING about the chemistry the representation extracts (which
carbons, which subdomains, which deltas); that is covered by golden fixtures frozen
under explicit review. The FucTIII training set is used only as a convenient,
stable fixture that is known to fit exactly with no cooperative terms.
"""
import numpy as np
import pytest

from esorex.energetic_specificity import EnergeticSpecificityModel
from esorex.constraint_system import rate_from_energy, RT_KCAL_298

try:
    from experiments.model_selection.datasets import fuctiii_regioselectivity as fuctiii
    _mols, _cores, _energies, _ = fuctiii.build_energetic()
except Exception as exc:                                  # dataset unavailable
    _mols = None
    _skip_reason = f"FucTIII fixture unavailable: {exc}"


requires_fixture = pytest.mark.skipif(_mols is None, reason="FucTIII fixture unavailable")


@requires_fixture
def test_training_energies_reproduced_exactly():
    m = EnergeticSpecificityModel()
    info = m.train(_mols, _cores, energies=_energies)
    assert info["exact"] is True
    preds = np.array([m.predict(mol, core).energy for mol, core in zip(_mols, _cores)])
    assert np.max(np.abs(preds - np.array(_energies))) < 1e-9


@requires_fixture
def test_training_is_deterministic():
    a = EnergeticSpecificityModel(); a.train(_mols, _cores, energies=_energies)
    b = EnergeticSpecificityModel(); b.train(_mols, _cores, energies=_energies)
    assert np.array_equal(a._cs.w_p, b._cs.w_p)
    for mol, core in zip(_mols, _cores):
        pa, pb = a.predict(mol, core), b.predict(mol, core)
        assert (pa.energy, pa.determined, pa.novelty) == (pb.energy, pb.determined, pb.novelty)


@requires_fixture
def test_training_substrate_prediction_is_determined():
    m = EnergeticSpecificityModel(); m.train(_mols, _cores, energies=_energies)
    # a substrate whose features are all precedented must be determined
    assert m.predict(_mols[0], _cores[0]).determined is True


@requires_fixture
def test_predicted_rate_is_the_energy_inverse():
    m = EnergeticSpecificityModel(); m.train(_mols, _cores, energies=_energies)
    p = m.predict(_mols[0], _cores[0])
    assert np.isclose(p.rate, float(rate_from_energy(p.energy, RT_KCAL_298)))


@requires_fixture
def test_atom_contributions_returns_numeric_map():
    m = EnergeticSpecificityModel(); m.train(_mols, _cores, energies=_energies)
    contrib = m.atom_contributions(_mols[0], _cores[0])
    assert isinstance(contrib, dict)
    assert all(isinstance(v, float) for v in contrib.values())
