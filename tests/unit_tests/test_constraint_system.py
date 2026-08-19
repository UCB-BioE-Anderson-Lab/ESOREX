"""Invariant tests for esorex.constraint_system.ConstraintSystem.

These are chemistry-free: every design matrix is built by hand from abstract
feature labels ("a", "b", ...) so the expected answer is known exactly. They pin
the mathematical guarantees the rest of ESOREX rests on:

  * exact interpolation      (the fit reproduces the supplied energies)
  * prediction determinacy   (qᵀN = 0  =>  every exact fit agrees on E_Q)
  * novelty                  (‖qN‖/‖q‖, zero iff determined)
  * null-space invariance    (moving w along N never changes a training energy)
  * cooperative repair       (restore_exactness reaches consistency, or refuses
                              when two identical feature vectors disagree)
  * determinism              (same inputs -> same solution, byte for byte)

Golden fixtures over real molecules are deliberately NOT here: those encode
chemistry and are frozen separately under explicit review.
"""
import numpy as np
import pytest

from esorex.constraint_system import (
    ConstraintSystem,
    energy_from_rate,
    rate_from_energy,
    RT_KCAL_298,
)


# --- rate <-> energy round trip ------------------------------------------------

def test_energy_rate_round_trip():
    k = np.array([1e-3, 0.5, 1.0, 42.0, 1234.0])
    assert np.allclose(rate_from_energy(energy_from_rate(k)), k)
    E = np.array([-2.0, -0.3, 0.0, 1.7, 5.0])
    assert np.allclose(energy_from_rate(rate_from_energy(E)), E)


def test_energy_from_rate_direction_and_scale():
    # lower energy = faster substrate; RT is the conversion scale
    assert energy_from_rate(10.0) < energy_from_rate(1.0)
    assert np.isclose(energy_from_rate(np.e), -RT_KCAL_298)


# --- exact interpolation on a fully determined system --------------------------

def test_full_rank_system_interpolates_and_is_determined():
    # 3 substrates, features a,b; columns [a, b, intercept] are independent.
    feature_sets = [{"a"}, {"b"}, {"a", "b"}]
    energies = [1.0, 2.0, 2.5]
    cs = ConstraintSystem(["a", "b"], feature_sets, energies)

    assert cs.consistent
    assert cs.null_dim == 0
    # hand solution: E0=0.5, w_a=0.5, w_b=1.5  (columns ordered a, b, intercept)
    assert np.allclose(cs.w_p, [0.5, 1.5, 0.5])

    for fs, e in zip(feature_sets, energies):
        E, determined, nov = cs.predict_energy(fs)
        assert np.isclose(E, e)
        assert determined is True
        assert np.isclose(nov, 0.0)


# --- parameter underdetermination != prediction underdetermination -----------

def _degenerate_system():
    # a and b always co-occur -> their columns are identical -> rank deficient.
    feature_sets = [{"a", "b"}, set()]
    energies = [1.0, 0.0]
    return ConstraintSystem(["a", "b"], feature_sets, energies), feature_sets, energies


def test_underdetermined_but_prediction_determined():
    cs, feature_sets, energies = _degenerate_system()
    assert cs.null_dim == 1                      # a/b weights are not individually pinned

    # a training substrate still gets a determined, exact prediction
    for fs, e in zip(feature_sets, energies):
        E, determined, nov = cs.predict_energy(fs)
        assert np.isclose(E, e)
        assert determined is True
        assert np.isclose(nov, 0.0)


def test_novel_direction_is_flagged_not_determined():
    cs, _, _ = _degenerate_system()
    # {a} alone was never seen (a and b only ever appeared together): novel direction
    E, determined, nov = cs.predict_energy({"a"})
    assert determined is False
    assert nov > 0.0


def test_null_space_leaves_training_energies_unchanged():
    cs, _, _ = _degenerate_system()
    base = cs.X @ cs.w_p
    for k in range(cs.null_dim):
        moved = cs.w_p + 3.7 * cs.null_basis[:, k]
        assert np.allclose(cs.X @ moved, base)   # prediction invariant along N


# --- cooperative repair of an additive conflict -------------------------------

def test_restore_exactness_adds_interaction_and_reaches_consistency():
    # additive model cannot fit: {} =0, {a}=1, {b}=1, {a,b}=1  (additive wants 2).
    feature_sets = [set(), {"a"}, {"b"}, {"a", "b"}]
    energies = [0.0, 1.0, 1.0, 1.0]
    cs = ConstraintSystem(["a", "b"], feature_sets, energies)
    assert not cs.consistent

    added = cs.restore_exactness()
    assert len(added) == 1
    assert cs.consistent
    assert cs.max_residual < 1e-9
    for fs, e in zip(feature_sets, energies):
        E, _, _ = cs.predict_energy(fs)
        assert np.isclose(E, e)


def test_restore_exactness_refuses_identical_feature_vectors():
    # two substrates with the SAME features but different energies: no interaction
    # of these features can separate them (a featurization-resolution limit).
    cs = ConstraintSystem(["a"], [{"a"}, {"a"}], [0.0, 1.0])
    assert not cs.consistent
    with pytest.raises(RuntimeError):
        cs.restore_exactness()


# --- determinism ---------------------------------------------------------------

def test_solution_is_deterministic():
    feature_sets = [{"a", "b"}, {"b"}, {"a"}, set()]
    energies = [2.5, 1.0, 1.5, 0.0]
    a = ConstraintSystem(["a", "b"], feature_sets, energies)
    b = ConstraintSystem(["a", "b"], feature_sets, energies)
    assert a.rank == b.rank and a.null_dim == b.null_dim
    assert np.array_equal(a.w_p, b.w_p)
    for fs in feature_sets + [{"a"}, {"b"}, set()]:
        assert a.predict_energy(fs) == b.predict_energy(fs)
