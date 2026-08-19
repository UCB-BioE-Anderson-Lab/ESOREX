"""
Unit tests for esorex.free_energy, zero-preserving log-scale rate ↔ normalized
activation free energy conversion.

Reference rate x0 = 1.0 M⁻¹s⁻¹ is used throughout unless stated otherwise.
"""

import math
import pytest
from esorex.free_energy import (
    compute_scaling_factor,
    rate_to_energy,
    energy_to_rate,
    normalize_rates,
)

X0 = 1.0  # reference rate used in most tests


# ---------------------------------------------------------------------------
# compute_scaling_factor
# ---------------------------------------------------------------------------

def test_scaling_factor_formula():
    # scaling_factor = ln(1 + xmax/x0)
    assert compute_scaling_factor(xmax=99.0, x0=1.0) == pytest.approx(math.log(100.0))

def test_scaling_factor_equals_one_when_xmax_equals_x0_times_e_minus_one():
    # ln(1 + (e-1)) = ln(e) = 1
    xmax = X0 * (math.e - 1)
    assert compute_scaling_factor(xmax, X0) == pytest.approx(1.0)

def test_scaling_factor_rejects_nonpositive_xmax():
    with pytest.raises(ValueError):
        compute_scaling_factor(xmax=0.0, x0=X0)
    with pytest.raises(ValueError):
        compute_scaling_factor(xmax=-5.0, x0=X0)

def test_scaling_factor_rejects_nonpositive_x0():
    with pytest.raises(ValueError):
        compute_scaling_factor(xmax=100.0, x0=0.0)
    with pytest.raises(ValueError):
        compute_scaling_factor(xmax=100.0, x0=-1.0)


# ---------------------------------------------------------------------------
# rate_to_energy, boundary conditions
# ---------------------------------------------------------------------------

def test_zero_rate_maps_to_zero_energy():
    sf = compute_scaling_factor(100.0, X0)
    assert rate_to_energy(0.0, X0, sf) == 0.0

def test_xmax_maps_to_unit_energy():
    xmax = 500.0
    sf = compute_scaling_factor(xmax, X0)
    assert rate_to_energy(xmax, X0, sf) == pytest.approx(1.0)

def test_energy_between_zero_and_one_for_intermediate_rate():
    xmax = 1000.0
    sf = compute_scaling_factor(xmax, X0)
    e = rate_to_energy(50.0, X0, sf)
    assert 0.0 < e < 1.0

def test_energy_strictly_monotone():
    xmax = 1000.0
    sf = compute_scaling_factor(xmax, X0)
    rates = [0.0, 1.0, 10.0, 100.0, 1000.0]
    energies = [rate_to_energy(x, X0, sf) for x in rates]
    for i in range(len(energies) - 1):
        assert energies[i] < energies[i + 1]

def test_rate_to_energy_rejects_negative_rate():
    sf = compute_scaling_factor(100.0, X0)
    with pytest.raises(ValueError):
        rate_to_energy(-1.0, X0, sf)

def test_rate_to_energy_rejects_nonpositive_x0():
    sf = compute_scaling_factor(100.0, X0)
    with pytest.raises(ValueError):
        rate_to_energy(50.0, x0=0.0, scaling_factor=sf)

def test_rate_to_energy_rejects_nonpositive_scaling_factor():
    with pytest.raises(ValueError):
        rate_to_energy(50.0, X0, scaling_factor=0.0)


# ---------------------------------------------------------------------------
# energy_to_rate, boundary and formula
# ---------------------------------------------------------------------------

def test_zero_energy_maps_to_zero_rate():
    sf = compute_scaling_factor(100.0, X0)
    assert energy_to_rate(0.0, X0, sf) == pytest.approx(0.0)

def test_unit_energy_maps_to_xmax():
    xmax = 250.0
    sf = compute_scaling_factor(xmax, X0)
    assert energy_to_rate(1.0, X0, sf) == pytest.approx(xmax)

def test_energy_to_rate_formula():
    # x = x0 * (exp(e * sf) - 1)
    xmax = 99.0
    sf = compute_scaling_factor(xmax, X0)
    e = 0.5
    expected = X0 * (math.exp(e * sf) - 1.0)
    assert energy_to_rate(e, X0, sf) == pytest.approx(expected)

def test_energy_to_rate_rejects_nonpositive_x0():
    sf = compute_scaling_factor(100.0, X0)
    with pytest.raises(ValueError):
        energy_to_rate(0.5, x0=0.0, scaling_factor=sf)

def test_energy_to_rate_rejects_nonpositive_scaling_factor():
    with pytest.raises(ValueError):
        energy_to_rate(0.5, X0, scaling_factor=-1.0)


# ---------------------------------------------------------------------------
# Round-trip: rate → energy → rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x", [0.0, 0.001, 1.0, 10.0, 100.0, 1234.5])
def test_round_trip_rate_to_energy_to_rate(x):
    xmax = 1234.5
    sf = compute_scaling_factor(xmax, X0)
    e = rate_to_energy(x, X0, sf)
    x_recovered = energy_to_rate(e, X0, sf)
    assert x_recovered == pytest.approx(x, rel=1e-9)


# ---------------------------------------------------------------------------
# Round-trip: energy → rate → energy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("e", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_round_trip_energy_to_rate_to_energy(e):
    xmax = 500.0
    sf = compute_scaling_factor(xmax, X0)
    x = energy_to_rate(e, X0, sf)
    e_recovered = rate_to_energy(x, X0, sf)
    assert e_recovered == pytest.approx(e, rel=1e-9)


# ---------------------------------------------------------------------------
# Free energy interpretation: for x >> x0, energy ≈ ΔΔG‡ / ΔΔG‡_max
# ---------------------------------------------------------------------------

def test_free_energy_approximation_holds_for_large_rates():
    # For x >> x0: ln(1 + x/x0) ≈ ln(x/x0)
    # So energy(x) ≈ ln(x/x0) / ln(xmax/x0) = ΔΔG‡(x) / ΔΔG‡(xmax)
    x0 = 1.0
    xmax = 1e6
    sf = compute_scaling_factor(xmax, x0)
    x = 1e4  # x >> x0
    exact = rate_to_energy(x, x0, sf)
    approx = math.log(x / x0) / math.log(xmax / x0)
    # Should agree to within 0.1% for x/x0 = 10000
    assert abs(exact - approx) / approx < 0.001

def test_free_energy_approximation_breaks_near_x0():
    # For x ≈ x0 the approximation ln(1+x/x0)≈ln(x/x0) is poor
    x0 = 1.0
    xmax = 1e6
    sf = compute_scaling_factor(xmax, x0)
    x = 0.5  # x < x0, approximation definitely breaks down
    exact = rate_to_energy(x, x0, sf)
    # approx would give ln(0.5)/ln(1e6) < 0, but exact must be > 0
    assert exact > 0.0


# ---------------------------------------------------------------------------
# normalize_rates
# ---------------------------------------------------------------------------

def test_normalize_rates_max_is_one():
    rates = [0.0, 5.0, 20.0, 100.0]
    energies, sf = normalize_rates(rates, X0)
    assert energies[-1] == pytest.approx(1.0)

def test_normalize_rates_zero_stays_zero():
    rates = [0.0, 50.0, 100.0]
    energies, sf = normalize_rates(rates, X0)
    assert energies[0] == 0.0

def test_normalize_rates_returns_correct_scaling_factor():
    rates = [0.0, 50.0, 100.0]
    energies, sf = normalize_rates(rates, X0)
    expected_sf = compute_scaling_factor(100.0, X0)
    assert sf == pytest.approx(expected_sf)

def test_normalize_rates_energies_in_unit_interval():
    rates = [0.0, 1.0, 7.3, 42.0, 999.0]
    energies, sf = normalize_rates(rates, X0)
    for e in energies:
        assert 0.0 <= e <= 1.0 + 1e-12

def test_normalize_rates_round_trip():
    rates = [0.0, 1.0, 7.3, 42.0, 999.0]
    energies, sf = normalize_rates(rates, X0)
    recovered = [energy_to_rate(e, X0, sf) for e in energies]
    for x, xr in zip(rates, recovered):
        assert xr == pytest.approx(x, rel=1e-9)

def test_normalize_rates_rejects_empty():
    with pytest.raises(ValueError):
        normalize_rates([], X0)

def test_normalize_rates_rejects_all_zero():
    with pytest.raises(ValueError):
        normalize_rates([0.0, 0.0, 0.0], X0)

def test_normalize_rates_single_positive():
    energies, sf = normalize_rates([42.0], X0)
    assert energies[0] == pytest.approx(1.0)

def test_normalize_rates_different_x0_scales():
    # Changing x0 changes energies but round-trip still holds
    rates = [0.0, 10.0, 100.0]
    for x0 in [0.1, 1.0, 10.0]:
        energies, sf = normalize_rates(rates, x0)
        recovered = [energy_to_rate(e, x0, sf) for e in energies]
        for x, xr in zip(rates, recovered):
            assert xr == pytest.approx(x, rel=1e-9)
