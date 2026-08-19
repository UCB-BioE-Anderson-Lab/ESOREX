"""
free_energy.py, Conversion between kinetic rates and normalized activation
free energies in [0, 1].

Motivation
----------
Enzymatic substrate activity data comes in several forms, exact kinetic rates
(kf/KD in M⁻¹s⁻¹), boolean active/inactive calls, or a mix of both, and
poses several practical challenges for machine learning:

1. Dynamic range.  Rates span many orders of magnitude (e.g. 0.01 to 10⁶ M⁻¹s⁻¹
   in a single dataset).  A linear scale compresses nearly all substrates into a
   tiny fraction of the feature space; a log scale fixes compression but is
   undefined at zero.

2. Aggregation across enzymes.  To combine predictions or labels from multiple
   enzymes or datasets into a single model, values must be on a common bounded
   scale.  The [0, 1] interval enables geometric means, weighted averages, and
   other aggregations that are ill-defined on an unbounded log scale.

3. Physical interpretability.  Log rate differences are proportional to activation
   free energy differences (ΔΔG‡) via the Eyring equation.  A normalized ΔΔG‡ in
   [0, 1] is more interpretable than a raw rate: 0.7 means "this substrate lowers
   the activation barrier by 70% of the maximum observed reduction."

4. Coexistence of boolean and kinetic labels.  Some substrates are characterized
   only as active (1) or inactive (0) without a measured rate.  By mapping exact
   zeros to 0 and the best kinetic rate to 1, boolean labels slot naturally into
   the same [0, 1] scale as normalized kinetic values.  A boolean-positive
   substrate with no rate data receives 1.0 by modeling convention (not as a
   consequence of the kinetic transform); a boolean-negative receives 0.0;
   substrates with measured rates receive intermediate values.  This allows a
   single model to be trained on heterogeneous datasets where some labels are
   kinetic and others are qualitative.

Physical interpretation
-----------------------
Transition state theory (Eyring equation) relates a rate constant k to its
activation free energy ΔG‡:

    k = (kT/h) · exp(−ΔG‡ / RT)

Inverting:

    ΔG‡ = −RT · ln(k · h/kT)
        ∝ −RT · ln(k)          (h/kT is a constant at fixed temperature)

So differences in log rate directly correspond to differences in ΔG‡:

    ΔΔG‡(x, x₀) = ΔG‡(x₀) − ΔG‡(x) = RT · [ln(x) − ln(x₀)] = RT · ln(x/x₀)

A substrate with a higher rate has a lower ΔG‡ and a positive ΔΔG‡ relative to
the reference.

Connection to the normalized energy score
------------------------------------------
This module computes a dimensionless, normalized form of ΔΔG‡.  Two parameters
are required:

  x₀             A fixed reference rate in the same units as the data (e.g.
                 1.0 M⁻¹s⁻¹).  It sets the transition point between the
                 approximately linear low-rate regime and the logarithmically
                 compressed high-rate regime.  Only x = 0 maps to energy = 0;
                 x₀ itself maps to ln(2) / scaling_factor (a small positive
                 value).  Physically, x₀ should be near the assay detection
                 floor so that all measurable substrates have positive
                 normalized energies.

  x_max          The maximum rate used to set the scale.  Its scope is a modeling
                 choice: per enzyme, per dataset, or global across a collection.
                 Whatever scope is chosen, the resulting scaling_factor must be
                 stored so scores can be inverted consistently.

  scaling_factor = ln(1 + x_max / x₀)
                 Computed once from x_max and x₀ and stored alongside any
                 serialized model.

Forward transform: rate → normalized energy
-------------------------------------------
    energy(x) = 0                                    if x = 0
              = ln(1 + x/x₀) / scaling_factor        if x > 0

Equivalently:

    energy(x) = ln(1 + x/x₀) / ln(1 + x_max/x₀)

Boundary conditions:
  energy(0)     = 0   exactly  (zero rate → worst substrate, baseline energy)
  energy(x_max) = 1   exactly  (best observed substrate)
  energy(x)     ∈ (0, 1)  for  0 < x < x_max  (strictly monotone)

Free energy interpretation (x ≫ x₀ regime):
When x ≫ x₀, the +1 inside the log is negligible:

    ln(1 + x/x₀) ≈ ln(x/x₀) = ln(x) − ln(x₀)
                              = [ΔG‡(x₀) − ΔG‡(x)] / RT

So in this regime:

    energy(x) ≈ [ΔG‡(x₀) − ΔG‡(x)] / [ΔG‡(x₀) − ΔG‡(x_max)]

This is a linearly normalized ΔΔG‡: zero at x = 0, one at the best observed
rate, and proportional to the free-energy advantage over the reference for
x ≫ x₀.  Note that x₀ itself maps to ln(2) / scaling_factor, not to zero.
Substrates with rates well above x₀ can therefore be thought of as having their
activation barrier reduced by a fraction `energy` of the maximum possible
reduction in the dataset.

The +1 inside the log (the "log1p shift") is what makes the formula well-defined
at x = 0, where ΔG‡ → +∞.  For x ≈ x₀ the approximation breaks down and the
formula transitions to an approximately linear regime (ln(1 + x/x₀) ≈ x/x₀ for
x ≪ x₀), giving a smooth, zero-preserving scale across the full range.

Normalized energies are scores are relative to the chosen normalization scope.
The same raw rate may map to different normalized energies under different choices
of x_max.  This is expected: a normalized energy of 0.7 means "70% of the way
up the free-energy scale for this dataset," not an absolute ΔG‡ value.

Inverse transform: normalized energy → rate
--------------------------------------------
Starting from  e = ln(1 + x/x₀) / scaling_factor  and solving for x:

    e · scaling_factor = ln(1 + x/x₀)
    exp(e · scaling_factor) = 1 + x/x₀
    x = x₀ · (exp(e · scaling_factor) − 1)

This is an exact algebraic inverse: energy_to_rate(rate_to_energy(x)) = x to
floating-point precision, provided the same x₀ and scaling_factor are used.

Choice of logarithm base
-------------------------
Natural logarithm (base e) is used throughout, consistent with the Eyring
equation.  The base cancels in the round-trip and does not affect correctness.

Usage pattern
-------------
  # Once per dataset:
  energies, scaling_factor = normalize_rates(raw_rates, x0=1.0)
  # Store scaling_factor and x0 with the model.

  # Later, to recover a rate from a model output:
  rate = energy_to_rate(predicted_energy, x0=1.0, scaling_factor=stored_sf)
"""

import math


def compute_scaling_factor(xmax: float, x0: float) -> float:
    """Return the dataset-wide scaling factor ln(1 + xmax / x0).

    Args:
        xmax: Maximum rate used for normalization (must be > 0).
        x0:   Reference rate in the same units (must be > 0).

    Returns:
        scaling_factor (float), store this alongside any serialized model.
    """
    if xmax <= 0:
        raise ValueError(f"xmax must be positive, got {xmax}")
    if x0 <= 0:
        raise ValueError(f"x0 must be positive, got {x0}")
    return math.log(1.0 + xmax / x0)


def rate_to_energy(x: float, x0: float, scaling_factor: float) -> float:
    """Convert one kinetic rate to a normalized activation free energy in [0, 1].

    For x ≫ x0 this is proportional to ΔΔG‡ relative to the reference rate x0,
    normalized so that the maximum observed rate maps to 1.  Zero maps exactly
    to 0 (the log1p shift keeps the formula well-defined at x = 0).

    Args:
        x:              Kinetic rate (>= 0).  0 maps exactly to 0.
        x0:             Reference rate (same units, must be > 0).
        scaling_factor: Dataset-wide constant; compute with compute_scaling_factor().

    Returns:
        Normalized free energy in [0, 1].
    """
    if x < 0:
        raise ValueError(f"Kinetic rate must be >= 0, got {x}")
    if x0 <= 0:
        raise ValueError(f"x0 must be positive, got {x0}")
    if scaling_factor <= 0:
        raise ValueError(f"scaling_factor must be positive, got {scaling_factor}")
    if x == 0.0:
        return 0.0
    return math.log(1.0 + x / x0) / scaling_factor


def energy_to_rate(e: float, x0: float, scaling_factor: float) -> float:
    """Convert a normalized free energy back to an estimated kinetic rate.

    Exact algebraic inverse of rate_to_energy.  Recovery is exact to
    floating-point precision only when the same x0 and scaling_factor are used.

    Args:
        e:              Normalized free energy (should be in [0, 1] for
                        in-distribution inputs).
        x0:             Reference rate (same units used when computing the energy).
        scaling_factor: The value returned by compute_scaling_factor() for that
                        dataset.

    Returns:
        Estimated kinetic rate (>= 0).
    """
    if x0 <= 0:
        raise ValueError(f"x0 must be positive, got {x0}")
    if scaling_factor <= 0:
        raise ValueError(f"scaling_factor must be positive, got {scaling_factor}")
    return x0 * (math.exp(e * scaling_factor) - 1.0)


def normalize_rates(
    rates: list,
    x0: float,
) -> tuple:
    """Normalize a list of kinetic rates to [0, 1] normalized free energies.

    Computes the scaling_factor from the maximum rate in the list, then converts
    every rate.  The scaling_factor is returned so it can be stored and later
    used with energy_to_rate() to recover original rates.

    Args:
        rates: List of kinetic values (all >= 0; at least one must be > 0).
        x0:    Reference rate in the same units (must be > 0).

    Returns:
        (energies, scaling_factor) where energies is a list of floats in [0, 1]
        and scaling_factor is the double that must be kept for inversion.
    """
    if not rates:
        raise ValueError("rates list must not be empty")
    xmax = max(rates)
    if xmax <= 0:
        raise ValueError("At least one rate must be positive to define the scale")
    scaling_factor = compute_scaling_factor(xmax, x0)
    energies = [rate_to_energy(x, x0, scaling_factor) for x in rates]
    return energies, scaling_factor
