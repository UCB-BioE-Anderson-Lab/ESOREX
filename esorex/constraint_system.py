"""ESOREX energetic constraint system.

Substrate specificity as a free-energy decomposition:

    E_S = E_0 + Σ_i w_i x_{S,i} + Σ_{i<j} w_{ij} x_{S,i} x_{S,j} + ...
                                                        with   E_S = -RT ln k_S

Rather than fitting one weight vector, this keeps the whole solution space of the
linear system  X w = E  (rows = substrates; columns = an intercept E_0, first-order
features, and any cooperative interaction features): a particular solution w_p plus
a basis N for the null space, so every exact-fitting weighting is  w = w_p + N z.

That representation separates the two failure modes:

  * Underdetermination, many exact solutions (rank < #columns).  Individual
    weights are not pinned, but a PREDICTION is still determined when the test
    substrate's feature vector q is orthogonal to the null space (qᵀN ≈ 0): then
    every exact-fitting weighting agrees on E_Q.  Parameter
    underdetermination is not prediction underdetermination.

  * Inconsistency, no exact solution (E ∉ column space of X).  The additive model
    is insufficiently expressive; cooperative interaction terms are generated to
    restore exactness, unless the conflict is between substrates with
    IDENTICAL feature vectors, a featurization limit no interaction can repair.

Energies carry an arbitrary zero (absorbed by E_0), so only differences are
physical; RT is a fixed scale that cancels between -RT ln k and its inverse.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.linalg import null_space

RT_KCAL_298 = 0.5924          # kcal/mol at 298 K


def energy_from_rate(k, RT: float = RT_KCAL_298):
    """E = -RT ln k  (lower E = faster substrate)."""
    return -RT * np.log(np.asarray(k, dtype=float))


def rate_from_energy(E, RT: float = RT_KCAL_298):
    """k = exp(-E / RT), the inverse of energy_from_rate."""
    return np.exp(-np.asarray(E, dtype=float) / RT)


class ConstraintSystem:
    """The linear system X w = E for one mechanistic class, with its solution space.

    Columns are ordered [first-order features..., interaction features..., intercept].
    An interaction feature is a tuple of first-order features whose product column is
    1 exactly where every member feature is present."""

    def __init__(self, features, feature_sets, energies,
                 interactions=None, consistency_tol: float = 1e-6):
        self.features = list(features)            # first-order columns
        self.feature_sets = [frozenset(fs) for fs in feature_sets]
        self.interactions = [tuple(t) for t in (interactions or [])]
        self.consistency_tol = consistency_tol
        self.E = np.asarray(energies, float)
        self.n = len(energies)
        self._rebuild()

    # -- matrix assembly ---------------------------------------------------------

    def _feature_column(self, f) -> np.ndarray:
        """Design column for a single first-order feature (which is itself a tuple)."""
        return np.array([1.0 if f in fs else 0.0 for fs in self.feature_sets])

    def _interaction_column(self, members) -> np.ndarray:
        """Product column: 1 exactly where every member feature is present."""
        return np.array([1.0 if all(m in fs for m in members) else 0.0
                         for fs in self.feature_sets])

    def _rebuild(self):
        cols = [self._feature_column(f) for f in self.features]
        cols += [self._interaction_column(t) for t in self.interactions]
        F = len(cols)
        X = np.ones((self.n, F + 1))              # last column = intercept E_0
        for j, c in enumerate(cols):
            X[:, j] = c
        self.X, self.F = X, F
        self._analyze()

    def _analyze(self):
        X, E = self.X, self.E
        self.rank = int(np.linalg.matrix_rank(X)) if self.n else 0
        # A particular solution (minimum-norm least squares); exact iff consistent.
        self.w_p, *_ = np.linalg.lstsq(X, E, rcond=None)
        self.max_residual = float(np.max(np.abs(X @ self.w_p - E))) if self.n else 0.0
        self.consistent = self.max_residual <= self.consistency_tol
        # Null space of X: weight changes that leave every training energy unchanged.
        self.null_basis = null_space(X) if self.n else np.zeros((self.F + 1, 0))
        self.null_dim = self.null_basis.shape[1]

    # -- cooperative-term generation ----------------------------------------------

    def _conflict_direction(self):
        """A left-null vector c of X with cᵀE ≠ 0: a set of substrates whose feature
        rows are linearly dependent while their energies are not (an additive
        conflict).  Returns (c, |cᵀE|) for the strongest conflict, or None if exact."""
        U, S, _ = np.linalg.svd(self.X, full_matrices=True)
        left_null = U[:, self.rank:]              # columns spanning the left null space
        if left_null.shape[1] == 0:
            return None
        comps = left_null.T @ self.E              # E's projection onto conflict space
        k = int(np.argmax(np.abs(comps)))
        if abs(comps[k]) <= self.consistency_tol:
            return None
        return left_null[:, k], float(abs(comps[k]))

    def _candidate_pairs(self, c, tol=1e-9):
        """First-order feature pairs whose product column breaks conflict c.  Because
        c is orthogonal to every existing column, cᵀp ≠ 0 guarantees the product is
        linearly independent of the current design.  Restricted to features
        co-occurring within the conflicting substrates."""
        conflict_subs = [s for s in range(self.n) if abs(c[s]) > tol]
        present = [f for f in self.features
                   if any(f in self.feature_sets[s] for s in conflict_subs)]
        out = []
        for i, j in combinations(present, 2):
            p = self._interaction_column((i, j))
            if abs(float(c @ p)) > tol:
                out.append((i, j))
        return out

    def restore_exactness(self, pick_pair=None, max_terms: int = 50):
        """Add minimal pairwise cooperative terms until the training energies are
        reproduced exactly.  pick_pair(candidates) chooses among distinguishing
        pairs (default: deterministic, first sorted); supply a most-abstract chooser
        to honor the feature hierarchy.  Returns the list of interactions added.

        Raises RuntimeError if a conflict has NO distinguishing pair, the conflicting
        substrates share an identical feature vector, so no interaction of these
        features can separate them (a featurization-resolution limit)."""
        added = []
        while not self.consistent and len(self.interactions) < max_terms:
            conflict = self._conflict_direction()
            if conflict is None:
                break
            c, _ = conflict
            candidates = self._candidate_pairs(c)
            if not candidates:
                raise RuntimeError(
                    "additive conflict with no distinguishing feature pair: the "
                    "conflicting substrates have identical feature vectors "
                    "(featurization cannot resolve them; no interaction term can fix "
                    "this)")
            pair = (pick_pair(candidates) if pick_pair else sorted(candidates, key=str)[0])
            self.interactions.append(tuple(pair))
            added.append(tuple(pair))
            self._rebuild()
        return added

    # -- prediction --------------------------------------------------------------

    def query_vector(self, feature_set) -> np.ndarray:
        fs = frozenset(feature_set)
        q = np.zeros(self.F + 1)
        j = 0
        for f in self.features:
            q[j] = 1.0 if f in fs else 0.0
            j += 1
        for t in self.interactions:
            q[j] = 1.0 if all(m in fs for m in t) else 0.0
            j += 1
        q[-1] = 1.0                               # intercept always active
        return q

    def is_determined(self, q, tol: float = 1e-6) -> bool:
        """A prediction is identifiable iff q is orthogonal to every null direction,
        so all exact-fitting weightings give the same energy."""
        if self.null_dim == 0:
            return True
        return bool(np.all(np.abs(q @ self.null_basis) <= tol))

    def novelty(self, q) -> float:
        """Fraction of the query outside the training row space, the part no exact
        fit constrains.  0 = fully determined (pure interpolation); toward 1 = mostly
        unprecedented combinations.  Because min-norm w_p lies in the row space,
        predict_energy already sets every unsupported direction to ΔE = 0; this
        reports how much was zeroed, i.e. how far the prediction extrapolates."""
        if self.null_dim == 0:
            return 0.0
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            return 0.0
        return float(np.linalg.norm(q @ self.null_basis) / norm)

    def predict_energy(self, feature_set):
        """Return (E_Q, determined, novelty).  E_Q is the row-space (supported)
        prediction: unique across all exact fits when determined is True, with
        unsupported novelty set to ΔE = 0 otherwise."""
        q = self.query_vector(feature_set)
        return float(q @ self.w_p), self.is_determined(q), self.novelty(q)

    def diagnostics(self) -> dict:
        return {
            "n_substrates": self.n,
            "n_features": len(self.features),
            "n_interactions": len(self.interactions),
            "n_columns": self.F + 1,             # + intercept
            "rank": self.rank,
            "null_dim": self.null_dim,
            "consistent": self.consistent,       # exact additive/cooperative solution?
            "max_residual": self.max_residual,
        }
