"""Phase-0 structural bootstrap: generate interior training data without MD.

A committor trained only on A/B end states has a flat gradient in the transition
region. To seed a non-flat ``q^0``, we add **structural interpolants**: linear
Cartesian interpolations between A and B configurations carry intermediate
structural character and are given soft labels. After optional energy
minimization (injected, since it needs an MD engine) they provide the variational
loss with interior anchor points.

Pure-numpy (the relaxation step is an injected callable), so this is testable
without an MD stack.
"""
from __future__ import annotations

import numpy as np


def structural_interpolants(x_A_set, x_B_set, alphas=(0.2, 0.4, 0.6, 0.8),
                            n_pairs=None, relax_fn=None, rng=None):
    """Generate linearly-interpolated structures with soft committor labels.

    For sampled pairs ``(x_A, x_B)`` and each ``alpha``::

        x = alpha * x_A + (1 - alpha) * x_B,    y = 1 - alpha

    so ``alpha -> 1`` gives an A-like structure (label ~0) and ``alpha -> 0`` a
    B-like structure (label ~1).

    Parameters
    ----------
    x_A_set, x_B_set : array-like, shape (n, n_atoms, 3) or (n, d)
        Ensembles of A-state and B-state configurations.
    alphas : sequence of float in (0, 1)
        Interpolation weights.
    n_pairs : int or None
        Number of (A, B) pairs to sample (default: min(len A, len B)).
    relax_fn : callable or None
        Optional ``x -> x_relaxed`` energy-minimization (e.g. OpenMM) applied to
        each interpolant to remove clashes. Identity if None.
    rng : numpy Generator or None

    Returns
    -------
    X : np.ndarray, shape (n_pairs * len(alphas), ...)
    y : np.ndarray, shape (n_pairs * len(alphas),)
        Soft committor labels in (0, 1).
    """
    rng = np.random.default_rng() if rng is None else rng
    A = np.asarray(x_A_set, dtype=float)
    B = np.asarray(x_B_set, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]

    if n_pairs is None:
        n_pairs = min(len(A), len(B))
    ia = rng.integers(0, len(A), size=n_pairs)
    ib = rng.integers(0, len(B), size=n_pairs)

    X_list = []
    y_list = []
    for a in alphas:
        x = a * A[ia] + (1.0 - a) * B[ib]
        if relax_fn is not None:
            x = np.asarray([relax_fn(xi) for xi in x], dtype=float)
        X_list.append(x)
        y_list.append(np.full(n_pairs, 1.0 - a))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)
    return X, y
