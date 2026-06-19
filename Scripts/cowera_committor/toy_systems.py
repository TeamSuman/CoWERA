r"""Analytic toy systems for validating the neural committor (no MD stack).

For a 1D overdamped Langevin process on a potential V(x),
    dx = -V'(x) dt + sqrt(2/beta) dW,
the committor between A = {x <= a} and B = {x >= b} has the closed form
    q(x) = \int_a^x e^{beta V(y)} dy  /  \int_a^b e^{beta V(y)} dy,
which we evaluate by quadrature to provide ground truth for the learned model.

This module provides the symmetric double well V(x) = (x^2 - 1)^2, a short-
trajectory generator (the source of semigroup training pairs), and the exact
committor.
"""
from __future__ import annotations

import numpy as np


def double_well_potential(x):
    x = np.asarray(x, dtype=float)
    return (x * x - 1.0) ** 2


def double_well_force(x):
    """-dV/dx for V = (x^2 - 1)^2."""
    x = np.asarray(x, dtype=float)
    return -4.0 * x * (x * x - 1.0)


def exact_committor_1d(x, beta, potential=double_well_potential, a=-1.0, b=1.0,
                       n_quad=20001):
    """Exact 1D committor q(x) for A={<=a}, B={>=b} by quadrature.

    Values are clipped to [0, 1] and are exactly 0 for x<=a and 1 for x>=b.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    grid = np.linspace(a, b, n_quad)
    integrand = np.exp(beta * potential(grid))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                           * np.diff(grid))])
    denom = cdf[-1]

    q = np.interp(x, grid, cdf / denom)
    q = np.clip(q, 0.0, 1.0)
    q[x <= a] = 0.0
    q[x >= b] = 1.0
    return q


def overdamped_langevin(x0, n_steps, dt, beta, force=double_well_force, rng=None):
    """Generate a single overdamped Langevin trajectory.

    Returns an array of shape (n_steps + 1,).
    """
    rng = np.random.default_rng() if rng is None else rng
    x = float(x0)
    out = np.empty(n_steps + 1)
    out[0] = x
    noise_scale = np.sqrt(2.0 * dt / beta)
    for t in range(n_steps):
        x = x + force(x) * dt + noise_scale * rng.standard_normal()
        out[t + 1] = x
    return out


def sample_semigroup_pairs(n_pairs, tau_steps, dt, beta, x_range=(-1.5, 1.5),
                           force=double_well_force, rng=None):
    """Sample short-trajectory pairs (x_t, x_{t+tau}) from the interior.

    Start points are drawn uniformly over ``x_range`` (a crude interior
    sampler sufficient for validation); each is propagated ``tau_steps`` to
    give the lag-tau partner. Returns ``(starts, ends)`` arrays of shape
    ``(n_pairs, 1)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    starts = rng.uniform(x_range[0], x_range[1], size=n_pairs)
    ends = np.empty_like(starts)
    noise_scale = np.sqrt(2.0 * dt / beta)
    for k in range(n_pairs):
        x = starts[k]
        for _ in range(tau_steps):
            x = x + force(x) * dt + noise_scale * rng.standard_normal()
        ends[k] = x
    return starts[:, None], ends[:, None]
