"""Tests for the Best-Hummer Q core (``cowera.features.compute_q``).

These validate the vectorized fraction-of-native-contacts formula (Eq. A1)
against a slow, explicit reference implementation. No mdtraj required.
"""
import numpy as np
import pytest

from cowera.features import compute_q, BETA_CONST, LAMBDA_CONST


def _reference_q(r, r0, beta, lam):
    """Slow, readable reference for the fraction of native contacts."""
    r = np.atleast_2d(r)
    r0 = np.asarray(r0).reshape(-1)
    n_frames, n_contacts = r.shape
    q = np.empty(n_frames)
    for f in range(n_frames):
        acc = 0.0
        for c in range(n_contacts):
            acc += 1.0 / (1.0 + np.exp(beta * (r[f, c] - lam * r0[c])))
        q[f] = acc / n_contacts
    return q


def test_compute_q_matches_reference():
    rng = np.random.default_rng(0)
    n_frames, n_contacts = 8, 25
    r0 = rng.uniform(0.30, 0.45, size=n_contacts)
    r = r0[None, :] + rng.normal(0.0, 0.05, size=(n_frames, n_contacts))

    got = compute_q(r, r0, beta=BETA_CONST, lam=LAMBDA_CONST)
    ref = _reference_q(r, r0, BETA_CONST, LAMBDA_CONST)
    assert np.allclose(got, ref)


def test_compute_q_native_structure_is_high():
    """A frame at the native distances should give Q near (but below) 1."""
    r0 = np.full(30, 0.4)
    q = compute_q(r0[None, :], r0)
    # at r == r0: 1/(1+exp(beta*(r0 - lam*r0))) with lam>1 -> > 0.5, near 1
    assert q[0] > 0.9


def test_compute_q_fully_broken_is_low():
    """Frames with all contacts far beyond the native distance -> Q near 0."""
    r0 = np.full(30, 0.4)
    r = np.full((1, 30), 2.0)
    q = compute_q(r, r0)
    assert q[0] < 0.05


def test_compute_q_bounded_unit_interval():
    rng = np.random.default_rng(1)
    r0 = rng.uniform(0.3, 0.45, size=40)
    r = rng.uniform(0.1, 1.5, size=(20, 40))
    q = compute_q(r, r0)
    assert np.all(q >= 0.0) and np.all(q <= 1.0)
