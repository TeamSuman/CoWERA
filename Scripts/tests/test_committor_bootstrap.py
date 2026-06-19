"""Tests for the Phase-0 structural interpolant bootstrap."""
import numpy as np
import pytest

from cowera_committor.bootstrap import structural_interpolants


def test_interpolant_shapes_and_labels():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 4, 3))   # 5 A structures, 4 atoms
    B = rng.standard_normal((6, 4, 3))
    alphas = (0.2, 0.4, 0.6, 0.8)
    X, y = structural_interpolants(A, B, alphas=alphas, n_pairs=10, rng=rng)
    assert X.shape == (10 * len(alphas), 4, 3)
    assert y.shape == (10 * len(alphas),)
    # soft labels are 1 - alpha
    assert set(np.round(np.unique(y), 3)) == {0.2, 0.4, 0.6, 0.8}


def test_interpolant_is_convex_combination():
    A = np.zeros((1, 2, 3))
    B = np.ones((1, 2, 3))
    X, y = structural_interpolants(A, B, alphas=(0.25,), n_pairs=1)
    # x = 0.25*A + 0.75*B = 0.75 ; label = 1 - 0.25 = 0.75
    assert np.allclose(X, 0.75)
    assert np.allclose(y, 0.75)


def test_interpolant_relax_fn_applied():
    A = np.zeros((1, 2))
    B = np.ones((1, 2))
    X, y = structural_interpolants(A, B, alphas=(0.5,), n_pairs=1,
                                   relax_fn=lambda x: x * 0.0)
    assert np.allclose(X, 0.0)


def test_interpolant_1d_inputs():
    A = np.array([-1.0, -1.1, -0.9])
    B = np.array([1.0, 1.1, 0.9])
    X, y = structural_interpolants(A, B, alphas=(0.5,), n_pairs=3)
    assert X.shape == (3, 1)
