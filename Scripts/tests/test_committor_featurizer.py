"""Tests for committor featurizers: shape, passthrough, and SE(3) invariance."""
import numpy as np
import pytest

from cowera_committor.featurizer import IdentityFeaturizer, DistanceFeaturizer


def test_identity_passthrough():
    f = IdentityFeaturizer(n_features=2)
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert np.allclose(f.featurize(X), X)
    assert f.n_features == 2


def test_identity_wrong_dim_raises():
    f = IdentityFeaturizer(n_features=3)
    with pytest.raises(ValueError):
        f.featurize(np.zeros((4, 2)))


def _random_rotation(rng):
    # random rotation matrix via QR
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_distance_features_invariant_to_rotation_translation():
    rng = np.random.default_rng(0)
    n_atoms = 6
    coords = rng.standard_normal((1, n_atoms, 3))
    pairs = [(0, 1), (1, 2), (0, 3), (2, 5), (3, 4)]
    f = DistanceFeaturizer(pairs)

    base = f.featurize(coords)

    R = _random_rotation(rng)
    t = rng.standard_normal(3)
    moved = coords @ R.T + t
    movedf = f.featurize(moved)

    assert np.allclose(base, movedf, atol=1e-10)
    assert f.n_features == len(pairs)


def test_distance_features_values():
    coords = np.array([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]])  # distance 5
    f = DistanceFeaturizer([(0, 1)])
    assert np.allclose(f.featurize(coords), [[5.0]])


def test_distance_inverse_option():
    coords = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])
    f = DistanceFeaturizer([(0, 1)], inverse=True)
    assert np.allclose(f.featurize(coords), [[0.5]], atol=1e-6)
