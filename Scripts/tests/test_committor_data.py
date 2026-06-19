"""Tests for the WE-weighted committor training buffers."""
import numpy as np

from cowera_committor.committor_data import (
    BoundaryBuffer,
    SemigroupBuffer,
    ShootingBuffer,
)


def test_boundary_buffer_accumulates_and_is_permanent():
    b = BoundaryBuffer()
    b.add(np.zeros((3, 2)), weights=[0.1, 0.2, 0.3])
    b.add(np.ones((2, 2)))
    X, w = b.arrays()
    assert X.shape == (5, 2)
    assert len(w) == 5
    assert np.allclose(w[:3], [0.1, 0.2, 0.3])
    assert np.allclose(w[3:], 1.0)
    assert len(b) == 5


def test_semigroup_buffer_sliding_window():
    s = SemigroupBuffer(max_size=4)
    s.add(np.arange(3)[:, None], np.arange(3)[:, None] + 10)
    s.add(np.arange(3, 6)[:, None], np.arange(3, 6)[:, None] + 10)
    Xt, Xtp, w = s.arrays()
    # only the most recent 4 pairs retained
    assert len(Xt) == 4
    assert np.allclose(Xt.ravel(), [2, 3, 4, 5])
    assert np.allclose(Xtp.ravel(), [12, 13, 14, 15])
    assert len(w) == 4


def test_semigroup_buffer_shape_mismatch():
    s = SemigroupBuffer()
    import pytest
    with pytest.raises(ValueError):
        s.add(np.zeros((2, 3)), np.zeros((2, 2)))


def test_shooting_buffer_sliding_window_and_counts():
    sb = ShootingBuffer(max_size=3)
    sb.add(np.zeros((2, 2)), n_A=[5, 3], n_B=[5, 7], weights=[1.0, 2.0])
    sb.add(np.ones((2, 2)), n_A=[0, 1], n_B=[10, 9])
    X, nA, nB, w = sb.arrays()
    assert len(X) == 3
    # most recent retained
    assert np.allclose(X[-1], 1.0)
    assert len(nA) == 3 and len(nB) == 3 and len(w) == 3


def test_empty_buffers_return_empty():
    assert len(BoundaryBuffer()) == 0
    assert len(SemigroupBuffer()) == 0
    assert len(ShootingBuffer()) == 0
    X, w = BoundaryBuffer().arrays()
    assert X.size == 0 and w.size == 0
