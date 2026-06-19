"""Tests for the committor-driven distance/intensity calculator (M2).

These validate the Δq resampling path and the committor-augmented merge distance
without ruptures/mdtraj/GPU, using an injected changepoint function and synthetic
committor histories.
"""
import numpy as np
import pytest

from cowera_committor.committor_metric import (
    CommittorDistances,
    augment_merge_distance,
    committor_displacement,
    suppress_tse_merges,
)


def test_suppress_tse_merges_blocks_near_transition_state():
    D = np.ones((3, 3)) - np.eye(3)
    q = np.array([0.05, 0.5, 0.95])   # walker 1 is at the TSE
    out = suppress_tse_merges(D, q, band=0.2)
    assert np.isinf(out[1, 0]) and np.isinf(out[0, 1])  # any pair with walker 1
    assert np.isinf(out[1, 2])
    assert out[0, 2] == 1.0           # pair not involving the TSE walker is finite
    assert out[1, 1] == 0.0           # diagonal preserved


def test_suppress_tse_merges_band_zero_is_noop():
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(suppress_tse_merges(D, np.array([0.4, 0.6]), band=0.0), D)
from cowera_committor.committor_model import MLPCommittor
from cowera_committor.featurizer import DistanceFeaturizer


# full-history-window changepoint substitute (no ruptures needed)
def _full_window(q):
    return np.array([len(q)])


# ------------------------- pure helpers ------------------------------------- #

def test_augment_merge_distance_combines_rmsd_and_committor():
    d_rmsd = np.zeros((3, 3))
    q = np.array([0.0, 0.5, 1.0])
    D = augment_merge_distance(d_rmsd, q, alpha=0.5)
    assert D[0, 2] == pytest.approx(0.5)   # 0.5*0 + 0.5*|1-0|
    assert D[0, 1] == pytest.approx(0.25)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)


def test_augment_merge_distance_alpha_one_is_pure_rmsd():
    d_rmsd = np.array([[0.0, 1.0], [1.0, 0.0]])
    q = np.array([0.0, 1.0])
    assert np.allclose(augment_merge_distance(d_rmsd, q, alpha=1.0), d_rmsd)


def test_augment_merge_distance_vector_committor_l1():
    d_rmsd = np.zeros((2, 2))
    q = np.array([[0.2, 0.8], [0.6, 0.4]])  # (N, M) vector committor
    D = augment_merge_distance(d_rmsd, q, alpha=0.0)
    assert D[0, 1] == pytest.approx(0.8)  # |0.2-0.6| + |0.8-0.4|


def test_committor_displacement():
    q = np.array([0.1, 0.2, 0.5, 0.9])
    dq, w = committor_displacement(q, change_points=[0, 4])
    assert dq == pytest.approx(0.8)   # 0.9 - 0.1
    assert w == pytest.approx(0.9)


# ------------------------- Δq phase path ------------------------------------ #

def _make_metric(**kw):
    return CommittorDistances(committor_model=None, featurizer=None,
                              increment=1, changepoint_fn=_full_window, **kw)


def test_phase_from_projection_returns_delta_q():
    m = _make_metric()
    q = np.linspace(0.2, 0.8, 8)
    bins, dq, w = m.phase_from_projection(q, drange=np.array([0, 1]), n_d=4, n_bins=100)
    assert dq == pytest.approx(0.6)
    assert w == pytest.approx(0.8)
    assert len(bins) == 4


def test_phase_from_projection_warped_walker():
    m = _make_metric()
    bins, dq, w = m.phase_from_projection(np.array([0.5]), drange=np.array([0, 1]),
                                          n_d=3, n_bins=100)
    assert dq == 0.0
    assert len(bins) == 3


def test_intensity_productive_walker_is_brightest():
    """End-to-end: increasing-q walker outranks flat and decreasing ones."""
    m = _make_metric()
    q_up = np.linspace(0.2, 0.8, 8)     # Δq = +0.6 (productive)
    q_flat = np.full(8, 0.5)            # Δq = 0
    q_down = np.linspace(0.8, 0.2, 8)   # Δq = -0.6 (regressing)
    projections = [q_up, q_flat, q_down]
    dranges = [np.array([0.0, 1.0])] * 3

    intensity, n_bins = m.intensity_from_projections(
        projections, dranges, n_d=4, it=0, n_bins=100, max_bins=100, n_jobs=1)

    assert np.argmax(intensity) == 0
    assert intensity[0] == pytest.approx(1.0)
    assert intensity[2] <= intensity[1] <= intensity[0]


# ------------------------- model inference ---------------------------------- #

def test_predict_q_state_with_distance_featurizer():
    pairs = [(0, 1), (1, 2), (0, 3)]
    feat = DistanceFeaturizer(pairs)
    model = MLPCommittor(input_dim=len(pairs), hidden=(8,), seed=0)
    m = CommittorDistances(committor_model=model, featurizer=feat,
                           increment=1, changepoint_fn=_full_window)

    coords = np.random.default_rng(0).standard_normal((4, 3))
    q = m.predict_q_state({'positions': coords})
    assert 0.0 <= q <= 1.0

    image = m.get_proj_coord({'positions': coords})
    assert image.shape == (1,)
    assert 0.0 <= image[0] <= 1.0
