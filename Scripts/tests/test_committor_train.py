"""Tests for committor training-data assembly and the retraining orchestration."""
import numpy as np
import pytest

from cowera_committor.committor_data import (
    BoundaryBuffer, SemigroupBuffer, ShootingBuffer,
)
from cowera_committor.committor_train import (
    extract_segment_training_data, assemble_training_data,
    standardization_pool, retrain_committor,
)
from cowera_committor.committor_model import MLPCommittor
from cowera_committor.phase_manager import BootstrapPhaseManager, SEMIGROUP
from cowera_committor.toy_systems import exact_committor_1d, sample_semigroup_pairs


# ------------------------ per-frame routing --------------------------------- #

def test_extract_segment_training_data_routes_frames():
    feats = np.array([[0.05], [0.5], [0.6], [0.95]])
    q = np.array([0.05, 0.5, 0.6, 0.95])
    A, B, Xt, Xtp = extract_segment_training_data(feats, q)
    assert np.allclose(A, [[0.05]])         # frame 0 -> A boundary
    assert np.allclose(B, [[0.95]])         # frame 3 -> B boundary
    assert np.allclose(Xt, [[0.5], [0.6]])  # intermediate starts
    assert np.allclose(Xtp, [[0.6], [0.95]])  # their successors


def test_extract_all_basin_gives_no_pairs():
    feats = np.array([[0.01], [0.02]])
    q = np.array([0.01, 0.02])
    A, B, Xt, Xtp = extract_segment_training_data(feats, q)
    assert len(A) == 2 and len(B) == 0 and len(Xt) == 0


# ------------------------ buffer assembly ----------------------------------- #

def test_assemble_training_data_from_buffers():
    bA, bB = BoundaryBuffer(), BoundaryBuffer()
    sem, shoot = SemigroupBuffer(), ShootingBuffer()
    bA.add(np.zeros((2, 1)))
    bB.add(np.ones((3, 1)))
    sem.add(np.zeros((4, 1)), np.ones((4, 1)))
    data = assemble_training_data(bA, bB, sem, shoot)
    assert data['A'].shape == (2, 1)
    assert data['B'].shape == (3, 1)
    assert data['sem'][0].shape == (4, 1)
    assert 'shoot' not in data  # empty buffer omitted
    pool = standardization_pool(data)
    assert pool.shape[0] == 2 + 3 + 4 + 4


def test_assemble_empty_is_empty():
    assert assemble_training_data() == {}
    assert standardization_pool({}) is None


# ------------------------ end-to-end retrain -------------------------------- #

def test_retrain_recovers_double_well_committor():
    """Retraining through the buffer/phase-manager path recovers the committor."""
    rng = np.random.default_rng(5)
    beta, dt, tau = 1.0, 0.002, 30
    Xt, Xtp = sample_semigroup_pairs(1500, tau, dt, beta, x_range=(-1.4, 1.4), rng=rng)
    xA = rng.uniform(-1.6, -1.0, 250)[:, None]
    xB = rng.uniform(1.0, 1.6, 250)[:, None]

    bA, bB = BoundaryBuffer(), BoundaryBuffer()
    sem = SemigroupBuffer(max_size=5000)
    bA.add(xA); bB.add(xB); sem.add(Xt, Xtp)

    data = assemble_training_data(bA, bB, sem)
    manager = BootstrapPhaseManager(start_phase=SEMIGROUP)
    # boost boundary enforcement for a tight fit
    manager_weights = manager.loss_weights()
    manager_weights.update({'lambda_A': 5.0, 'lambda_B': 5.0})
    model = MLPCommittor(input_dim=1, hidden=(32, 32), seed=11, l2=1e-5)
    # use the manager's phase weights via retrain_committor, but override lambdas
    model.fit(data, weights=manager_weights, epochs=500, lr=5e-3,
              standardize_from=standardization_pool(data))

    grid = np.linspace(-1, 1, 101)
    mae = np.mean(np.abs(model.predict(grid[:, None]) - exact_committor_1d(grid, beta)))
    assert mae < 0.08, f"committor MAE too high through training path: {mae}"


def test_retrain_committor_uses_phase_weights():
    """retrain_committor pulls loss weights from the phase manager."""
    bA, bB = BoundaryBuffer(), BoundaryBuffer()
    sem = SemigroupBuffer()
    bA.add(np.full((5, 1), -1.0)); bB.add(np.full((5, 1), 1.0))
    sem.add(np.zeros((10, 1)), np.full((10, 1), 0.1))
    data = assemble_training_data(bA, bB, sem)

    model = MLPCommittor(input_dim=1, hidden=(8,), seed=0)
    manager = BootstrapPhaseManager(start_phase=SEMIGROUP)
    hist = retrain_committor(model, data, manager, epochs=50, lr=1e-2)
    assert hist[-1] < hist[0]
