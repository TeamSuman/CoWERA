"""Tests for the pure-numpy MLP committor.

Includes a finite-difference gradient check (the correctness guarantee for the
manual backprop) and an end-to-end recovery of the analytic 1D double-well
committor from semigroup + boundary data.
"""
import numpy as np
import pytest

from cowera_committor.committor_model import MLPCommittor
from cowera_committor.toy_systems import (
    exact_committor_1d,
    sample_semigroup_pairs,
)


def _make_data(rng, F=3):
    A = rng.standard_normal((5, F))
    B = rng.standard_normal((4, F))
    Xt = rng.standard_normal((6, F))
    Xtp = rng.standard_normal((6, F))
    ws = rng.uniform(0.5, 1.5, size=6)
    Xs = rng.standard_normal((3, F))
    nA = rng.integers(0, 5, size=3).astype(float)
    nB = rng.integers(0, 5, size=3).astype(float)
    return {
        'A': A, 'B': B, 'wA': rng.uniform(0.5, 1.5, 5), 'wB': rng.uniform(0.5, 1.5, 4),
        'sem': (Xt, Xtp, ws),
        'shoot': (Xs, nA, nB, rng.uniform(0.5, 1.5, 3)),
    }


def test_gradient_check_finite_difference():
    """Analytic gradients must match finite differences for all loss terms."""
    rng = np.random.default_rng(1)
    model = MLPCommittor(input_dim=3, hidden=(5, 4), seed=2)
    data = _make_data(rng)
    weights = {'boundary': 1.0, 'semigroup': 1.0, 'aimmd': 1.0,
               'lambda_A': 1.0, 'lambda_B': 1.0}

    theta0 = model.get_params_flat().copy()
    analytic = model.grads_flat(data, weights)

    eps = 1e-6
    num = np.zeros_like(theta0)
    # check a random subset of parameters for speed
    idxs = rng.choice(len(theta0), size=min(40, len(theta0)), replace=False)
    for i in idxs:
        tp = theta0.copy(); tp[i] += eps
        model.set_params_flat(tp)
        lp, _ = model.loss_and_grads(data, weights)
        tm = theta0.copy(); tm[i] -= eps
        model.set_params_flat(tm)
        lm, _ = model.loss_and_grads(data, weights)
        num[i] = (lp - lm) / (2 * eps)
    model.set_params_flat(theta0)

    assert np.allclose(analytic[idxs], num[idxs], rtol=1e-4, atol=1e-6)


def test_loss_decreases_on_fit():
    rng = np.random.default_rng(3)
    model = MLPCommittor(input_dim=3, hidden=(8,), seed=4)
    data = _make_data(rng)
    pool = np.concatenate([data['A'], data['B'], data['sem'][0]], axis=0)
    hist = model.fit(data, weights={'aimmd': 1.0}, epochs=100, lr=1e-2,
                     standardize_from=pool)
    assert hist[-1] < hist[0]


def test_predict_in_unit_interval():
    model = MLPCommittor(input_dim=2, hidden=(4,), seed=0)
    q = model.predict(np.random.default_rng(0).standard_normal((10, 2)))
    assert np.all(q >= 0.0) and np.all(q <= 1.0)
    assert q.shape == (10,)


@pytest.mark.parametrize("beta", [1.0])
def test_double_well_committor_recovery(beta):
    """Train on semigroup + boundary data; recover the analytic committor."""
    rng = np.random.default_rng(7)
    dt, tau_steps = 0.002, 30

    # semigroup pairs from the interior
    Xt, Xtp = sample_semigroup_pairs(2000, tau_steps, dt, beta,
                                     x_range=(-1.4, 1.4), rng=rng)

    # boundary frames: A = {x <= -1}, B = {x >= 1}
    xA = rng.uniform(-1.6, -1.0, size=300)[:, None]
    xB = rng.uniform(1.0, 1.6, size=300)[:, None]

    data = {
        'A': xA, 'B': xB,
        'sem': (Xt, Xtp, np.ones(len(Xt))),
    }
    pool = np.concatenate([xA, xB, Xt, Xtp], axis=0)

    model = MLPCommittor(input_dim=1, hidden=(32, 32), seed=11, l2=1e-5)
    model.fit(data, weights={'boundary': 1.0, 'semigroup': 1.0,
                             'lambda_A': 5.0, 'lambda_B': 5.0},
              epochs=600, lr=5e-3, standardize_from=pool)

    grid = np.linspace(-1.0, 1.0, 101)
    q_pred = model.predict(grid[:, None])
    q_true = exact_committor_1d(grid, beta)

    mae = np.mean(np.abs(q_pred - q_true))
    # symmetry: q at the barrier (x=0) should be ~0.5
    q0 = model.predict(np.array([[0.0]]))[0]

    assert mae < 0.06, f"committor MAE too high: {mae}"
    assert abs(q0 - 0.5) < 0.1, f"q(0) = {q0}, expected ~0.5"
    # strong monotonic agreement with the true committor
    assert np.corrcoef(q_pred, q_true)[0, 1] > 0.99
