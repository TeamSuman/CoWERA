"""Tests for the analytic toy systems (double well + exact committor)."""
import numpy as np

from cowera_committor.toy_systems import (
    double_well_potential,
    double_well_force,
    exact_committor_1d,
    overdamped_langevin,
)


def test_double_well_minima_and_force():
    # minima at x = +/-1, barrier at 0
    assert double_well_potential(1.0) == 0.0
    assert double_well_potential(-1.0) == 0.0
    assert double_well_potential(0.0) == 1.0
    # force vanishes at minima and barrier top
    assert np.allclose(double_well_force([-1.0, 0.0, 1.0]), 0.0)


def test_exact_committor_boundary_values_and_symmetry():
    q = exact_committor_1d(np.array([-1.0, 0.0, 1.0]), beta=1.0)
    assert q[0] == 0.0
    assert q[2] == 1.0
    assert abs(q[1] - 0.5) < 1e-6  # symmetric potential -> q(0)=0.5


def test_exact_committor_monotonic():
    grid = np.linspace(-1.0, 1.0, 50)
    q = exact_committor_1d(grid, beta=2.0)
    assert np.all(np.diff(q) >= -1e-12)


def test_langevin_runs_and_shapes():
    rng = np.random.default_rng(0)
    traj = overdamped_langevin(-1.0, n_steps=100, dt=0.002, beta=1.0, rng=rng)
    assert traj.shape == (101,)
    assert np.isfinite(traj).all()
