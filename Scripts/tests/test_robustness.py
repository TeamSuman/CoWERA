"""Unit tests for Phase 1.4 robustness/reproducibility helpers.

Covers the scheduler-aware joblib worker-count resolution and the feature-read
fallback counter. Pure logic -- no MD stack required (mdtraj is imported lazily
inside the feature functions, so the counter itself is importable).
"""
import numpy as np
import pytest

from cowera.metric import _resolve_n_jobs
import cowera.features as features


# ------------------------- _resolve_n_jobs (B5) ----------------------------- #

def test_resolve_n_jobs_explicit_value_wins(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    # an explicit positive n_jobs is respected as-is
    assert _resolve_n_jobs(4) == 4


def test_resolve_n_jobs_uses_slurm_allocation(monkeypatch):
    monkeypatch.delenv("PBS_NP", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    # auto (-1) should be replaced by the allocation, not "all cores"
    assert _resolve_n_jobs(-1) == 12
    assert _resolve_n_jobs(None) == 12


def test_resolve_n_jobs_falls_back_to_all_cores_off_scheduler(monkeypatch):
    for var in ("SLURM_CPUS_PER_TASK", "PBS_NP", "OMP_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    assert _resolve_n_jobs(-1) == -1


def test_resolve_n_jobs_ignores_bad_env(monkeypatch):
    for var in ("SLURM_CPUS_PER_TASK", "PBS_NP", "OMP_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    assert _resolve_n_jobs(-1) == -1


# ------------------------ feature fallback counter (A4) --------------------- #

def test_fallback_counter_increments_and_reraises_without_init(monkeypatch):
    before = features.get_fallback_count()
    # No init_file -> genuine errors must surface (not silently substituted),
    # and the counter must NOT advance on the re-raise path.
    with pytest.raises(Exception):
        features.best_hummer_q(traj=np.zeros((1, 3)), native_file=None, init_file=None)
    assert features.get_fallback_count() == before
