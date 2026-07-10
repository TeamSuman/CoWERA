"""Unit tests for the pure-numpy algorithm cores in ``cowera.metric``.

These exercise the coherence/intensity machinery (paper Eqs. 3-6, Appendices
B-F) directly, with no dependency on OpenMM, mdtraj, MDAnalysis or ruptures,
and in particular pin down the edge-case behavior of the bug fixes:

* operator-precedence fix in the relevant-history window,
* divide-by-zero guards in the phase / weight / intensity scaling.
"""
import numpy as np
import pytest

from cowera.metric import (
    phase_from_bins,
    scale_phases,
    scale_weights,
    compute_intensity,
    relevant_change_points,
    euclidean_matrix_from_projections,
)


# ----------------- euclidean_matrix_from_projections (P3a) ------------------ #

def _naive_distmat(projections):
    """Reference: the pre-P3a per-pair norm the vectorized helper replaces."""
    n = len(projections)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = np.linalg.norm(np.asarray(projections[i], dtype=float)
                                     - np.asarray(projections[j], dtype=float))
    return M


def test_euclidean_matrix_scalar_projections_matches_naive():
    projs = [np.array([0.5]), np.array([0.2]), np.array([0.9]), np.array([0.55])]
    M = euclidean_matrix_from_projections(projs)
    assert np.allclose(M, _naive_distmat(projs))   # exact equivalence to old path
    assert M[0, 1] == pytest.approx(0.3)           # |0.5 - 0.2|
    assert np.allclose(np.diag(M), 0.0)            # self-distance
    assert np.allclose(M, M.T)                     # symmetric


def test_euclidean_matrix_multidim_projections_matches_naive():
    # multi-D CV projections -> Euclidean norm over the CV dimensions
    projs = [np.array([0.1, 0.2]), np.array([0.4, 0.6]), np.array([0.0, 1.0])]
    M = euclidean_matrix_from_projections(projs)
    assert np.allclose(M, _naive_distmat(projs))
    assert M[0, 1] == pytest.approx(0.5)           # sqrt(0.3^2 + 0.4^2)


def test_euclidean_matrix_plain_scalars():
    # accepts a flat sequence of scalars too (ndim==1 path)
    M = euclidean_matrix_from_projections([0.5, 0.2, 0.9])
    assert M[0, 2] == pytest.approx(0.4)
    assert np.allclose(np.diag(M), 0.0)


def test_calculate_distances_deepcopy_excludes_mda_cache():
    # Reproduces the Trp-cage end-of-run crash: a non-deep-copyable MDAnalysis
    # Universe cached on the distance object must be dropped from copy/pickle.
    import copy
    from cowera.metric import Calculate_Distances

    class _Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("MDAnalysis-Universe-like: not deep-copyable")

    cd = Calculate_Distances(feat="rmsd_backbone", increment=1,
                             distance_criterion="pairwise_rmsd")
    cd._mda_ref = _Uncopyable()
    cd._mda_backbone = [1, 2, 3]
    dup = copy.deepcopy(cd)                  # would raise if the cache weren't excluded
    assert dup._mda_ref is None              # lazily rebuilt on next use
    assert dup._mda_backbone is None
    assert dup.distance_criterion == "pairwise_rmsd"   # everything else copied


def test_changepoint_window_plumbing_and_short_signal_guard():
    from cowera.metric import Calculate_Distances
    # default is unbounded/exact; the option is stored
    assert Calculate_Distances(feat="best_hummer_q", increment=1).changepoint_window is None
    cd = Calculate_Distances(feat="best_hummer_q", increment=1, changepoint_window=200)
    assert cd.changepoint_window == 200
    # _detect_changes degrades to a single segment on a too-short signal (no ruptures
    # needed, no crash) -> the last index only
    assert list(cd._detect_changes(np.array([0.5, 0.6, 0.7]))) == [3]
    assert list(cd._detect_changes(np.array([0.5]))) == [1]


def test_adaptive_changepoint_window_finds_stable_last_changepoint(monkeypatch):
    from cowera.metric import Calculate_Distances
    # Fake detector: a single true changepoint 40 frames from the end. Small windows
    # (<40) see no interior changepoint -> must expand; once the window contains it,
    # the position is stable across a doubling -> accepted. Validates the loop logic
    # (expansion + stability) without ruptures.
    def fake_detect(projection):
        n = len(projection)
        cp = n - 40                       # 40 frames from the end
        return np.array([cp, n]) if cp >= 2 else np.array([n])
    cd = Calculate_Distances(feat="best_hummer_q", increment=1, changepoint_window=8)
    monkeypatch.setattr(cd, "_detect_changes", staticmethod(fake_detect))
    proj = np.linspace(0.2, 0.8, 500)
    drange = np.array([0.2, 0.8])
    bins, phase, weight = cd.phase_from_projection(proj, drange, n_d=5, n_bins=122)
    assert weight == pytest.approx(proj[-1])        # tail (current CV) preserved
    assert np.isfinite(phase)


# ---------------------- A2: incremental CV read dispatch --------------------- #

def _cd(**kw):
    from cowera.metric import Calculate_Distances
    return Calculate_Distances(feat="best_hummer_q", increment=1, **kw)


def test_a2_default_is_incremental():
    # A2 verified exact on both feature paths -> incremental is the default.
    assert _cd().incremental_cv_read is True


def test_a2_disabled_uses_full_load():
    cd = _cd(incremental_cv_read=False)           # explicit opt-out -> exact full load
    cd._project_new_incremental = lambda i, p, o: (_ for _ in ()).throw(AssertionError("nope"))
    cd._project_new_full = lambda i, p, o: (np.array([1.0]), np.array([0, 1]), 10, False)
    assert cd.project_new_frames(0, "x/", 5)[0][0] == 1.0


def test_a2_uses_incremental_when_enabled():
    cd = _cd(incremental_cv_read=True)
    cd._project_new_incremental = lambda i, p, o: (np.array([2.0]), np.array([0, 1]), 10, False)
    cd._project_new_full = lambda i, p, o: (np.array([9.0]), np.array([0, 1]), 10, False)
    assert cd.project_new_frames(0, "x/", 5)[0][0] == 2.0


def test_a2_falls_back_to_full_on_error():
    cd = _cd(incremental_cv_read=True)
    def boom(i, p, o):
        raise RuntimeError("mdtraj boom")
    cd._project_new_incremental = boom
    cd._project_new_full = lambda i, p, o: (np.array([9.0]), np.array([0, 1]), 10, False)
    assert cd.project_new_frames(0, "x/", 5)[0][0] == 9.0        # correctness preserved


def test_a2_verify_mismatch_returns_full():
    cd = _cd(incremental_cv_read=True, verify_incremental_cv=True)
    cd._project_new_incremental = lambda i, p, o: (np.array([0.1, 0.2]), np.array([0, 1]), 10, False)
    cd._project_new_full = lambda i, p, o: (np.array([0.1, 0.9]), np.array([0, 1]), 10, False)
    assert np.allclose(cd.project_new_frames(0, "x/", 5)[0], [0.1, 0.9])   # full (ref) wins


def test_a2_verify_match_returns_incremental():
    cd = _cd(incremental_cv_read=True, verify_incremental_cv=True)
    cd._project_new_incremental = lambda i, p, o: (np.array([0.1, 0.2]), np.array([0, 1]), 10, False)
    cd._project_new_full = lambda i, p, o: (np.array([0.1, 0.2]), np.array([0, 1]), 10, False)
    assert np.allclose(cd.project_new_frames(0, "x/", 5)[0], [0.1, 0.2])


# --------------------------- phase_from_bins -------------------------------- #

def test_phase_monotonic_forward():
    assert phase_from_bins([1, 2, 3, 4]) == pytest.approx(1.0)


def test_phase_monotonic_backward():
    assert phase_from_bins([4, 3, 2, 1]) == pytest.approx(-1.0)


def test_phase_stalled_is_zero():
    assert phase_from_bins([2, 2, 2, 2]) == pytest.approx(0.0)


def test_phase_mixed():
    # diffs: +1, +1, -1 -> signs +1,+1,-1 -> mean = 1/3
    assert phase_from_bins([1, 2, 3, 2]) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("seg", [[], [5]])
def test_phase_empty_or_single_is_zero_not_nan(seg):
    """Bug fix: previously ``np.diff`` of an empty/length-1 segment -> NaN."""
    val = phase_from_bins(seg)
    assert val == 0.0
    assert not np.isnan(val)


# ----------------------------- scale_phases --------------------------------- #

def test_scale_phases_spans_range():
    out = scale_phases([-1.0, 0.0, 1.0])
    assert out == pytest.approx([-1.0, 0.0, 1.0])


def test_scale_phases_maps_to_unit_interval():
    out = scale_phases([0.0, 0.5, 1.0])
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(1.0)


def test_scale_phases_degenerate_no_nan():
    """Bug fix: all-equal phases previously produced 0/0 = NaN."""
    out = scale_phases([0.3, 0.3, 0.3, 0.3])
    assert np.allclose(out, 0.0)
    assert not np.any(np.isnan(out))


# ----------------------------- scale_weights -------------------------------- #

def test_scale_weights_forward():
    out = scale_weights([0.0, 1.0, 2.0], increment=1)
    assert out == pytest.approx([0.0, 0.5, 1.0])


def test_scale_weights_reverse_direction():
    out = scale_weights([0.0, 1.0, 2.0], increment=-1)
    assert out == pytest.approx([1.0, 0.5, 0.0])


def test_scale_weights_degenerate_no_nan():
    out = scale_weights([2.0, 2.0, 2.0], increment=1)
    assert np.allclose(out, 0.5)
    assert not np.any(np.isnan(out))


# --------------------------- compute_intensity ------------------------------ #

def test_compute_intensity_normalized_to_max_one():
    intensity = compute_intensity(
        weights_scaled=np.array([0.2, 0.5, 1.0]),
        phases_scaled=np.array([-1.0, 0.0, 1.0]),
    )
    assert intensity.max() == pytest.approx(1.0)
    assert np.all(intensity >= 0.0)


def test_compute_intensity_all_zero_interference_is_uniform():
    """Bug fix: zero interference previously divided by zero -> NaN/inf."""
    intensity = compute_intensity(
        weights_scaled=np.zeros(4),
        phases_scaled=np.full(4, -1.0),  # sqrt(1 + (-1)) = 0
    )
    assert np.allclose(intensity, 0.25)
    assert not np.any(np.isnan(intensity))


# ------------------------ relevant_change_points ---------------------------- #

def test_relevant_change_points_with_buffer():
    # all interior changepoints minus n_d remain positive -> buffered form
    out = relevant_change_points(changes=[5, 10, 20], n_d=2)
    assert list(out) == [0, 3, 8, 20]


def test_relevant_change_points_single_changepoint():
    out = relevant_change_points(changes=[7], n_d=2)
    assert list(out) == [0, 7]


def test_relevant_change_points_precedence_fix():
    """Demonstrates the operator-precedence bug fix.

    With changes=[1, 10] and n_d=5 the interior changepoint shifted back by n_d
    is negative, so the fallback (no buffer) branch must be taken. The old code
    ``np.all(changes[:-1] - n_d) > 0`` evaluated ``np.all(...)`` to a bool first
    and compared that to 0, taking the wrong branch.
    """
    out = relevant_change_points(changes=[1, 10], n_d=5)
    assert list(out) == [0, 1, 10]
