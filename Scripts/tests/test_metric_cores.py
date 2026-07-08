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
)


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
