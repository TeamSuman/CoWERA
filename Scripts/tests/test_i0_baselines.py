"""Tests for the config-exposed I0 / phase variants (Phase 3.3).

Verify that the paper's uniform-I0 option and the phase-omitted targeted-WE
baseline are reachable through scale_weights / compute_intensity. Pure numpy.
"""
import numpy as np
import pytest

from cowera.metric import scale_weights, compute_intensity


# ------------------------------- scale_weights ------------------------------ #

def test_i0_uniform_is_all_ones():
    out = scale_weights([0.1, 0.5, 0.9], increment=1, i0_mode="uniform")
    assert np.allclose(out, 1.0)


def test_i0_projection_still_default():
    # projection form unchanged from before (min-max, directed by increment)
    out = scale_weights([0.0, 1.0, 2.0], increment=1)  # default i0_mode
    assert out == pytest.approx([0.0, 0.5, 1.0])


def test_i0_mode_invalid_raises():
    with pytest.raises(ValueError):
        scale_weights([0.1, 0.2], increment=1, i0_mode="bogus")


# ----------------------------- compute_intensity ---------------------------- #

def test_use_phase_false_reduces_to_scaled_i0():
    weights_scaled = np.array([0.2, 0.5, 1.0])
    phases_scaled = np.array([-1.0, 0.0, 1.0])
    # targeted-WE baseline: intensity is just I0 normalized to its max, phase ignored
    out = compute_intensity(weights_scaled, phases_scaled, use_phase=False)
    assert out == pytest.approx([0.2, 0.5, 1.0])


def test_use_phase_true_incorporates_phase():
    weights_scaled = np.array([1.0, 1.0])
    phases_scaled = np.array([1.0, -1.0])  # sqrt(2) vs sqrt(0)=0
    out = compute_intensity(weights_scaled, phases_scaled, use_phase=True)
    # phase changes the ranking: walker 0 dominant, walker 1 -> 0
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)


def test_uniform_i0_plus_no_phase_is_flat():
    # uniform I0 + phase omitted -> every walker equal (degenerate targeting)
    w = scale_weights([0.1, 0.7, 0.3], increment=-1, i0_mode="uniform")
    out = compute_intensity(w, np.zeros(3), use_phase=False)
    assert np.allclose(out, 1.0)
