"""Tests for adaptive-bin factor forwarding (Phase 2.2 / review A5, D-1).

The in-memory analysis path previously dropped the bin_increase/decrease factors,
always using the intensity_from_phases defaults. These tests pin down that the
factors now actually change the adaptive bin count. Pure numpy (no ruptures/mdtraj:
intensity_from_phases operates on already-computed bins/phases/weights).
"""
from cowera.metric import Calculate_Distances


def _cd():
    # __init__ only stores attributes; no heavy imports triggered.
    return Calculate_Distances(feat="best_hummer_q", increment=1)


def test_increase_factor_changes_n_bins():
    cd = _cd()
    # every walker sits in one bin over 6 frames -> <20% unique -> increase branch
    bins_list = [[5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5]]
    phase_arr = [0.0, 0.0]
    weight_arr = [0.4, 0.6]
    _, n_low = cd.intensity_from_phases(bins_list, phase_arr, weight_arr, 2, 0,
                                        n_bins=100, max_bins=200,
                                        bin_increase_factor=1.2, bin_decrease_factor=0.8)
    _, n_high = cd.intensity_from_phases(bins_list, phase_arr, weight_arr, 2, 0,
                                         n_bins=100, max_bins=200,
                                         bin_increase_factor=1.5, bin_decrease_factor=0.8)
    assert n_low == int((100 - 1) * 1.2)   # 118
    assert n_high == int((100 - 1) * 1.5)  # 148
    assert n_high > n_low


def test_decrease_factor_changes_n_bins():
    cd = _cd()
    # every frame a distinct bin -> >80% unique -> decrease branch
    bins_list = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    phase_arr = [0.0, 0.0]
    weight_arr = [0.4, 0.6]
    _, n_a = cd.intensity_from_phases(bins_list, phase_arr, weight_arr, 2, 0,
                                      n_bins=100, max_bins=200,
                                      bin_increase_factor=1.2, bin_decrease_factor=0.8)
    _, n_b = cd.intensity_from_phases(bins_list, phase_arr, weight_arr, 2, 0,
                                      n_bins=100, max_bins=200,
                                      bin_increase_factor=1.2, bin_decrease_factor=0.75)
    assert n_a == int((100 - 1) * 0.8)    # 79
    assert n_b == int((100 - 1) * 0.75)   # 74
    assert n_b < n_a
