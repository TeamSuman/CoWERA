"""Unit tests for the in-memory CV history (Phase 2).

Covers the incremental update window, warp-reset detection, optional bounded
window, and the clone/merge reindexing that mirrors
``cowera.file_resampler.update_dcd_files`` -- all without any MD dependency.
"""
import numpy as np
import pytest

from cowera.cv_history import (
    CVHistory,
    incremental_window,
    DECISION_NOTHING,
    DECISION_CLONE,
    DECISION_KEEP_MERGE,
    DECISION_SQUASH,
)


# --------------------------- incremental_window ----------------------------- #

def test_incremental_window_normal_growth():
    was_reset, start, new_offset = incremental_window(prev_offset=10, n_frames=15)
    assert was_reset is False
    assert start == 10
    assert new_offset == 15


def test_incremental_window_no_new_frames():
    was_reset, start, new_offset = incremental_window(prev_offset=10, n_frames=10)
    assert was_reset is False
    assert start == 10
    assert new_offset == 10


def test_incremental_window_detects_reset():
    # file shrank (e.g. recreated after a warp) -> reprocess from scratch
    was_reset, start, new_offset = incremental_window(prev_offset=20, n_frames=3)
    assert was_reset is True
    assert start == 0
    assert new_offset == 3


# ------------------------------- extend/reset ------------------------------- #

def test_extend_accumulates():
    h = CVHistory(2)
    h.extend(0, [0.1, 0.2])
    h.extend(0, [0.3])
    assert np.allclose(h.get(0), [0.1, 0.2, 0.3])
    assert h.get(1).size == 0


def test_extend_empty_is_noop():
    h = CVHistory(1)
    h.extend(0, [])
    assert h.get(0).size == 0


def test_window_bounds_history():
    h = CVHistory(1, window=3)
    h.extend(0, [1, 2])
    h.extend(0, [3, 4, 5])
    # only the most recent 3 retained
    assert np.allclose(h.get(0), [3, 4, 5])


def test_reset_clears_history_and_offset():
    h = CVHistory(2)
    h.extend(0, [1, 2, 3])
    h.offsets[0] = 3
    h.reset(0)
    assert h.get(0).size == 0
    assert h.offsets[0] == 0


# ------------------------------- reindex ------------------------------------ #

def _rec(walker_idx, decision_id, target_idxs):
    return {
        "walker_idx": np.array([walker_idx]),
        "decision_id": np.array([decision_id]),
        "target_idxs": np.array(target_idxs),
    }


def test_reindex_clone_copies_history_to_targets():
    """Walker 0 clones into slots 0 and 1; walker 1 is squashed into 0."""
    h = CVHistory(2)
    h.set(0, [0.1, 0.2, 0.3])
    h.set(1, [9.0, 9.0])
    h.offsets[:] = [3, 2]

    records = [
        _rec(0, DECISION_CLONE, [0, 1]),  # parent 0 -> slots 0 and 1
        _rec(1, DECISION_SQUASH, [0]),    # walker 1 discarded
    ]
    h.reindex(records)

    assert np.allclose(h.get(0), [0.1, 0.2, 0.3])
    assert np.allclose(h.get(1), [0.1, 0.2, 0.3])  # clone got parent history
    assert list(h.offsets) == [3, 3]


def test_reindex_nothing_keeps_in_place():
    h = CVHistory(2)
    h.set(0, [1.0, 2.0])
    h.set(1, [3.0, 4.0])
    h.offsets[:] = [2, 2]

    records = [
        _rec(0, DECISION_NOTHING, [0]),
        _rec(1, DECISION_NOTHING, [1]),
    ]
    h.reindex(records)
    assert np.allclose(h.get(0), [1.0, 2.0])
    assert np.allclose(h.get(1), [3.0, 4.0])


def test_reindex_keep_merge_retains_merged_walker():
    h = CVHistory(2)
    h.set(0, [5.0, 6.0, 7.0])
    h.set(1, [0.0])
    h.offsets[:] = [3, 1]

    records = [
        _rec(0, DECISION_KEEP_MERGE, [0]),  # walker 0 is the merge survivor
        _rec(1, DECISION_CLONE, [1]),       # something fills slot 1
    ]
    h.reindex(records)
    assert np.allclose(h.get(0), [5.0, 6.0, 7.0])


def test_reindex_raises_if_slot_uncovered():
    h = CVHistory(2)
    h.set(0, [1.0])
    h.set(1, [2.0])
    # only slot 0 is covered -> slot 1 left without a history
    records = [_rec(0, DECISION_NOTHING, [0])]
    with pytest.raises(ValueError):
        h.reindex(records)
