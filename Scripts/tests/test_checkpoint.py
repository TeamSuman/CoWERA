"""Unit tests for the restart checkpoint-discovery logic (Phase 1.2).

These cover the pure filesystem helper that selects which checkpoint a
``--restart`` should resume from. No MD stack required.
"""
import pickle

from walker_pkl_reporter import (
    latest_checkpoint,
    WALKERS_TEMPLATE,
    CHECKPOINT_TEMPLATE,
    _cycle_of,
)


def _write_pair(d, cycle, walkers=True, meta=True):
    if walkers:
        (d / WALKERS_TEMPLATE.format(cycle)).write_bytes(pickle.dumps(["w"]))
    if meta:
        (d / CHECKPOINT_TEMPLATE.format(cycle)).write_bytes(
            pickle.dumps({"cycle_idx": cycle, "n_bins": 40 + cycle}))


def test_cycle_of_parses_and_rejects():
    assert _cycle_of("walkers_cycle_17.pkl", WALKERS_TEMPLATE) == 17
    assert _cycle_of("checkpoint_cycle_3.pkl", CHECKPOINT_TEMPLATE) == 3
    assert _cycle_of("walkers_cycle_x.pkl", WALKERS_TEMPLATE) is None
    assert _cycle_of("something_else.pkl", WALKERS_TEMPLATE) is None
    # a walkers file must not be misread as a checkpoint file
    assert _cycle_of("walkers_cycle_5.pkl", CHECKPOINT_TEMPLATE) is None


def test_latest_checkpoint_none_when_empty(tmp_path):
    assert latest_checkpoint(str(tmp_path)) is None


def test_latest_checkpoint_missing_dir():
    assert latest_checkpoint("/no/such/dir/here") is None


def test_latest_checkpoint_picks_highest_complete(tmp_path):
    _write_pair(tmp_path, 10)
    _write_pair(tmp_path, 20)
    _write_pair(tmp_path, 30)
    walkers_path, meta_path, cycle = latest_checkpoint(str(tmp_path))
    assert cycle == 30
    assert walkers_path.endswith(WALKERS_TEMPLATE.format(30))
    assert meta_path.endswith(CHECKPOINT_TEMPLATE.format(30))
    with open(meta_path, "rb") as f:
        assert pickle.load(f)["cycle_idx"] == 30


def test_latest_checkpoint_ignores_incomplete_pair(tmp_path):
    # cycle 40 has walkers but the metadata sidecar never got written
    # (job killed mid-checkpoint) -> must fall back to the last complete pair.
    _write_pair(tmp_path, 20)
    _write_pair(tmp_path, 40, meta=False)
    walkers_path, meta_path, cycle = latest_checkpoint(str(tmp_path))
    assert cycle == 20
