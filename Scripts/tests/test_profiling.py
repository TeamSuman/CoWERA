"""Unit tests for the A0 profiling harness (timing reporter + summarizer).

Pure bookkeeping -- no MD stack. Validates the per-cycle bucket math, the CSV
row the TimingReporter writes, and the bottleneck-ranking summary.
"""
import csv
import types

import pytest

from cowera.timing_reporter import TimingReporter, _worker_seg_stats, FIELDNAMES
from tools.bench_run import summarize_profile


# ---------------------------- _worker_seg_stats ----------------------------- #

def test_worker_seg_stats_empty():
    assert _worker_seg_stats(None) == (0.0, 0.0, 0.0)
    assert _worker_seg_stats({}) == (0.0, 0.0, 0.0)


def test_worker_seg_stats_values():
    # worker 0 ran two walkers (0.4, 0.3); worker 1 ran one (0.35)
    seg_sum, seg_max, busy_max = _worker_seg_stats({0: [0.4, 0.3], 1: [0.35]})
    assert round(seg_sum, 3) == 1.05     # aggregate GPU-seconds
    assert round(seg_max, 3) == 0.4      # busiest single walker segment
    assert round(busy_max, 3) == 0.7     # busiest worker's total


# ------------------------------ TimingReporter ------------------------------ #

def test_timing_reporter_writes_expected_row(tmp_path):
    path = tmp_path / "profile.csv"
    resampler = types.SimpleNamespace(_last_subtimings={
        "images_time": 0.10, "distmat_time": 0.20, "intensity_time": 0.30})
    rep = TimingReporter(save_path=str(path))
    rep.init(resampler=resampler)
    rep.report(cycle_idx=0, n_segment_steps=1000,
               cycle_runner_time=1.0, cycle_bc_time=0.1, cycle_resampling_time=0.5,
               worker_segment_times={0: [0.4, 0.3], 1: [0.35]})
    rep.cleanup()

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert list(row.keys()) == FIELDNAMES
    assert float(row["cycle_wall"]) == pytest.approx(1.6)      # 1.0 + 0.1 + 0.5
    assert float(row["mapper_overhead"]) == pytest.approx(0.3) # runner_time - busiest worker total
    assert float(row["intensity_time"]) == pytest.approx(0.30)
    assert int(row["n_segment_steps"]) == 1000


def test_timing_reporter_header_once_then_appends(tmp_path):
    path = tmp_path / "profile.csv"
    resampler = types.SimpleNamespace(_last_subtimings={})
    for cyc in range(2):
        rep = TimingReporter(save_path=str(path))
        rep.init(resampler=resampler)  # simulates a restart re-opening the file
        rep.report(cycle_idx=cyc, cycle_runner_time=1.0, cycle_bc_time=0.0,
                   cycle_resampling_time=0.0, worker_segment_times={0: [0.9]})
        rep.cleanup()
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    # one header + two data rows (header not duplicated on the second open)
    assert lines[0].startswith("cycle_idx")
    assert len(lines) == 3


# ------------------------------ summarize_profile --------------------------- #

def test_summarize_ranks_buckets_by_time():
    rows = [
        {"cycle_wall": "1.0", "runner_time": "0.2", "bc_time": "0.1",
         "resampling_time": "0.6", "mapper_overhead": "0.1",
         "images_time": "0.05", "distmat_time": "0.15", "intensity_time": "0.4"},
        {"cycle_wall": "1.0", "runner_time": "0.2", "bc_time": "0.1",
         "resampling_time": "0.6", "mapper_overhead": "0.1",
         "images_time": "0.05", "distmat_time": "0.15", "intensity_time": "0.4"},
    ]
    s = summarize_profile(rows)
    assert s["cycles"] == 2
    # resampling dominates -> ranked first
    assert s["buckets"][0][0] == "resampling (total)"
    # percentages are relative to mean cycle wall (1.0 s)
    top_label, top_mean, top_pct = s["buckets"][0]
    assert round(top_mean, 3) == 0.6
    assert round(top_pct, 1) == 60.0


def test_summarize_empty_is_safe():
    assert summarize_profile([])["cycles"] == 0
