"""Tests for the cold-start BootstrapPhaseManager state machine."""
import numpy as np
import pytest

from cowera_committor.phase_manager import (
    BootstrapPhaseManager,
    segment_category, intermediate_variance, histogram_test_pass,
    structural_bootstrap_valid,
    STRUCTURAL, SEMIGROUP, SHOOTING, MATURE,
)


# ----------------------------- pure helpers --------------------------------- #

@pytest.mark.parametrize("q,expected", [
    (0.05, "boundary_A"), (0.95, "boundary_B"), (0.5, "semigroup"),
    (0.1, "semigroup"), (0.9, "semigroup"),
])
def test_segment_category(q, expected):
    assert segment_category(q) == expected


def test_intermediate_variance_ignores_basins():
    q = np.array([0.01, 0.02, 0.4, 0.5, 0.6, 0.98, 0.99])
    v = intermediate_variance(q)
    assert v == pytest.approx(np.var([0.4, 0.5, 0.6]))


def test_intermediate_variance_too_few_is_zero():
    assert intermediate_variance(np.array([0.01, 0.99])) == 0.0


def test_histogram_test_pass():
    rng = np.random.default_rng(0)
    good = 0.5 + 0.05 * rng.standard_normal(50)
    bad = np.full(50, 0.1)
    assert histogram_test_pass(good)
    assert not histogram_test_pass(bad)
    assert not histogram_test_pass([])


def test_structural_bootstrap_valid():
    assert structural_bootstrap_valid([0.05, 0.1], [0.8, 0.95])
    assert not structural_bootstrap_valid([0.05, 0.4], [0.8, 0.95])  # q_A too high
    assert not structural_bootstrap_valid([], [0.9])


# ------------------------------ state machine ------------------------------- #

def test_starts_in_structural_phase():
    m = BootstrapPhaseManager()
    assert m.current_phase == STRUCTURAL
    assert m.phase_name == "structural"
    assert not m.committor_is_guiding


def test_phase0_advances_on_valid_bootstrap():
    m = BootstrapPhaseManager()
    assert not m.complete_structural_bootstrap([0.4], [0.9])   # invalid
    assert m.current_phase == STRUCTURAL
    assert m.complete_structural_bootstrap([0.05], [0.95])     # valid
    assert m.current_phase == SEMIGROUP
    assert m.committor_is_guiding


def test_loss_weights_and_merge_band_by_phase():
    m = BootstrapPhaseManager()
    assert m.loss_weights()['interpolant'] > 0   # phase 0 uses interpolants
    assert m.loss_weights()['semigroup'] == 0
    band0 = m.merge_band()
    m.complete_structural_bootstrap([0.05], [0.95])  # -> phase 1
    assert m.loss_weights()['semigroup'] == 1.0
    assert m.merge_band() < band0                    # less conservative


def test_should_retrain_respects_warmup_and_interval():
    m = BootstrapPhaseManager(warmup_cycles=20, retrain_interval=10)
    assert not m.should_retrain(10)   # before warmup
    assert m.should_retrain(20)
    assert not m.should_retrain(25)
    assert m.should_retrain(30)


def test_phase1_advances_after_consecutive_passes():
    m = BootstrapPhaseManager(var_threshold=0.05, min_intermediate_pairs=10,
                              required_consecutive=3)
    m.complete_structural_bootstrap([0.05], [0.95])  # -> SEMIGROUP
    good = {'intermediate_variance': 0.2, 'n_intermediate_pairs': 100}
    assert not m.record_and_maybe_advance(good)  # 1
    assert not m.record_and_maybe_advance(good)  # 2
    assert m.record_and_maybe_advance(good)      # 3 -> advance
    assert m.current_phase == SHOOTING


def test_phase1_consecutive_counter_resets_on_failure():
    m = BootstrapPhaseManager(var_threshold=0.05, min_intermediate_pairs=10,
                              required_consecutive=3)
    m.complete_structural_bootstrap([0.05], [0.95])
    good = {'intermediate_variance': 0.2, 'n_intermediate_pairs': 100}
    bad = {'intermediate_variance': 0.0, 'n_intermediate_pairs': 100}
    m.record_and_maybe_advance(good)
    m.record_and_maybe_advance(bad)   # resets
    m.record_and_maybe_advance(good)
    m.record_and_maybe_advance(good)
    assert m.current_phase == SEMIGROUP  # only 2 consecutive -> not advanced yet


def test_phase2_advances_on_histogram_test():
    rng = np.random.default_rng(1)
    m = BootstrapPhaseManager(required_consecutive=2, start_phase=SHOOTING)
    pB = {'p_B': 0.5 + 0.05 * rng.standard_normal(40)}
    assert not m.record_and_maybe_advance(pB)
    assert m.record_and_maybe_advance(pB)
    assert m.current_phase == MATURE


def test_history_audit_trail():
    m = BootstrapPhaseManager(start_phase=SEMIGROUP)
    m.record_and_maybe_advance({'intermediate_variance': 0.1, 'n_intermediate_pairs': 5})
    assert len(m.history) == 1
    assert m.history[0]['phase'] == SEMIGROUP
