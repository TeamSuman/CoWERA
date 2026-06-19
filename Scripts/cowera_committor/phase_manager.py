"""Cold-start bootstrap phase manager for CoWERA-Committor.

Training a committor that is useful for resampling is a chicken-and-egg problem:
a committor trained only on A/B end states has a flat, useless gradient in the
transition region (so resampling degenerates to random), yet generating
transition-region data efficiently needs good committor-guided resampling. The
``BootstrapPhaseManager`` breaks this loop with a self-consistent, four-phase
bootstrap, and crucially controls **how much CoWERA trusts the committor** at
each stage.

Phases (doc: CoWERA-Committor cold-start protocol):

* **0 STRUCTURAL** -- pre-simulation: structural A<->B interpolants (soft labels)
  give a coarse ``q^0`` with a non-flat gradient. Exit when the model is smooth
  across the barrier (``max q|_A < 0.3``, ``min q|_B > 0.7``).
* **1 SEMIGROUP** -- early WE: variational/semigroup training on WE segment pairs;
  conservative merges. Exit when the committor varies across intermediates
  (``var(q) > threshold``) with enough intermediate pairs, for N consecutive
  retrains.
* **2 SHOOTING** -- AIMMD shooting-point augmentation. Exit when the committor
  histogram test passes for N consecutive retrains.
* **3 MATURE** -- full committor guidance, normal merges, sliding-window data.

This module is pure-Python/numpy and fully unit-testable; it emits decisions
(loss weights, merge band, retrain timing, data routing, phase advancement) that
the resampler acts on, with no coupling to the MD propagation loop.
"""
from __future__ import annotations

import numpy as np


# Phase constants
STRUCTURAL = 0
SEMIGROUP = 1
SHOOTING = 2
MATURE = 3

PHASE_NAMES = {STRUCTURAL: "structural", SEMIGROUP: "semigroup",
               SHOOTING: "shooting", MATURE: "mature"}


# --------------------------------------------------------------------------- #
# Pure helpers (data routing + quality metrics)
# --------------------------------------------------------------------------- #

def segment_category(q_start, a_cutoff=0.1, b_cutoff=0.9):
    """Route a WE segment to a training buffer by its starting committor value."""
    if q_start < a_cutoff:
        return "boundary_A"
    if q_start > b_cutoff:
        return "boundary_B"
    return "semigroup"


def intermediate_variance(q_values, a_cutoff=0.1, b_cutoff=0.9):
    """Variance of committor values among intermediate-region walkers."""
    q = np.asarray(q_values, dtype=float)
    mask = (q > a_cutoff) & (q < b_cutoff)
    if mask.sum() < 2:
        return 0.0
    return float(np.var(q[mask]))


def histogram_test_pass(p_B, mean_tol=0.1, std_tol=0.15):
    """Committor histogram test: empirical p_B at the TSE should peak at 0.5."""
    p_B = np.asarray(p_B, dtype=float)
    if len(p_B) == 0:
        return False
    return bool(abs(p_B.mean() - 0.5) < mean_tol and p_B.std() < std_tol)


def structural_bootstrap_valid(q_A, q_B, max_qA=0.3, min_qB=0.7):
    """Phase-0 validation: q stays low on A frames and high on B frames."""
    q_A = np.asarray(q_A, dtype=float)
    q_B = np.asarray(q_B, dtype=float)
    if len(q_A) == 0 or len(q_B) == 0:
        return False
    return bool(np.max(q_A) < max_qA and np.min(q_B) > min_qB)


# --------------------------------------------------------------------------- #
# Phase manager
# --------------------------------------------------------------------------- #

# loss-weight presets per phase (consumed by the model trainer)
_LOSS_WEIGHTS = {
    STRUCTURAL: {'boundary': 1.0, 'semigroup': 0.0, 'aimmd': 0.0, 'interpolant': 0.1,
                 'lambda_A': 1.0, 'lambda_B': 1.0},
    SEMIGROUP:  {'boundary': 1.0, 'semigroup': 1.0, 'aimmd': 0.0, 'interpolant': 0.0,
                 'lambda_A': 1.0, 'lambda_B': 1.0},
    SHOOTING:   {'boundary': 1.0, 'semigroup': 0.5, 'aimmd': 2.0, 'interpolant': 0.0,
                 'lambda_A': 1.0, 'lambda_B': 1.0},
    MATURE:     {'boundary': 1.0, 'semigroup': 0.5, 'aimmd': 2.0, 'interpolant': 0.0,
                 'lambda_A': 1.0, 'lambda_B': 1.0},
}

# half-width of the "no-action" band around q=0.5 within which merges are
# suppressed (wide while the model is unreliable, zero once validated)
_MERGE_BAND = {STRUCTURAL: 0.4, SEMIGROUP: 0.3, SHOOTING: 0.15, MATURE: 0.0}


class BootstrapPhaseManager:
    """State machine driving the cold-start committor bootstrap."""

    def __init__(self, warmup_cycles=20, retrain_interval=10,
                 a_cutoff=0.1, b_cutoff=0.9,
                 var_threshold=0.05, min_intermediate_pairs=50,
                 hist_mean_tol=0.1, hist_std_tol=0.15,
                 required_consecutive=3, start_phase=STRUCTURAL):
        self.phase = int(start_phase)
        self.warmup_cycles = int(warmup_cycles)
        self.retrain_interval = int(retrain_interval)
        self.a_cutoff = float(a_cutoff)
        self.b_cutoff = float(b_cutoff)
        self.var_threshold = float(var_threshold)
        self.min_intermediate_pairs = int(min_intermediate_pairs)
        self.hist_mean_tol = float(hist_mean_tol)
        self.hist_std_tol = float(hist_std_tol)
        self.required_consecutive = int(required_consecutive)

        self._consecutive = 0
        self.history = []  # list of recorded metrics dicts (audit trail)

    # ------------------------------------------------------------------ #
    @property
    def current_phase(self):
        return self.phase

    @property
    def phase_name(self):
        return PHASE_NAMES[self.phase]

    @property
    def committor_is_guiding(self):
        """Whether resampling should be fully committor-guided (phase >= 1)."""
        return self.phase >= SEMIGROUP

    def loss_weights(self):
        return dict(_LOSS_WEIGHTS[self.phase])

    def merge_band(self):
        """No-action half-width around q=0.5 (conservative merges early)."""
        return _MERGE_BAND[self.phase]

    def should_retrain(self, cycle):
        return cycle >= self.warmup_cycles and (cycle % self.retrain_interval == 0)

    def segment_category(self, q_start):
        return segment_category(q_start, self.a_cutoff, self.b_cutoff)

    # ------------------------------------------------------------------ #
    def complete_structural_bootstrap(self, q_A, q_B):
        """Advance STRUCTURAL -> SEMIGROUP if the Phase-0 model validates."""
        if self.phase != STRUCTURAL:
            return False
        if structural_bootstrap_valid(q_A, q_B):
            self._advance(SEMIGROUP)
            return True
        return False

    def record_and_maybe_advance(self, metrics):
        """Record post-retrain quality metrics and advance phase if criteria met.

        ``metrics`` may contain:
          'intermediate_variance', 'n_intermediate_pairs' (phase 1 -> 2)
          'p_B'                                            (phase 2 -> 3)
        Returns True if the phase advanced this call.
        """
        self.history.append({'phase': self.phase, **metrics})

        if self.phase == SEMIGROUP:
            ok = (metrics.get('intermediate_variance', 0.0) > self.var_threshold
                  and metrics.get('n_intermediate_pairs', 0) >= self.min_intermediate_pairs)
            return self._tick(ok, SHOOTING)

        if self.phase == SHOOTING:
            ok = histogram_test_pass(metrics.get('p_B', []),
                                     self.hist_mean_tol, self.hist_std_tol)
            return self._tick(ok, MATURE)

        # STRUCTURAL advances via complete_structural_bootstrap; MATURE is terminal
        return False

    # ------------------------------------------------------------------ #
    def _tick(self, ok, next_phase):
        self._consecutive = self._consecutive + 1 if ok else 0
        if self._consecutive >= self.required_consecutive:
            self._advance(next_phase)
            return True
        return False

    def _advance(self, next_phase):
        self.phase = int(next_phase)
        self._consecutive = 0
