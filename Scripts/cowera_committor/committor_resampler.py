"""CommittorResampler: CoWERA resampling with on-the-fly committor learning.

This thin subclass of :class:`cowera.resampler.CoWERAResampler` wires together
the committor pieces:

* a :class:`cowera_committor.committor_metric.CommittorDistances` as the
  resampler's ``distance`` (so clone/merge is committor-guided -- see M2);
* a :class:`cowera_committor.phase_manager.BootstrapPhaseManager` driving the
  cold-start phases (loss composition, conservative-merge band, retrain timing,
  phase advancement -- see M3);
* WE-weighted training buffers that accrue boundary frames and semigroup pairs
  from each cycle's trajectories and are used to retrain the committor.

The orchestration here is intentionally minimal; all decision logic lives in the
unit-tested helpers (``phase_manager``, ``committor_train``,
``committor_metric``). The frame-loading / featurization steps require the MD
stack and are therefore validated on the simulation host, not in CPU unit tests.
"""
from __future__ import annotations

import logging

import numpy as np

from cowera.resampler import CoWERAResampler

from cowera_committor.committor_metric import CommittorDistances
from cowera_committor.phase_manager import (
    BootstrapPhaseManager, intermediate_variance,
)
from cowera_committor.committor_data import (
    BoundaryBuffer, SemigroupBuffer, ShootingBuffer,
)
from cowera_committor.committor_train import (
    extract_segment_training_data, assemble_training_data, retrain_committor,
)


class CommittorResampler(CoWERAResampler):
    """Committor-guided CoWERA resampler with cold-start retraining."""

    def __init__(self, committor_model, featurizer,
                 phase_manager=None,
                 merge_alpha=0.7,
                 distance_criterion="pairwise_rmsd",
                 native_file=None, top_file=None,
                 retrain_epochs=200, retrain_lr=5e-3,
                 semigroup_buffer_size=10000, shooting_buffer_size=2000,
                 increment=1, **resampler_kwargs):

        self.manager = phase_manager or BootstrapPhaseManager()

        # the committor acts as the progress coordinate (M2)
        distance = CommittorDistances(
            committor_model=committor_model, featurizer=featurizer,
            increment=increment, merge_alpha=merge_alpha,
            merge_band=self.manager.merge_band(),
            distance_criterion=distance_criterion,
            native_file=native_file, top_file=top_file,
        )
        super().__init__(distance=distance, increment=increment,
                         **resampler_kwargs)

        self.model = committor_model
        self.featurizer = featurizer
        self.retrain_epochs = int(retrain_epochs)
        self.retrain_lr = float(retrain_lr)

        # WE-weighted training buffers
        self.boundary_A = BoundaryBuffer()
        self.boundary_B = BoundaryBuffer()
        self.semigroup = SemigroupBuffer(max_size=semigroup_buffer_size)
        self.shooting = ShootingBuffer(max_size=shooting_buffer_size)

        # latest per-walker committor values (for quality metrics)
        self._last_q = np.array([])

    # ------------------------------------------------------------------ #
    def resample(self, walkers, cycle_id, n_bins, max_bins=125):
        # keep the conservative-merge band in sync with the current phase
        self.distance.merge_band = self.manager.merge_band()

        result = super().resample(walkers, cycle_id, n_bins, max_bins)
        resampled_walkers = result[0]

        # accrue training data from this cycle and retrain when scheduled
        try:
            self._collect_training_data(resampled_walkers)
            if self.manager.should_retrain(cycle_id):
                self._retrain_and_advance()
        except Exception as exc:  # never let training kill a running simulation
            logging.error(f"Committor training step failed at cycle {cycle_id}: {exc}")

        return result

    # ------------------------------------------------------------------ #
    def _collect_training_data(self, walkers):
        """Featurize each walker's frames, classify by q, and fill the buffers.

        Requires the MD stack (loads DCD frames via the committor metric). The
        per-frame routing itself is the unit-tested
        ``extract_segment_training_data``.
        """
        q_now = []
        for i, walker in enumerate(walkers):
            frames = self.distance.load_frames(i, self.dcd_folder)   # (T, n_atoms, 3)
            if frames is None or len(frames) == 0:
                continue
            feats = self.featurizer.featurize(frames)      # (T, F)
            q = self.distance.predict_q(frames)            # (T,)
            q_now.append(q[-1])

            A, B, Xt, Xtp = extract_segment_training_data(
                feats, q, self.manager.a_cutoff, self.manager.b_cutoff)
            w = walker.weight
            self.boundary_A.add(A, weights=np.full(len(A), w))
            self.boundary_B.add(B, weights=np.full(len(B), w))
            self.semigroup.add(Xt, Xtp, weights=np.full(len(Xt), w))

        self._last_q = np.asarray(q_now, dtype=float)

    def _retrain_and_advance(self):
        """Retrain the committor with phase-appropriate weights, then assess phase."""
        data = assemble_training_data(
            self.boundary_A, self.boundary_B, self.semigroup, self.shooting)
        if not data:
            return
        retrain_committor(self.model, data, self.manager,
                          epochs=self.retrain_epochs, lr=self.retrain_lr)

        metrics = {
            'intermediate_variance': intermediate_variance(
                self._last_q, self.manager.a_cutoff, self.manager.b_cutoff),
            'n_intermediate_pairs': len(self.semigroup),
        }
        advanced = self.manager.record_and_maybe_advance(metrics)
        if advanced:
            logging.info(f"Committor bootstrap advanced to phase "
                         f"{self.manager.phase_name}")
