"""Committor-driven distance/intensity calculator for CoWERA.

``CommittorDistances`` subclasses :class:`cowera.metric.Calculate_Distances` and
swaps the geometric progress coordinate (Q / RMSD) for a learned committor
``q(x)``. Because the Phase-2 refactor decoupled the resampling *analysis* from
the projection *source*, this is a drop-in: a ``CommittorDistances`` instance can
be passed directly as the ``distance`` object of ``CoWERAResampler`` with **no
change to the resampler**. The committor path:

* projects each walker frame to ``q`` via model inference (``project_new_frames``
  / ``get_proj_coord``);
* replaces the sign-based phase with the **continuous committor displacement**
  ``Δq = q(T) − q(T − τ_i)`` over the relevant history window, then reuses the
  unchanged ``intensity_from_projections`` / ``scale_phases`` / ``compute_intensity``
  machinery (Δq strictly generalizes ``np.sign(np.diff(...))``);
* augments the merge proximity with committor distance
  ``D_ij = α·D_RMSD + (1−α)·|q_i − q_j|`` so kinetically distinct walkers with
  similar RMSD are not merged.

``q`` is defined toward the target state B, so the natural ``increment`` is ``+1``
(higher ``q`` ⇒ closer to target ⇒ higher initial intensity ``I0``).

The committor ``q`` series is stored per-frame in the existing
:class:`cowera.cv_history.CVHistory` exactly as the geometric projection was —
no change to that machinery is required.
"""
from __future__ import annotations

import numpy as np

from cowera.metric import (
    Calculate_Distances,
    detect_changepoints,
    relevant_change_points,
)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable without ruptures / mdtraj / a trained model)
# --------------------------------------------------------------------------- #

def augment_merge_distance(d_rmsd, q, alpha):
    """Combine a geometric distance matrix with committor proximity.

    ``D_ij = α·D_RMSD_ij + (1−α)·|q_i − q_j|`` (two-state). For the multi-state
    vector committor this generalizes to the L1 distance ``‖q_i − q_j‖₁``.
    """
    d_rmsd = np.asarray(d_rmsd, dtype=float)
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        dq = np.abs(q[:, None] - q[None, :])
    else:  # (N, M) vector committor
        dq = np.abs(q[:, None, :] - q[None, :, :]).sum(axis=-1)
    return alpha * d_rmsd + (1.0 - alpha) * dq


def suppress_tse_merges(D, q, band, q_center=0.5):
    """Make merges involving near-transition-state walkers impossible.

    Sets ``D_ij = inf`` whenever walker ``i`` or ``j`` has a committor within
    ``band`` of ``q_center`` (=0.5). This implements the cold-start protocol's
    "conservative merge near q=0.5": while the committor is unreliable, walkers
    near the (precious) transition state are never merged away. ``band == 0``
    leaves ``D`` unchanged.
    """
    D = np.array(D, dtype=float, copy=True)
    if band <= 0:
        return D
    q = np.asarray(q, dtype=float)
    near = np.abs(q - q_center) < band
    if near.any():
        D[near, :] = np.inf
        D[:, near] = np.inf
        np.fill_diagonal(D, 0.0)
    return D


def committor_displacement(q_series, change_points):
    """Continuous committor displacement over the relevant history window.

    Returns ``(delta_q, weight)`` where ``delta_q = q[-1] − q[start]`` with
    ``start`` the most recent relevant changepoint, and ``weight = q[-1]`` (the
    current committor value, used as the per-walker initial intensity ``I0``).
    """
    q = np.asarray(q_series, dtype=float)
    start = int(np.asarray(change_points).ravel()[-2])
    return float(q[-1] - q[start]), float(q[-1])


# --------------------------------------------------------------------------- #
# Committor distance/intensity calculator
# --------------------------------------------------------------------------- #

class CommittorDistances(Calculate_Distances):
    def __init__(self, committor_model, featurizer,
                 increment=1, merge_alpha=0.7, merge_band=0.0,
                 distance_criterion="pairwise_rmsd",
                 changepoint_fn=None, **kwargs):
        """
        Parameters
        ----------
        committor_model : object with ``predict_batch(features) -> (N,)``
        featurizer : object with ``featurize(frames) -> (N, F)``
        merge_alpha : float in [0, 1]
            Weight of geometric RMSD vs. committor proximity in the merge
            distance (1.0 = pure RMSD, 0.0 = pure committor).
        merge_band : float
            No-action half-width around q=0.5; merges involving walkers within
            this band are suppressed (cold-start "conservative merge"). Updated
            per cycle by the phase manager. 0.0 disables.
        changepoint_fn : callable or None
            Override for changepoint detection (defaults to
            ``cowera.metric.detect_changepoints``); injectable for testing.
        """
        super().__init__(feat="committor", increment=increment,
                         distance_criterion=distance_criterion, **kwargs)
        self.model = committor_model
        self.featurizer = featurizer
        self.merge_alpha = float(merge_alpha)
        self.merge_band = float(merge_band)
        self._changepoint_fn = changepoint_fn

    # --------------------------- inference -------------------------------- #
    def predict_q(self, frames):
        """Committor values for a stack of frames (model inference)."""
        feats = self.featurizer.featurize(frames)
        return np.asarray(self.model.predict_batch(feats), dtype=float).ravel()

    def predict_q_state(self, state):
        """Committor value of a single walker state (final frame)."""
        pos = np.asarray(state['positions'], dtype=float)
        frames = pos[None, ...] if pos.ndim == 2 else pos
        return float(self.predict_q(frames)[-1])

    def get_proj_coord(self, state):
        """Image of a walker state = its committor value (array, length 1)."""
        pos = np.asarray(state['positions'], dtype=float)
        frames = pos[None, ...] if pos.ndim == 2 else pos
        return self.predict_q(frames)

    def load_frames(self, i, path):
        """Load a walker's trajectory frames as an (T, n_atoms, 3) array.

        Returns ``None`` if the file is missing (e.g. just after a warp). Used by
        the resampler to harvest training data. Requires mdtraj.
        """
        import mdtraj as md

        top = self.top_file or self.native_file
        try:
            traj = md.load(f"{path}walker_{i}.dcd", top=top)
        except Exception:
            return None
        return traj.xyz

    def _load_projection(self, i, path, frame_slice=None):
        """Load walker frames and project them onto the committor (range [0,1])."""
        import mdtraj as md

        top = self.top_file or self.native_file
        traj = md.load(f"{path}walker_{i}.dcd", top=top)
        if frame_slice is not None:
            traj = traj[frame_slice]
        q = self.predict_q(traj.xyz)
        return q, np.array([0.0, 1.0])

    # ----------------------- committor phase (Δq) ------------------------- #
    def _changepoints(self, q):
        if self._changepoint_fn is not None:
            return np.asarray(self._changepoint_fn(q))
        return detect_changepoints(q)

    def phase_from_projection(self, projection, drange, n_d, n_bins=100):
        """Continuous committor displacement Δq over the relevant history.

        Drop-in replacement for the base sign-based phase: returns
        ``(bins, Δq, weight)`` consumed unchanged by ``intensity_from_phases``.
        """
        q = np.asarray(projection, dtype=float)
        if len(q) <= 1:
            return np.zeros(n_d), 0.0, (float(q[0]) if len(q) else 0.0)

        changes = self._changepoints(q)
        cps = relevant_change_points(changes, n_d)
        delta_q, weight = committor_displacement(q, cps)

        # bins over the committor range [0, 1] feed the adaptive-binning
        # "fraction of unique bins" logic only (CoWERA remains binless).
        bin_edges = self.get_bin_edges(x_min=0.0, x_max=1.0, n_bins=n_bins)
        current_bins = np.digitize(q, bin_edges, right=True)
        return current_bins[-n_d:], delta_q, weight

    # ----------------------- committor merge dist ------------------------- #
    def pairwise_distance_matrix(self, walkers):
        """Committor-augmented merge distance matrix (with optional TSE band)."""
        d_rmsd = super().pairwise_distance_matrix(walkers)
        q = np.array([self.predict_q_state(w.state) for w in walkers])
        if self.merge_alpha < 1.0:
            D = augment_merge_distance(d_rmsd, q, self.merge_alpha)
        else:
            D = d_rmsd
        return suppress_tse_merges(D, q, self.merge_band)
