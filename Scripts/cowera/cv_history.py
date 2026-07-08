"""In-memory per-walker collective-variable (CV) history for CoWERA.

CoWERA's coherence metric needs each walker's progress-coordinate time series
*up to the current resampling instant* (paper Appendix B/C). The original
implementation re-read every walker's entire accumulated ``walker_{i}.dcd`` from
disk and re-projected the whole trajectory on **every** resampling cycle, which:

* grows without bound (O(T) disk reads, O(T) re-projection, O(T^2) changepoint
  detection) as a walker's history lengthens, and
* couples the resampling analysis to the trajectory-file bookkeeping
  (``file_resampler``), which is fragile under clone/merge/warp.

``CVHistory`` keeps the accumulated per-walker CV series in memory and updates it
incrementally:

* :meth:`extend` appends only the *new* frames produced in the latest segment;
* :meth:`reindex` reorganizes histories under clone/merge decisions by array
  reassignment (mirroring the file-copy semantics of
  ``cowera.file_resampler.update_dcd_files``) instead of touching disk;
* :meth:`reset` clears a walker's history after a warp event.

An optional ``window`` caps the retained history length, bounding the per-cycle
changepoint cost (an algorithmic improvement over unbounded growth).

This module is intentionally free of any heavy dependency (OpenMM / mdtraj /
MDAnalysis / ruptures) so it can be unit tested directly.
"""
from __future__ import annotations

import numpy as np

# CloneMerge decision ids (see wepy.resampling.decisions.clone_merge and the
# mirror logic in cowera.file_resampler.update_dcd_files).
DECISION_NOTHING = 1
DECISION_CLONE = 2
DECISION_KEEP_MERGE = 3
DECISION_SQUASH = 4


def incremental_window(prev_offset, n_frames):
    """Decide which frames of a (possibly reset) trajectory are new.

    Parameters
    ----------
    prev_offset : int
        Number of frames already consumed from this walker's trajectory.
    n_frames : int
        Current number of frames available in the trajectory file.

    Returns
    -------
    was_reset : bool
        True if the trajectory appears to have been reset (e.g. the file was
        truncated/recreated by a warp), i.e. it now has fewer frames than were
        previously consumed.
    start : int
        Index of the first new frame to project.
    new_offset : int
        The offset to store for the next cycle (== ``n_frames``).
    """
    if n_frames < prev_offset:
        return True, 0, n_frames
    return False, prev_offset, n_frames


class CVHistory:
    """Accumulated per-walker CV time series held in memory."""

    def __init__(self, n_walkers, window=None):
        """
        Parameters
        ----------
        n_walkers : int
            Number of walkers (fixed for the simulation).
        window : int or None
            If given, only the most recent ``window`` CV values are retained per
            walker, bounding the changepoint-detection cost.
        """
        self._n = int(n_walkers)
        self.window = window
        self._hist = [np.empty(0, dtype=float) for _ in range(self._n)]
        # frames already consumed from each walker's trajectory source
        self.offsets = np.zeros(self._n, dtype=int)

    # ------------------------------------------------------------------ #
    @property
    def n_walkers(self):
        return self._n

    def get(self, walker_idx):
        """Return the accumulated CV series for a walker (a view's copy)."""
        return self._hist[walker_idx]

    def as_list(self):
        """Return copies of all per-walker CV series."""
        return [h.copy() for h in self._hist]

    def lengths(self):
        return np.array([h.size for h in self._hist], dtype=int)

    # ------------------------------------------------------------------ #
    def _apply_window(self, arr):
        if self.window is not None and arr.size > self.window:
            return arr[-self.window:]
        return arr

    def extend(self, walker_idx, new_values):
        """Append new CV frames to a walker's accumulated history."""
        new_values = np.atleast_1d(np.asarray(new_values, dtype=float))
        if new_values.size == 0:
            return
        arr = np.concatenate([self._hist[walker_idx], new_values])
        self._hist[walker_idx] = self._apply_window(arr)

    def set(self, walker_idx, values):
        """Replace a walker's history outright (e.g. seeding)."""
        arr = np.atleast_1d(np.asarray(values, dtype=float))
        self._hist[walker_idx] = self._apply_window(arr)

    def reset(self, walker_idx):
        """Clear a walker's history (used after a warp event)."""
        self._hist[walker_idx] = np.empty(0, dtype=float)
        self.offsets[walker_idx] = 0

    # ------------------------------------------------------------------ #
    def reindex(self, resampling_records):
        """Reorganize histories according to clone/merge decisions in memory.

        Mirrors ``cowera.file_resampler.update_dcd_files``: a slot's new history
        is the (copied) history of the walker whose content lands in that slot.

        Parameters
        ----------
        resampling_records : list of dict
            Each record must provide ``walker_idx`` (source slot), ``decision_id``
            and ``target_idxs`` (destination slots), as produced by
            ``CoWERAResampler.resample``.
        """
        new_hist = [None] * self._n
        new_offsets = np.zeros(self._n, dtype=int)

        for record in resampling_records:
            src = int(np.asarray(record["walker_idx"]).ravel()[0])
            decision = int(np.asarray(record["decision_id"]).ravel()[0])
            targets = [int(t) for t in np.asarray(record["target_idxs"]).ravel()]

            if decision in (DECISION_NOTHING, DECISION_CLONE):
                for t in targets:
                    new_hist[t] = self._hist[src].copy()
                    new_offsets[t] = self.offsets[src]
            elif decision == DECISION_KEEP_MERGE:
                # the merged-into walker keeps its own history
                dest = targets[0] if targets else src
                new_hist[dest] = self._hist[src].copy()
                new_offsets[dest] = self.offsets[src]
            elif decision == DECISION_SQUASH:
                # walker is discarded into another; nothing to place
                continue

        # Every slot must be filled by exactly one resulting walker.
        missing = [i for i, h in enumerate(new_hist) if h is None]
        if missing:
            raise ValueError(
                f"reindex left walker slots without a history: {missing}. "
                "The resampling records do not cover every output slot."
            )

        self._hist = new_hist
        self.offsets = new_offsets
