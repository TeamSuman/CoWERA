"""Featurizers mapping molecular configurations to committor-model inputs.

The committor must be invariant to rigid-body rotation and translation
(SE(3)-invariant). Internal pairwise distances are the simplest invariant
representation and are used by :class:`DistanceFeaturizer`. For analytic toy
systems whose coordinates already live in a low-dimensional CV space,
:class:`IdentityFeaturizer` passes coordinates through unchanged.

All featurizers are pure-numpy and produce a ``(n_frames, n_features)`` array,
so they are directly unit-testable without any MD stack.
"""
from __future__ import annotations

import numpy as np


class Featurizer:
    """Interface: map a stack of frames to a feature matrix."""

    @property
    def n_features(self):
        raise NotImplementedError

    def featurize(self, frames):
        """frames: (n_frames, n_atoms, 3) or (n_frames, d) -> (n_frames, n_features)."""
        raise NotImplementedError


class IdentityFeaturizer(Featurizer):
    """Pass coordinates through unchanged (for toy systems in CV space)."""

    def __init__(self, n_features):
        self._n = int(n_features)

    @property
    def n_features(self):
        return self._n

    def featurize(self, frames):
        arr = np.atleast_2d(np.asarray(frames, dtype=float))
        if arr.shape[-1] != self._n:
            raise ValueError(
                f"IdentityFeaturizer expected {self._n} features, got {arr.shape[-1]}")
        return arr


class DistanceFeaturizer(Featurizer):
    """SE(3)-invariant pairwise-distance features over a fixed set of atom pairs.

    Parameters
    ----------
    pairs : array-like of shape (n_pairs, 2)
        Atom-index pairs whose interatomic distances form the feature vector.
        For molecular systems these are typically heavy-atom / native-contact
        pairs (reuse ``cowera.features.get_native_contacts`` to obtain them).
    inverse : bool
        If True, use 1/d features (emphasizes close contacts; bounded for
        contact-formation coordinates). Default False (raw distances, nm).
    """

    def __init__(self, pairs, inverse=False):
        self.pairs = np.asarray(pairs, dtype=int)
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2)")
        self.inverse = bool(inverse)

    @property
    def n_features(self):
        return len(self.pairs)

    def featurize(self, frames):
        frames = np.asarray(frames, dtype=float)
        if frames.ndim == 2:
            # single frame (n_atoms, 3) -> add frame axis
            frames = frames[None, :, :]
        if frames.ndim != 3 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape (n_frames, n_atoms, 3)")

        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        # (n_frames, n_pairs, 3) displacement -> (n_frames, n_pairs) distances
        diff = frames[:, i, :] - frames[:, j, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        if self.inverse:
            return 1.0 / (dist + 1e-9)
        return dist
