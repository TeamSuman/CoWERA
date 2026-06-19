"""WE-weighted training buffers for the committor cold-start protocol.

Three buffers mirror the data types in the cold-start protocol:

* :class:`BoundaryBuffer`  -- hard A/B boundary frames (retained permanently;
  they never go stale, per doc 1 Phase 3).
* :class:`SemigroupBuffer` -- intermediate-region trajectory pairs
  ``(x_t, x_{t+tau})`` for the variational loss (sliding window).
* :class:`ShootingBuffer`  -- AIMMD shooting outcomes ``(x, n_A, n_B)``
  (sliding window).

Every entry carries a WE walker weight ``w`` so the training loss can apply
inverse-probability weighting. All buffers are pure-numpy and unit-testable.
"""
from __future__ import annotations

import numpy as np


class BoundaryBuffer:
    """Permanent store of boundary frames and their WE weights."""

    def __init__(self):
        self._X = []
        self._w = []

    def add(self, features, weights=None):
        features = np.atleast_2d(np.asarray(features, dtype=float))
        if len(features) == 0:
            return
        if weights is None:
            weights = np.ones(len(features))
        weights = np.atleast_1d(np.asarray(weights, dtype=float))
        self._X.append(features)
        self._w.append(weights)

    def arrays(self):
        if not self._X:
            return np.empty((0, 0)), np.empty(0)
        return np.concatenate(self._X, axis=0), np.concatenate(self._w)

    def __len__(self):
        return int(sum(len(x) for x in self._X))


class SemigroupBuffer:
    """Sliding-window store of (x_t, x_{t+tau}, w) trajectory pairs."""

    def __init__(self, max_size=10000):
        self.max_size = int(max_size)
        self._Xt = None
        self._Xtp = None
        self._w = None

    def add(self, x_t, x_tp, weights=None):
        x_t = np.atleast_2d(np.asarray(x_t, dtype=float))
        x_tp = np.atleast_2d(np.asarray(x_tp, dtype=float))
        if len(x_t) == 0:
            return
        if x_t.shape != x_tp.shape:
            raise ValueError("x_t and x_tp must have the same shape")
        if weights is None:
            weights = np.ones(len(x_t))
        weights = np.atleast_1d(np.asarray(weights, dtype=float))

        if self._Xt is None:
            self._Xt, self._Xtp, self._w = x_t, x_tp, weights
        else:
            self._Xt = np.concatenate([self._Xt, x_t], axis=0)
            self._Xtp = np.concatenate([self._Xtp, x_tp], axis=0)
            self._w = np.concatenate([self._w, weights])

        # enforce sliding window (keep most recent)
        if len(self._Xt) > self.max_size:
            self._Xt = self._Xt[-self.max_size:]
            self._Xtp = self._Xtp[-self.max_size:]
            self._w = self._w[-self.max_size:]

    def arrays(self):
        if self._Xt is None:
            return np.empty((0, 0)), np.empty((0, 0)), np.empty(0)
        return self._Xt, self._Xtp, self._w

    def __len__(self):
        return 0 if self._Xt is None else len(self._Xt)


class ShootingBuffer:
    """Sliding-window store of AIMMD outcomes (x, n_A, n_B, w)."""

    def __init__(self, max_size=2000):
        self.max_size = int(max_size)
        self._X = None
        self._nA = None
        self._nB = None
        self._w = None

    def add(self, features, n_A, n_B, weights=None):
        features = np.atleast_2d(np.asarray(features, dtype=float))
        if len(features) == 0:
            return
        n_A = np.atleast_1d(np.asarray(n_A, dtype=float))
        n_B = np.atleast_1d(np.asarray(n_B, dtype=float))
        if weights is None:
            weights = np.ones(len(features))
        weights = np.atleast_1d(np.asarray(weights, dtype=float))

        if self._X is None:
            self._X, self._nA, self._nB, self._w = features, n_A, n_B, weights
        else:
            self._X = np.concatenate([self._X, features], axis=0)
            self._nA = np.concatenate([self._nA, n_A])
            self._nB = np.concatenate([self._nB, n_B])
            self._w = np.concatenate([self._w, weights])

        if len(self._X) > self.max_size:
            self._X = self._X[-self.max_size:]
            self._nA = self._nA[-self.max_size:]
            self._nB = self._nB[-self.max_size:]
            self._w = self._w[-self.max_size:]

    def arrays(self):
        if self._X is None:
            return np.empty((0, 0)), np.empty(0), np.empty(0), np.empty(0)
        return self._X, self._nA, self._nB, self._w

    def __len__(self):
        return 0 if self._X is None else len(self._X)
