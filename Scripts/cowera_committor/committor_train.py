"""Training-data assembly and retraining orchestration for CoWERA-Committor.

These helpers turn the WE-weighted buffers into the ``data`` dict consumed by
``MLPCommittor.fit`` and run a phase-aware retraining step. They are pure (model
+ buffers are numpy), so the cold-start training loop can be unit-tested without
any MD/GPU stack.
"""
from __future__ import annotations

import numpy as np


def extract_segment_training_data(features, q, a_cutoff=0.1, b_cutoff=0.9):
    """Route a walker's per-frame features into boundary frames + semigroup pairs.

    Given per-frame features ``(T, F)`` and committor values ``q`` ``(T,)`` for a
    single walker segment, classify each frame by its committor value:

    * ``q < a_cutoff``  -> A-boundary frame (label 0),
    * ``q > b_cutoff``  -> B-boundary frame (label 1),
    * intermediate      -> a semigroup pair ``(x_t, x_{t+1})`` with the next frame.

    Returns ``(A_frames, B_frames, sem_starts, sem_ends)`` as numpy arrays
    (any may be empty). Pure-numpy / unit-testable.
    """
    features = np.atleast_2d(np.asarray(features, dtype=float))
    q = np.asarray(q, dtype=float)
    T = len(q)

    A_idx, B_idx, s_starts, s_ends = [], [], [], []
    for t in range(T):
        if q[t] < a_cutoff:
            A_idx.append(t)
        elif q[t] > b_cutoff:
            B_idx.append(t)
        elif t < T - 1:  # intermediate frame -> semigroup pair with successor
            s_starts.append(t)
            s_ends.append(t + 1)

    A = features[A_idx] if A_idx else np.empty((0, features.shape[1]))
    B = features[B_idx] if B_idx else np.empty((0, features.shape[1]))
    Xt = features[s_starts] if s_starts else np.empty((0, features.shape[1]))
    Xtp = features[s_ends] if s_ends else np.empty((0, features.shape[1]))
    return A, B, Xt, Xtp


def assemble_training_data(boundary_A=None, boundary_B=None, semigroup=None,
                           shooting=None, interp=None):
    """Build the ``data`` dict for ``MLPCommittor.fit`` from buffers.

    Parameters are buffer objects exposing ``.arrays()`` (or None / featurized
    tuples). ``interp`` is an optional ``(X, y, w)`` tuple for Phase-0 soft
    labels.
    """
    data = {}

    if boundary_A is not None and len(boundary_A) > 0:
        XA, wA = boundary_A.arrays()
        data['A'] = XA
        data['wA'] = wA
    if boundary_B is not None and len(boundary_B) > 0:
        XB, wB = boundary_B.arrays()
        data['B'] = XB
        data['wB'] = wB
    if semigroup is not None and len(semigroup) > 0:
        Xt, Xtp, ws = semigroup.arrays()
        data['sem'] = (Xt, Xtp, ws)
    if shooting is not None and len(shooting) > 0:
        Xs, nA, nB, ws = shooting.arrays()
        data['shoot'] = (Xs, nA, nB, ws)
    if interp is not None and len(interp[0]) > 0:
        data['interp'] = interp

    return data


def standardization_pool(data):
    """Collect a representative feature pool from a training-data dict."""
    pools = []
    if 'A' in data:
        pools.append(np.atleast_2d(data['A']))
    if 'B' in data:
        pools.append(np.atleast_2d(data['B']))
    if 'sem' in data:
        pools.append(np.atleast_2d(data['sem'][0]))
        pools.append(np.atleast_2d(data['sem'][1]))
    if 'shoot' in data:
        pools.append(np.atleast_2d(data['shoot'][0]))
    if 'interp' in data:
        pools.append(np.atleast_2d(data['interp'][0]))
    if not pools:
        return None
    return np.concatenate(pools, axis=0)


def retrain_committor(model, data, phase_manager, epochs=200, lr=5e-3,
                      standardize=True, **fit_kwargs):
    """Run one phase-aware retraining step; returns the loss history.

    The loss-term weights are taken from ``phase_manager.loss_weights()`` so the
    composition automatically follows the bootstrap phase (interpolant -> semigroup
    -> AIMMD).
    """
    weights = phase_manager.loss_weights()
    pool = standardization_pool(data) if standardize else None
    return model.fit(data, weights=weights, epochs=epochs, lr=lr,
                     standardize_from=pool, **fit_kwargs)
