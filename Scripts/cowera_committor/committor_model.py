"""Committor models for CoWERA-Committor.

``CommittorModel`` is the stable interface consumed by the resampling
integration; the concrete near-term backend is :class:`MLPCommittor`, a
dependency-free multilayer perceptron with manual backpropagation. A future
SE(3)-equivariant GVP-GNN backend can implement the same interface without
touching any caller.

The model is trained with the weighted variational (Dirichlet / semigroup) loss
plus boundary and (optional) AIMMD shooting-point terms:

    L = w_b * [ lambda_A <w q^2>_A + lambda_B <w (1-q)^2>_B ]      (boundaries)
      + w_s * < w (q(x_t) - q(x_{t+tau}))^2 >                       (semigroup)
      + w_a * (-< w [n_A log(1-q) + n_B log q] >)                  (AIMMD)

Every term carries the WE walker weight ``w`` (inverse-probability weighting),
which corrects the non-equilibrium sampling bias of the WE trajectory pool.

Two-state committors use a sigmoid output (``n_states == 2``). The output head is
shaped so that a softmax multi-state head (``n_states > 2``) drops in later; the
multi-state training path is not implemented in this milestone.
"""
from __future__ import annotations

import numpy as np


class CommittorModel:
    """Interface for committor predictors."""

    n_states = 2

    def predict(self, X):
        """Return committor value(s) for feature matrix X of shape (N, F)."""
        raise NotImplementedError

    def predict_batch(self, X):
        return self.predict(X)

    def fit(self, *args, **kwargs):
        raise NotImplementedError


def _sigmoid(z):
    # numerically stable logistic
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class MLPCommittor(CommittorModel):
    """Pure-numpy MLP committor with manual backprop and an Adam optimizer."""

    def __init__(self, input_dim, hidden=(32, 32), n_states=2, seed=0, l2=0.0):
        if n_states != 2:
            raise NotImplementedError(
                "Multi-state (softmax) head is designed-in but not implemented "
                "in this milestone; use n_states=2.")
        self.input_dim = int(input_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.n_states = 2
        self.l2 = float(l2)

        rng = np.random.default_rng(seed)
        dims = [self.input_dim, *self.hidden, 1]
        self.W = []
        self.b = []
        for din, dout in zip(dims[:-1], dims[1:]):
            # Glorot/Xavier init (tanh-friendly)
            limit = np.sqrt(6.0 / (din + dout))
            self.W.append(rng.uniform(-limit, limit, size=(din, dout)))
            self.b.append(np.zeros(dout))

        # feature standardization (set during fit)
        self._mu = np.zeros(self.input_dim)
        self._sigma = np.ones(self.input_dim)

    # ------------------------------------------------------------------ #
    # forward / backward
    # ------------------------------------------------------------------ #
    def _standardize(self, X):
        return (np.asarray(X, dtype=float) - self._mu) / self._sigma

    def _forward(self, X):
        """Return (q, cache). q has shape (N, 1)."""
        a = self._standardize(X)
        acts = [a]
        pre = []
        for li in range(len(self.W) - 1):
            z = a @ self.W[li] + self.b[li]
            a = np.tanh(z)
            pre.append(z)
            acts.append(a)
        z_out = a @ self.W[-1] + self.b[-1]
        q = _sigmoid(z_out)
        return q, (acts, pre, z_out)

    def _backward(self, cache, dL_dq):
        """Backprop dL/dq (shape (N,1)) -> list of (dW, db) per layer."""
        acts, pre, _ = cache
        q, _ = None, None
        # recompute q for sigmoid derivative
        q = _sigmoid(cache[2])
        dz = dL_dq * q * (1.0 - q)  # through sigmoid

        gW = [None] * len(self.W)
        gb = [None] * len(self.b)

        # output layer
        a_prev = acts[-1]
        gW[-1] = a_prev.T @ dz
        gb[-1] = dz.sum(axis=0)
        da = dz @ self.W[-1].T

        # hidden layers (tanh)
        for li in range(len(self.W) - 2, -1, -1):
            a_li = acts[li + 1]
            dz_h = da * (1.0 - a_li * a_li)  # tanh'
            a_prev = acts[li]
            gW[li] = a_prev.T @ dz_h
            gb[li] = dz_h.sum(axis=0)
            da = dz_h @ self.W[li].T

        return gW, gb

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def predict(self, X):
        q, _ = self._forward(np.atleast_2d(X))
        return q.ravel()

    # ------------------------------------------------------------------ #
    # loss + gradients
    # ------------------------------------------------------------------ #
    def _accumulate(self, gW, gb, gW_add, gb_add, scale=1.0):
        for i in range(len(gW)):
            gW[i] = (gW[i] + scale * gW_add[i]) if gW[i] is not None else scale * gW_add[i]
            gb[i] = (gb[i] + scale * gb_add[i]) if gb[i] is not None else scale * gb_add[i]
        return gW, gb

    def loss_and_grads(self, data, weights=None):
        """Compute combined loss and parameter gradients.

        ``data`` keys (all optional):
          'A': (N_A, F) boundary-A features
          'B': (N_B, F) boundary-B features
          'wA','wB': per-sample WE weights (default 1)
          'sem': (X_t, X_tp, w) semigroup tuple of arrays
          'shoot': (X, n_A, n_B, w) AIMMD tuple of arrays
        ``weights`` overrides the loss-term weights
        {'boundary','semigroup','aimmd','lambda_A','lambda_B'}.
        """
        w = {'boundary': 1.0, 'semigroup': 1.0, 'aimmd': 0.0,
             'lambda_A': 1.0, 'lambda_B': 1.0}
        if weights:
            w.update(weights)

        loss = 0.0
        gW = [np.zeros_like(W) for W in self.W]
        gb = [np.zeros_like(b) for b in self.b]

        # ---- boundary A (target 0) ----
        if data.get('A') is not None and len(data['A']) > 0:
            A = np.atleast_2d(data['A'])
            wA = np.ones(len(A)) if data.get('wA') is None else np.asarray(data['wA'], float)
            qA, cA = self._forward(A)
            qA = qA.ravel()
            n = len(A)
            loss += w['boundary'] * w['lambda_A'] * np.mean(wA * qA ** 2)
            dq = (w['boundary'] * w['lambda_A'] * 2.0 * wA * qA / n)[:, None]
            gWa, gba = self._backward(cA, dq)
            self._accumulate(gW, gb, gWa, gba)

        # ---- boundary B (target 1) ----
        if data.get('B') is not None and len(data['B']) > 0:
            B = np.atleast_2d(data['B'])
            wB = np.ones(len(B)) if data.get('wB') is None else np.asarray(data['wB'], float)
            qB, cB = self._forward(B)
            qB = qB.ravel()
            n = len(B)
            loss += w['boundary'] * w['lambda_B'] * np.mean(wB * (1.0 - qB) ** 2)
            dq = (w['boundary'] * w['lambda_B'] * 2.0 * wB * (qB - 1.0) / n)[:, None]
            gWb, gbb = self._backward(cB, dq)
            self._accumulate(gW, gb, gWb, gbb)

        # ---- semigroup (variational) ----
        if data.get('sem') is not None and len(data['sem'][0]) > 0:
            Xt, Xtp, ws = data['sem']
            Xt = np.atleast_2d(Xt)
            Xtp = np.atleast_2d(Xtp)
            ws = np.ones(len(Xt)) if ws is None else np.asarray(ws, float)
            n = len(Xt)
            qt, ct = self._forward(Xt)
            qtp, ctp = self._forward(Xtp)
            r = (qt - qtp).ravel()
            loss += w['semigroup'] * np.mean(ws * r ** 2)
            dqt = (w['semigroup'] * 2.0 * ws * r / n)[:, None]
            dqtp = -dqt
            gWt, gbt = self._backward(ct, dqt)
            gWtp, gbtp = self._backward(ctp, dqtp)
            self._accumulate(gW, gb, gWt, gbt)
            self._accumulate(gW, gb, gWtp, gbtp)

        # ---- AIMMD shooting ----
        if w['aimmd'] > 0 and data.get('shoot') is not None and len(data['shoot'][0]) > 0:
            Xs, nA, nB, ws = data['shoot']
            Xs = np.atleast_2d(Xs)
            nA = np.asarray(nA, float)
            nB = np.asarray(nB, float)
            ws = np.ones(len(Xs)) if ws is None else np.asarray(ws, float)
            n = len(Xs)
            qs, cs = self._forward(Xs)
            qs = np.clip(qs.ravel(), 1e-6, 1 - 1e-6)
            loss += w['aimmd'] * (-np.mean(ws * (nA * np.log(1 - qs) + nB * np.log(qs))))
            dq = (w['aimmd'] * ws * (nA / (1 - qs) - nB / qs) / n)[:, None]
            gWs, gbs = self._backward(cs, dq)
            self._accumulate(gW, gb, gWs, gbs)

        # ---- L2 regularization ----
        if self.l2 > 0:
            for i in range(len(self.W)):
                loss += 0.5 * self.l2 * np.sum(self.W[i] ** 2)
                gW[i] = gW[i] + self.l2 * self.W[i]

        return loss, (gW, gb)

    # ------------------------------------------------------------------ #
    # training (full-batch Adam)
    # ------------------------------------------------------------------ #
    def fit(self, data, weights=None, epochs=200, lr=1e-2,
            beta1=0.9, beta2=0.999, eps=1e-8, standardize_from=None, verbose=False):
        """Train in place. Returns the loss history (list)."""
        # feature standardization from a representative pool
        if standardize_from is not None and len(standardize_from) > 0:
            S = np.atleast_2d(standardize_from)
            self._mu = S.mean(axis=0)
            self._sigma = S.std(axis=0)
            self._sigma[self._sigma < 1e-8] = 1.0

        mW = [np.zeros_like(W) for W in self.W]
        vW = [np.zeros_like(W) for W in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]

        history = []
        for t in range(1, epochs + 1):
            loss, (gW, gb) = self.loss_and_grads(data, weights)
            history.append(loss)
            for i in range(len(self.W)):
                mW[i] = beta1 * mW[i] + (1 - beta1) * gW[i]
                vW[i] = beta2 * vW[i] + (1 - beta2) * gW[i] ** 2
                mhat = mW[i] / (1 - beta1 ** t)
                vhat = vW[i] / (1 - beta2 ** t)
                self.W[i] -= lr * mhat / (np.sqrt(vhat) + eps)

                mb[i] = beta1 * mb[i] + (1 - beta1) * gb[i]
                vb[i] = beta2 * vb[i] + (1 - beta2) * gb[i] ** 2
                mhatb = mb[i] / (1 - beta1 ** t)
                vhatb = vb[i] / (1 - beta2 ** t)
                self.b[i] -= lr * mhatb / (np.sqrt(vhatb) + eps)
            if verbose and (t % max(1, epochs // 10) == 0):
                print(f"epoch {t:4d}  loss {loss:.6f}")
        return history

    # ------------------------------------------------------------------ #
    # flat parameter access (for finite-difference gradient checks / saving)
    # ------------------------------------------------------------------ #
    def get_params_flat(self):
        return np.concatenate([W.ravel() for W in self.W]
                              + [b.ravel() for b in self.b])

    def set_params_flat(self, theta):
        theta = np.asarray(theta, float)
        idx = 0
        for i in range(len(self.W)):
            n = self.W[i].size
            self.W[i] = theta[idx:idx + n].reshape(self.W[i].shape)
            idx += n
        for i in range(len(self.b)):
            n = self.b[i].size
            self.b[i] = theta[idx:idx + n].reshape(self.b[i].shape)
            idx += n

    def grads_flat(self, data, weights=None):
        _, (gW, gb) = self.loss_and_grads(data, weights)
        return np.concatenate([g.ravel() for g in gW] + [g.ravel() for g in gb])
