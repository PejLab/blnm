r"""
Implementation of a Binomial-Logit-Normal (BLN) mixture model with an optional
uniform-outlier component, following the MATLAB reference implementation.

The mixture models count data :math:`x_i` with totals :math:`n_i` through a
hierarchical model:

.. math::
    p(x_i) = (1 - w_0) \sum_{k=1}^K \pi_k \, \text{BLN}_k(x_i \mid n_i; \mu_k, \sigma_k)
             + w_0 \frac{1}{n_i + 1},

where :math:`\text{BLN}` denotes the binomial-logit-normal distribution. Each
component integrates over a latent logistic-normal random effect and the
integral is approximated with Gauss–Hermite quadrature.

The class provides:
    - `BLNMixture.fit` for EM-based fitting with optional shared variance
    - `BLNMixture.pdf` to evaluate mixture probabilities (or log-probabilities)
    - `BLNMixture.rnd` to simulate counts from the fitted model

This module depends only on NumPy. SciPy is not required; a light-weight BFGS
optimizer is implemented here for the one- and two-parameter updates that
occur in the M-step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


_LOG_MIN_SIGMA = math.log(1e-12)
_LOG_MAX_SIGMA = math.log(1e3)
_REALMIN = np.finfo(float).tiny


def _ensure_numpy_column(data: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(data), dtype=float).reshape(-1)
    return arr


def _logistic(z: np.ndarray) -> np.ndarray:
    # Stable logistic, clipping avoids log(0) downstream.
    out = 1.0 / (1.0 + np.exp(-z))
    return np.clip(out, 1e-12, 1.0 - 1e-12)


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    s = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(s, axis=axis)


def _gh_nodes_weights(num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return Gauss–Hermite nodes/weights for standard normal expectations."""
    nodes, weights = np.polynomial.hermite.hermgauss(num_nodes)
    weights = weights / math.sqrt(math.pi)
    return nodes.astype(float), weights.astype(float)


def _kmeans_1d(
    z: np.ndarray,
    K: int,
    rng: np.random.Generator,
    max_iter: int = 100,
) -> np.ndarray:
    """Lloyd's k-means on 1-D data with k-means++ seeding; returns labels 0..K-1."""
    centers = np.empty(K)
    centers[0] = z[rng.integers(z.size)]
    for k in range(1, K):
        d2 = np.min((z[:, None] - centers[None, :k]) ** 2, axis=1)
        total = d2.sum()
        if total > 0:
            centers[k] = z[rng.choice(z.size, p=d2 / total)]
        else:
            centers[k] = z[rng.integers(z.size)]

    labels = np.full(z.size, -1)
    for _ in range(max_iter):
        new_labels = np.argmin(np.abs(z[:, None] - centers[None, :]), axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(K):
            if np.any(labels == k):
                centers[k] = z[labels == k].mean()
    return labels


def _bfgs_minimize(
    fun,
    x0: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Very small BFGS optimizer for smooth objectives providing value and gradient.
    The callable `fun` must accept a parameter vector and return (value, grad).
    """
    x = np.asarray(x0, dtype=float)
    n_dim = x.size
    identity = np.eye(n_dim)
    hess_inv = np.eye(n_dim)

    value, grad = fun(x)
    info = {"iterations": 0, "converged": False}

    for it in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            info["converged"] = True
            break

        direction = -hess_inv @ grad
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0:
            direction = -grad

        # Backtracking line search with Armijo condition.
        step = 1.0
        c1 = 1e-4
        while step > 1e-12:
            candidate = x + step * direction
            cand_value, cand_grad = fun(candidate)
            if not np.isfinite(cand_value):
                step *= 0.5
                continue
            if cand_value <= value + c1 * step * np.dot(grad, direction):
                break
            step *= 0.5
        else:
            # Line search failed: retry once along the steepest-descent
            # direction, and stop if that also makes no progress.
            if np.array_equal(hess_inv, identity):
                break
            hess_inv = identity
            continue

        s_vec = candidate - x
        y_vec = cand_grad - grad
        ys = np.dot(y_vec, s_vec)
        if ys > 1e-12:
            rho = 1.0 / ys
            vt = identity - rho * np.outer(s_vec, y_vec)
            hess_inv = vt @ hess_inv @ vt.T + rho * np.outer(s_vec, s_vec)
        else:
            hess_inv = identity

        x, value, grad = candidate, cand_value, cand_grad
        info["iterations"] = it + 1

    return x, info


@dataclass
class BLNMixture:
    K: int = 1
    mu: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    sigma: np.ndarray = field(default_factory=lambda: np.full(1, 0.5, dtype=float))
    pi: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=float))
    uniform_weight: float = 1e-3
    quad_nodes: int = 30
    loglik: float = math.nan
    iters: int = 0
    converged: bool = False
    resp: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float).reshape(-1)
        self.sigma = np.asarray(self.sigma, dtype=float).reshape(-1)
        self.pi = np.asarray(self.pi, dtype=float).reshape(-1)
        self.K = int(self.K)
        if self.mu.size != self.K:
            self.mu = np.full(self.K, self.mu.mean() if self.mu.size else 0.0)
        if self.sigma.size != self.K:
            self.sigma = np.full(self.K, 0.5)
        if self.pi.size != self.K:
            self.pi = np.ones(self.K) / self.K

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    @classmethod
    def fit(
        cls,
        x: Iterable[float],
        xc: Iterable[float],
        K: int,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
        quad_nodes: int = 30,
        restarts: int = 3,
        display: str = "off",
        init: Optional[Dict[str, Iterable[float]]] = None,
        uniform_weight: float = 1e-3,
        shared_sigma: bool = False,
        random_state: Optional[int] = None,
    ) -> "BLNMixture":
        rng = np.random.default_rng(random_state)

        X = _ensure_numpy_column(x)
        Xc = _ensure_numpy_column(xc)
        if X.size != Xc.size:
            raise ValueError("x and xc must have the same length.")
        # Reject rather than drop invalid observations, so that `resp` and
        # `labels` stay aligned with the caller's input.
        if X.size == 0:
            raise ValueError("No observations supplied.")
        if not np.all(np.isfinite(X) & np.isfinite(Xc) & (X >= 0) & (Xc >= 0)):
            raise ValueError("Counts must be finite and non-negative.")
        n = X + Xc
        num_obs = X.size

        w0 = min(max(float(uniform_weight), 0.0), 0.5)
        quad_nodes = int(max(5, quad_nodes))

        logC = np.array(
            [math.lgamma(ni + 1.0) - math.lgamma(xi + 1.0) - math.lgamma(xci + 1.0) for ni, xi, xci in zip(n, X, Xc)],
            dtype=float,
        )

        xh, wh = _gh_nodes_weights(quad_nodes)

        best_model: Optional[BLNMixture] = None
        best_ll = -math.inf

        for restart in range(max(1, restarts)):
            mu, sigma, pi = cls._initialise_parameters(
                K=K,
                X=X,
                Xc=Xc,
                n=n,
                rng=rng,
                init=init if restart == 0 and init else None,
                uniform_weight=w0,
                shared_sigma=shared_sigma,
            )

            prev_ll = -math.inf
            converged = False
            resp = None
            labels = None

            for iteration in range(1, max_iter + 1):
                resp, LL = cls._e_step(X, Xc, n, logC, mu, sigma, pi, w0, xh, wh)

                resp_bln = resp[:, :K]
                resp_uni = resp[:, K]

                Nk = resp_bln.sum(axis=0)
                if Nk.sum() <= 0.0:
                    pi = np.full(K, (1.0 - w0) / K)
                else:
                    pi = (1.0 - w0) * (Nk / Nk.sum())

                if not shared_sigma:
                    for k in range(K):
                        if Nk[k] <= 0.0:
                            continue
                        weights = resp_bln[:, k] / max(Nk[k], _REALMIN)
                        theta0 = np.array([mu[k], math.log(max(sigma[k], 1e-4))], dtype=float)

                        def obj(theta: np.ndarray) -> Tuple[float, np.ndarray]:
                            val, grad = cls._nll_grad_weighted(theta, X, Xc, logC, weights, xh, wh)
                            return val, grad

                        theta_opt, _ = _bfgs_minimize(obj, theta0, max_iter=200, tol=1e-5)
                        mu[k] = float(theta_opt[0])
                        sigma[k] = float(np.exp(np.clip(theta_opt[1], _LOG_MIN_SIGMA, _LOG_MAX_SIGMA)))
                else:
                    for k in range(K):
                        if Nk[k] <= 0.0:
                            continue
                        weights = resp_bln[:, k] / max(Nk[k], _REALMIN)
                        logsig = math.log(max(sigma[k], 1e-4))

                        def obj_mu(theta: np.ndarray) -> Tuple[float, np.ndarray]:
                            val, grad_mu = cls._nll_grad_mu_only(theta[0], X, Xc, logC, weights, math.exp(logsig), xh, wh)
                            return val, np.array([grad_mu], dtype=float)

                        mu_opt, _ = _bfgs_minimize(obj_mu, np.array([mu[k]], dtype=float), max_iter=200, tol=1e-5)
                        mu[k] = float(mu_opt[0])

                    logsig0 = math.log(max(np.median(sigma), 1e-4))

                    def obj_sig(theta: np.ndarray) -> Tuple[float, np.ndarray]:
                        val, grad = cls._nll_grad_logsigma_shared(theta[0], X, Xc, logC, resp_bln, Nk, mu, xh, wh)
                        return val, np.array([grad], dtype=float)

                    logsig_opt, _ = _bfgs_minimize(obj_sig, np.array([logsig0], dtype=float), max_iter=200, tol=1e-5)
                    sigma_shared = float(np.exp(np.clip(logsig_opt[0], _LOG_MIN_SIGMA, _LOG_MAX_SIGMA)))
                    sigma = np.full(K, sigma_shared)

                rel_impr = (LL - prev_ll) / (1.0 + abs(prev_ll))
                if display.lower() == "iter":
                    print(
                        f"Restart {restart + 1} | iter {iteration:3d}  LL = {LL:.6f}  "
                        f"relImpr = {rel_impr:.3e}  (w0={w0:.2e}{', shared sigma' if shared_sigma else ''})"
                    )
                if abs(rel_impr) < tol and iteration > 2:
                    converged = True
                    break
                prev_ll = LL

            # The M-step above moved the parameters past the last E-step;
            # recompute so that loglik, resp and labels match mu, sigma, pi.
            resp, LL = cls._e_step(X, Xc, n, logC, mu, sigma, pi, w0, xh, wh)

            # Report components in increasing order of mu; the uniform
            # component stays last.
            order = np.argsort(mu, kind="stable")
            mu, sigma, pi = mu[order], sigma[order], pi[order]
            resp = resp[:, np.append(order, K)]
            labels = np.argmax(resp, axis=1) + 1

            model = cls(
                K=K,
                mu=mu.copy(),
                sigma=sigma.copy(),
                pi=pi.copy(),
                uniform_weight=w0,
                quad_nodes=quad_nodes,
                loglik=LL,
                iters=iteration,
                converged=converged,
                resp=resp.copy(),
                labels=labels,
                details={
                    "restarts": restarts,
                    "restart_index": restart,
                    "shared_sigma": shared_sigma,
                    "quad_nodes": quad_nodes,
                },
            )

            if LL > best_ll:
                best_ll = LL
                best_model = model

        if best_model is None:
            raise RuntimeError("BLNMixture fitting failed to find a solution.")

        return best_model

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #
    def pdf(self, x: Iterable[float], n: Iterable[float], return_log: bool = False) -> np.ndarray:
        X = _ensure_numpy_column(x)
        n = _ensure_numpy_column(n)
        if X.size != n.size:
            raise ValueError("x and n must have the same length.")
        if np.any(n < 0) or np.any(X < 0) or np.any(X > n):
            raise ValueError("Counts must satisfy 0 <= x <= n.")
        Xc = n - X
        logC = np.array(
            [math.lgamma(ni + 1.0) - math.lgamma(xi + 1.0) - math.lgamma(xci + 1.0) for ni, xi, xci in zip(n, X, Xc)],
            dtype=float,
        )
        xh, wh = _gh_nodes_weights(self.quad_nodes)
        logpk = self._comp_loglik_matrix(X, Xc, logC, self.mu, self.sigma, xh, wh)
        logpi = np.log(np.maximum(self.pi, _REALMIN))
        log_bln = _logsumexp(logpk + logpi[np.newaxis, :], axis=1)
        log_uniform = -np.log(n + 1.0)
        # pi already sums to 1 - uniform_weight, so the BLN part needs no
        # further (1 - w0) factor.
        logw_uni = math.log(max(self.uniform_weight, _REALMIN))
        logp = _logsumexp(
            np.column_stack([log_bln, logw_uni + log_uniform]),
            axis=1,
        )
        return logp if return_log else np.exp(logp)

    def rnd(self, n: Iterable[float], random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = _ensure_numpy_column(n)
        rng = np.random.default_rng(random_state)
        w_bln = np.maximum(self.pi, 0.0)
        w0 = max(self.uniform_weight, 0.0)
        mix_weights = np.append(w_bln, w0)
        mix_weights = mix_weights / mix_weights.sum()
        comp_edges = np.cumsum(mix_weights)
        comp_edges[-1] = 1.0

        uniforms = rng.random(n.size)
        labels = np.searchsorted(comp_edges, uniforms, side="right")

        x = np.zeros_like(n, dtype=int)
        xc = np.zeros_like(n, dtype=int)

        for k in range(self.K):
            idx = labels == k
            if not np.any(idx):
                continue
            nk = n[idx]
            z = rng.normal(loc=self.mu[k], scale=self.sigma[k], size=nk.size)
            p = _logistic(z)
            x[idx] = rng.binomial(nk.astype(int), p)
            xc[idx] = nk.astype(int) - x[idx]

        uniform_idx = labels == self.K
        if np.any(uniform_idx):
            nk = n[uniform_idx].astype(int)
            draws = np.array([rng.integers(low=0, high=ni + 1) for ni in nk], dtype=int)
            x[uniform_idx] = draws
            xc[uniform_idx] = nk - draws

        return x, xc, labels + 1

    # ------------------------------------------------------------------ #
    # Internals for EM helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _initialise_parameters(
        K: int,
        X: np.ndarray,
        Xc: np.ndarray,
        n: np.ndarray,
        rng: np.random.Generator,
        init: Optional[Dict[str, Iterable[float]]],
        uniform_weight: float,
        shared_sigma: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w_bln = 1.0 - uniform_weight
        if init:
            mu = _ensure_numpy_column(init.get("mu", np.zeros(K)))
            if mu.size != K:
                raise ValueError("Init.mu must have length K.")
            sigma_init = init.get("sigma", np.full(K, 0.5))
            sigma = _ensure_numpy_column(sigma_init)
            if sigma.size == 1 and shared_sigma:
                sigma = np.full(K, max(float(sigma[0]), 1e-4))
            elif sigma.size != K:
                raise ValueError("Init.sigma must have length K or be scalar when SharedSigma is True.")
            sigma = np.maximum(sigma, 1e-4)

            pi = init.get("pi")
            if pi is not None:
                pi = _ensure_numpy_column(pi)
                if pi.size != K:
                    raise ValueError("Init.pi must have length K.")
                total = pi.sum()
                if total <= 0:
                    pi = np.full(K, w_bln / K)
                else:
                    pi = w_bln * pi / total
            else:
                pi = np.full(K, w_bln / K)
        else:
            phat = (X + 0.5) / (n + 1.0)
            logits = np.log(phat) - np.log1p(-phat)
            logits[~np.isfinite(logits)] = 0.0
            if K > 1 and X.size >= K:
                # Randomly seeded k-means, as in the MATLAB/R versions, so
                # that each restart begins from a different partition.
                labels = _kmeans_1d(logits, K, rng)
            else:
                labels = np.zeros_like(logits, dtype=int)

            mu = np.zeros(K)
            sigma = np.zeros(K)
            pi = np.zeros(K)
            for k in range(K):
                idx = labels == k
                if not np.any(idx):
                    idx = rng.integers(low=0, high=X.size, size=max(1, X.size // K))
                else:
                    idx = np.where(idx)[0]
                z = logits[idx]
                mu[k] = np.median(z)
                sigma[k] = max(np.std(z), 0.3)
                pi[k] = w_bln * len(idx) / X.size

            total = pi.sum()
            if total <= 0:
                pi = np.full(K, w_bln / K)
            else:
                pi = w_bln * pi / total

        if shared_sigma:
            sigma_shared = np.median(sigma)
            sigma = np.full(K, max(sigma_shared, 1e-4))

        return mu.astype(float), sigma.astype(float), pi.astype(float)

    @classmethod
    def _e_step(
        cls,
        X: np.ndarray,
        Xc: np.ndarray,
        n: np.ndarray,
        logC: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        pi: np.ndarray,
        w0: float,
        xh: np.ndarray,
        wh: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Return the (N, K+1) responsibilities and the data log-likelihood."""
        logpk = cls._comp_loglik_matrix(X, Xc, logC, mu, sigma, xh, wh)
        logpi = np.log(np.maximum(pi, _REALMIN))
        log_uniform = -np.log(n + 1.0) + math.log(max(w0, _REALMIN))
        log_components = np.column_stack(
            [logpk + logpi[np.newaxis, :], log_uniform.reshape(-1, 1)]
        )
        log_norm = _logsumexp(log_components, axis=1)
        resp = np.exp(log_components - log_norm[:, np.newaxis])
        return resp, float(np.sum(log_norm))

    @staticmethod
    def _comp_loglik_matrix(
        X: np.ndarray,
        Xc: np.ndarray,
        logC: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        xh: np.ndarray,
        wh: np.ndarray,
    ) -> np.ndarray:
        N = X.size
        K = mu.size
        log_w = np.log(np.maximum(wh, _REALMIN))
        base = math.sqrt(2.0) * xh
        L = np.empty((N, K), dtype=float)

        for k in range(K):
            Z = mu[k] + sigma[k] * base
            P = _logistic(Z)
            logP = np.log(P)
            log1mP = np.log1p(-P)
            LL = logC[:, None] + X[:, None] * logP + Xc[:, None] * log1mP + log_w[None, :]
            L[:, k] = _logsumexp(LL, axis=1)

        return L

    @staticmethod
    def _nll_grad_weighted(
        theta: np.ndarray,
        X: np.ndarray,
        Xc: np.ndarray,
        logC: np.ndarray,
        weights: np.ndarray,
        xh: np.ndarray,
        wh: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        mu = theta[0]
        sigma = max(math.exp(theta[1]), 1e-12)

        Z = mu + sigma * math.sqrt(2.0) * xh
        P = _logistic(Z)
        logP = np.log(P)
        log1mP = np.log1p(-P)

        LL = logC[:, None] + X[:, None] * logP + Xc[:, None] * log1mP + np.log(np.maximum(wh, _REALMIN))[None, :]
        s = _logsumexp(LL, axis=1)
        f = -np.sum(weights * s)

        A = np.exp(LL - s[:, None])
        n = X + Xc
        score = X[:, None] - n[:, None] * P[None, :]
        Ez = np.sum(A * score, axis=1)
        Esg = np.sum(A * (score * (math.sqrt(2.0) * xh)[None, :]), axis=1)

        g_mu = -np.sum(weights * Ez)
        g_sigma = -np.sum(weights * Esg) * sigma
        return float(f), np.array([g_mu, g_sigma], dtype=float)

    @classmethod
    def _nll_grad_mu_only(
        cls,
        mu: float,
        X: np.ndarray,
        Xc: np.ndarray,
        logC: np.ndarray,
        weights: np.ndarray,
        sigma_fixed: float,
        xh: np.ndarray,
        wh: np.ndarray,
    ) -> Tuple[float, float]:
        theta = np.array([mu, math.log(max(sigma_fixed, 1e-12))], dtype=float)
        f, grad = cls._nll_grad_weighted(theta, X, Xc, logC, weights, xh, wh)
        return f, grad[0]

    @classmethod
    def _nll_grad_logsigma_shared(
        cls,
        logsigma: float,
        X: np.ndarray,
        Xc: np.ndarray,
        logC: np.ndarray,
        resp_bln: np.ndarray,
        Nk: np.ndarray,
        mu: np.ndarray,
        xh: np.ndarray,
        wh: np.ndarray,
    ) -> Tuple[float, float]:
        sigma = max(math.exp(logsigma), 1e-12)
        total_f = 0.0
        total_grad = 0.0
        for k in range(mu.size):
            if Nk[k] <= 0.0:
                continue
            weights = resp_bln[:, k] / max(Nk[k], _REALMIN)
            theta = np.array([mu[k], logsigma], dtype=float)
            f, grad = cls._nll_grad_weighted(theta, X, Xc, logC, weights, xh, wh)
            total_f += Nk[k] * f
            total_grad += Nk[k] * grad[1]
        return total_f, total_grad
