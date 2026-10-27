"""
Python translation of the MATLAB BLNMixture_Demo.m script.

Usage:
    python3 example/demo_bln_mixture.py

This will simulate a three-component BLN mixture with a small uniform
contamination, fit the model back via EM, report key diagnostics, and show
the parameter recovery compared to the ground truth.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from blnm.mixture import BLNMixture, _REALMIN


def main() -> None:
    rng = np.random.default_rng(1)

    true_model = BLNMixture(
        K=3,
        mu=np.array([-0.6, 0.0, 0.8], dtype=float),
        sigma=np.array([0.155, 0.13, 0.145], dtype=float),
        pi=(1.0 - 1e-3) * np.array([0.4, 0.15, 0.55], dtype=float),
        uniform_weight=1e-3,
        quad_nodes=30,
    )

    N = 2000
    totals = rng.integers(10, 1001, size=N)
    x, xc, z = true_model.rnd(totals, random_state=1)

    for idx in range(min(30, N)):
        ni = totals[idx]
        draw = rng.integers(0, ni + 1)
        x[idx] = draw
        xc[idx] = ni - draw

    print("Fitting BLNMixture...")
    fitted = BLNMixture.fit(
        x,
        xc,
        3,
        restarts=3,
        uniform_weight=1e-3,
        display="iter",
        random_state=42,
    )

    print(f"Converged: {fitted.converged} | LL: {fitted.loglik:.3f} | iters: {fitted.iters}")
    counts = Counter(fitted.labels.tolist())
    print("Label counts (1..K=BLN comps, K+1=uniform):")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")

    probs = fitted.pdf(x, totals)
    mean_log_p = float(np.mean(np.log(np.maximum(probs, _REALMIN))))
    print(f"Mean log pmf on data: {mean_log_p:.3f}")

    print("\nTrue vs estimated parameters:")
    print("Component | mu_true   mu_hat   sigma_true sigma_hat")
    for k in range(3):
        print(
            f"    {k+1:>2}    | "
            f"{true_model.mu[k]:7.3f}  {fitted.mu[k]:7.3f}    "
            f"{true_model.sigma[k]:7.3f}   {fitted.sigma[k]:7.3f}"
        )


if __name__ == "__main__":
    main()
