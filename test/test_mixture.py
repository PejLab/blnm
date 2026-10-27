from unittest import TestCase, main

import numpy as np
from blnm import dist
from blnm import fit
from blnm.mixture import BLNMixture


class TestPdf(TestCase):
    def setUp(self):
        self.model = BLNMixture(K=2,
                                mu=np.array([-1.0, 1.5]),
                                sigma=np.array([0.4, 0.8]),
                                pi=np.array([0.3, 0.7]),
                                uniform_weight=0.0)

    def test_sums_to_one(self):
        for w0 in [0.0, 0.05]:
            self.model.uniform_weight = w0
            self.model.pi = (1 - w0) * np.array([0.3, 0.7])
            for n in [1, 10, 75]:
                x = np.arange(n + 1)
                total = np.sum(self.model.pdf(x, np.full(n + 1, n)))
                self.assertAlmostEqual(total, 1.0, places=6)

    def test_log(self):
        x = np.array([0, 3, 10])
        n = np.array([10, 10, 10])
        np.testing.assert_allclose(self.model.pdf(x, n, return_log=True),
                                   np.log(self.model.pdf(x, n)))

    def test_matches_monte_carlo_pmf(self):
        # Single component BLN against the Monte Carlo pmf in blnm.dist
        model = BLNMixture(K=1, mu=np.array([0.5]), sigma=np.array([0.7]),
                           pi=np.array([1.0]), uniform_weight=0.0)
        x = np.arange(21)
        n = np.full(21, 20)
        mc = dist.pmf(x, n, 0.5, 0.7**2, size=200000, seed=0)
        np.testing.assert_allclose(model.pdf(x, n), mc, atol=2e-3)

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):
            self.model.pdf([5], [4])


class TestRnd(TestCase):
    def test_shapes_and_labels(self):
        model = BLNMixture(K=3, uniform_weight=0.1)
        n = np.full(500, 30)
        x, xc, labels = model.rnd(n, random_state=0)

        self.assertEqual(x.shape, (500,))
        np.testing.assert_array_equal(x + xc, n)
        self.assertTrue(np.all((x >= 0) & (x <= n)))
        self.assertTrue(set(labels.tolist()) <= {1, 2, 3, 4})


class TestFit(TestCase):
    def setUp(self):
        self.true_model = BLNMixture(K=2,
                                     mu=np.array([-1.5, 1.0]),
                                     sigma=np.array([0.3, 0.5]),
                                     pi=(1 - 1e-3) * np.array([0.4, 0.6]),
                                     uniform_weight=1e-3)
        totals = np.random.default_rng(3).integers(20, 200, size=600)
        self.x, self.xc, _ = self.true_model.rnd(totals, random_state=4)

    def test_parameter_recovery(self):
        fitted = BLNMixture.fit(self.x, self.xc, 2, random_state=0)
        order = np.argsort(fitted.mu)

        self.assertTrue(fitted.converged)
        np.testing.assert_allclose(fitted.mu[order], self.true_model.mu,
                                   atol=0.15)
        np.testing.assert_allclose(fitted.sigma[order],
                                   self.true_model.sigma, atol=0.15)
        np.testing.assert_allclose(fitted.pi[order], self.true_model.pi,
                                   atol=0.06)

    def test_loglik_consistent_with_parameters(self):
        fitted = BLNMixture.fit(self.x, self.xc, 2, random_state=0)
        ll = np.sum(fitted.pdf(self.x, self.x + self.xc, return_log=True))

        self.assertAlmostEqual(fitted.loglik, ll, places=6)
        self.assertEqual(fitted.labels.shape, self.x.shape)
        np.testing.assert_allclose(fitted.resp.sum(axis=1), 1.0)

    def test_restarts_differ(self):
        inits = []
        for restart_seed in range(2):
            rng = np.random.default_rng(restart_seed)
            mu, _, _ = BLNMixture._initialise_parameters(
                K=3, X=self.x.astype(float), Xc=self.xc.astype(float),
                n=(self.x + self.xc).astype(float), rng=rng, init=None,
                uniform_weight=1e-3, shared_sigma=False)
            inits.append(np.sort(mu))
        self.assertFalse(np.allclose(inits[0], inits[1]))

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):
            BLNMixture.fit([1, -1], [2, 2], 1)
        with self.assertRaises(ValueError):
            BLNMixture.fit([1, 2], [2], 1)


class TestAgreesWithFitBlnm(TestCase):
    """Where the models overlap (shared variance, no uniform component)
    BLNMixture and fit.blnm should find the same parameters."""

    def test_shared_sigma(self):
        true_model = BLNMixture(K=2,
                                mu=np.array([-1.5, 1.5]),
                                sigma=np.full(2, 0.5),
                                pi=np.array([0.3, 0.7]),
                                uniform_weight=0.0)
        n = np.full(300, 50)
        x, _, _ = true_model.rnd(n, random_state=11)

        fitted = BLNMixture.fit(x, n - x, 2, shared_sigma=True,
                                uniform_weight=0.0, random_state=0)
        reference = fit.blnm(x, n, 2,
                             coefs=np.array([0.5, 0.5]),
                             means=np.array([-1.0, 1.0]),
                             variance=0.5,
                             seed=0, disp=False, tolerance=1e-4,
                             integral_n_samples=2000)

        order = np.argsort(fitted.mu)
        ref_order = np.argsort(reference["means"])

        np.testing.assert_allclose(fitted.mu[order],
                                   reference["means"][ref_order], atol=0.1)
        np.testing.assert_allclose(fitted.pi[order],
                                   reference["coefs"][ref_order], atol=0.03)
        self.assertAlmostEqual(fitted.sigma[0]**2, reference["variance"],
                               delta=0.08)


if __name__ == "__main__":
    main()
