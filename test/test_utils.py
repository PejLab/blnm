from unittest import TestCase, main

import numpy as np
from blnm import utils


class TestMbic(TestCase):
    log_likelihood = -1
    n_samples = 100
    k_mixtures = 5
    bic = 2*k_mixtures*np.log(n_samples) - 2*log_likelihood

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            utils.mbic(self.log_likelihood,
                      self.n_samples,
                      -3)

        with self.assertRaises(ValueError):
            utils.mbic(self.log_likelihood,
                    -4,
                    self.k_mixtures)

        with self.assertRaises(ValueError):
            utils.mbic(self.log_likelihood,
                    -4,
                    -3)
        
        with self.assertRaises(ValueError):
            utils.mbic(np.array([-1,-2,-3]),
                      self.n_samples,
                      self.k_mixtures)

    def test_output(self):
        self.assertTrue(np.isnan(utils.mbic(np.nan, 1000, 3)))

        self.assertEqual(utils.mbic(self.log_likelihood,
                                    self.n_samples,
                                    self.k_mixtures),
                         self.bic)


class TestLogistic(TestCase):
    def test_output(self):
        self.assertAlmostEqual(utils.logistic(np.log(2)), 2/3)

        self.assertAlmostEqual(utils.logistic(-np.log(2)), 1/3)

        s = np.full(100, -np.log(2))
        out = utils.logistic(s)

        self.assertEqual(out.size, s.size)

        for val in out:
            self.assertAlmostEqual(val, 1/3)



if __name__ == "__main__":
    main()
