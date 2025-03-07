from unittest import TestCase, main

import numpy as np
from blnm import fit



class TestInitPars(TestCase):
    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            fit.init_pars()


class TestFitParCheck(TestCase):
    def setUp(self):
        rng = np.random.default_rng()

        self.n = 300
        self.k_mixtures = 4
        self.coefs = np.array([0.1,0.4,0.3,0.2])
        self.means = np.array([-2,-1,1,2], dtype=float)
        self.variance = 1.1
        self.xcount = rng.choice(250, size=self.n, replace=True)
        self.ncount = np.full((self.n,), 250)

    def test_correct_input(self):
        fit.blnm(self.xcount, self.ncount, self.k_mixtures,
                 coefs=self.coefs, means=self.means,
                 variance=self.variance)

    def test_check_coefs_negative_val(self):
        self.coefs[2] = -self.coefs[2]

        with self.assertRaises(ValueError):
            fit.blnm(self.xcount, self.ncount, self.k_mixtures,
                     coefs=self.coefs, means=self.means,
                     variance=self.variance)

    def test_check_coefs_wrong_number(self):
        self.coefs = self.coefs[0:2]

        with self.assertRaises(ValueError):
            fit.blnm(self.xcount, self.ncount, self.k_mixtures,
                     coefs=self.coefs, means=self.means,
                     variance=self.variance)


    def test_check_coefs_wrong_number(self):
        self.coefs[2] = self.coefs[2] * 1.1

        with self.assertRaises(ValueError):
            fit.blnm(self.xcount, self.ncount, self.k_mixtures,
                     coefs=self.coefs, means=self.means,
                     variance=self.variance)

    def test_check_coefs_nan(self):
        self.coefs[2] = np.nan

        with self.assertRaises(ValueError):
            fit.blnm(self.xcount, self.ncount, self.k_mixtures,
                     coefs=self.coefs, means=self.means,
                     variance=self.variance)

    def test_check_means_wrong_number(self):
        self.means = self.means[0:2]

        with self.assertRaises(ValueError):
            fit.blnm(self.xcount, self.ncount, self.k_mixtures,
                     coefs=self.coefs, means=self.means,
                     variance=self.variance)


    def test_check_means_wrong_number(self):
        self.means[2] = np.nan

        with self.assertRaises(ValueError):
            fit.blnm(self.xcount, self.ncount, self.k_mixtures,
                     coefs=self.coefs, means=self.means,
                     variance=self.variance)

    def test_check_variance(self):
        self.variance = -1.2

        with self.assertRaises(ValueError):
            fit.blnm(self.xcount, self.ncount, self.k_mixtures,
                     coefs=self.coefs, means=self.means,
                     variance=self.variance)

    def test_check_nan(self):
        self.variance = np.nan

        with self.assertRaises(ValueError):
            fit.blnm(self.xcount, self.ncount, self.k_mixtures,
                     coefs=self.coefs, means=self.means,
                     variance=self.variance)


if __name__ == "__main__":
    main()
