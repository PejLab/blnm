from unittest import TestCase, main

import numpy as np
from blnm import utils


class TestBIC(TestCase):
    log_likelihood = -1
    n_samples = 100
    k_mixtures = 5

    def test_input_sanitize(self):
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



if __name__ == "__main__":
    main()
