from unittest import TestCase, main

import numpy as np
from blnm import dist


class TestSampleBlnm(TestCase):
    def test_output(self):
        x = dist.sample_blnm([0.3, 0.7], [-1.0, 1.0], 0.25, 40, 200, rng=0)

        self.assertEqual(x.shape, (200,))
        self.assertEqual(x.dtype, np.int64)
        self.assertTrue(np.all((x >= 0) & (x <= 40)))


if __name__ == "__main__":
    main()
