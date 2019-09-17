from unittest import TestCase
import numpy as np
from ..measures import caterogical


class TestCategorical(TestCase):

    def setUp(self):
        self.data = np.array([
            [1, 2, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 2],
        ])

    def test_overlap(self):
        out = caterogical.overlap(self.data)
        desired = np.array([
            [1, 0.6666667],
            [0.6666667, 1]
        ])
        np.testing.assert_almost_equal(out, desired)

    def test_iof(self):
        out = caterogical.iof(self.data)
        desired = np.array([
            [1, 1],
            [1, 1]
        ])
        # because log(1) = 0, 1 + log(1) * log(1) = 1
        np.testing.assert_almost_equal(out, desired)

    def test_of(self):
        out = caterogical.of(self.data)
        desired = np.array([
            [1, 0.891823],
            [0.891823, 1]
        ])
        # because 1 /(1 + log(2/1) * log(2/1)) = 1/1.4761 = 0.67746
        # (1 * 3 + 0.67746 * 2) / 5 = 0.891823
        np.testing.assert_almost_equal(out, desired)

    def test_goodall3(self):
        out = caterogical.goodall_3(self.data)
        desired = np.array([
            [1, 0],
            [0, 1]
        ])
        # because p^2_k = 2 * 1 / 2 * 1 = 1
        # 1 - p^2_k = 0
        np.testing.assert_almost_equal(out, desired)
