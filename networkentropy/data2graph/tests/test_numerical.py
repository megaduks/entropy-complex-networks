from unittest import TestCase
import numpy as np
from ..measures import measure, caterogical, numerical

#TODO fix numerical tests with simple test case

class TestNumerical(TestCase):

    def setUp(self):
        self.data = np.array([
            [0, 100, 2],
            [-10, 100, 0],
            [10, 100, 1]
        ])

        self.data2 = np.array([
            [0, 1, 2],
            [0, 2, 4],
            [1, 2, 1]
        ])

        self.data3 = np.array([
            [0, 1, 2, 0, 0, 0, 1, 1],
            [0, -2, -4, 0, 0, 0, -1, -1]
        ])

    def test_euclidean(self):
        desired = np.array([
            [0.        ,  2.23606798,  1.41421356],
            [2.23606798,  0.        ,  2.23606798],
            [1.41421356,  2.23606798,  0.]
        ])
        out = numerical.euclidean(self.data3)
        np.testing.assert_almost_equal(out, desired)

    def test_manhattan(self):
        desired = np.array([
            [0, 12, 11],
            [12, 0, 21],
            [11, 21, 0]
        ])
        out = numerical.manhattan(self.data)
        np.testing.assert_almost_equal(out, desired)

    def test_cosine(self):
        desired = np.array([
            [0, 0, 0.26970325666],
            [0, 0, 0.26970325666],
            [0.26970325666, 0.26970325666, 0]
        ])
        out = numerical.cosine(self.data2)
        np.testing.assert_almost_equal(out, desired)

    def test_correlation(self):
        desired = np.array([
            [0, 1.96698755683],
            [1.96698755683, 0]
        ])
        out = numerical.cosine(self.data3)
        np.testing.assert_almost_equal(out, desired)

