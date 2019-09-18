from unittest import TestCase
import numpy as np

from ..measures import measure, caterogical, numerical
from ..measures.concat import weighted_average_algorithm

#TODO: fix measure test

class TestMeasure(TestCase):

    def setUp(self):
        self.data = np.array([
            [1, 2, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 2],
        ])

    def test_concat_weighted_average(self):
        measure_strategy1 = measure.Measure(numerical_strategy=numerical.manhattan,
                                            categorical_strategy=caterogical.overlap,
                                            concat_strategy=weighted_average_algorithm)

        measures1 = measure_strategy1.compute(self.data, "numerical")
        desired1 = np.array([
            [0, 2.828427124],
            [2.828427124, 0]
        ])
        np.testing.assert_almost_equal(measures1, desired1)

        measures2 = measure_strategy1.compute(self.data, "categorical")
        desired2 = np.array([
            [1, 0.6666667],
            [0.6666667, 1]
        ])
        np.testing.assert_almost_equal(measures2, desired2)

        desc = ["categorical",
                "numerical",
                "categorical",
                "numerical",
                "numerical",
                "categorical"]

        measures3 = measure_strategy1.compute(self.data, desc)
        desired3 = np.array([
            [0, 0.6666667],
            [0.6666667, 0]
        ])
        # because
        # categorical dist = 0.33
        # num dist = 1 (after normalization to <0;1>)
        # (1 + 0.33) / 2 = 0.66666667
        np.testing.assert_almost_equal(measures3, desired3)
