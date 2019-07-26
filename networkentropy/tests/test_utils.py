import unittest
import numpy as np
from networkentropy import utils


class UtilsTests(unittest.TestCase):

    def test_precision_at_k_full_coverage(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 8, 9]
        k = 3

        self.assertEqual(utils.precision_at_k(y_true=y_true, y_pred=y_pred, k=k), 1.0)

    def test_precision_at_k_partial_coverage(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 8, 9]
        k = 4

        self.assertEqual(utils.precision_at_k(y_true=y_true, y_pred=y_pred, k=k), 0.75)

    def test_precision_at_k_no_coverage(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [9, 8, 7, 6, 5]
        k = 3

        self.assertEqual(utils.precision_at_k(y_true=y_true, y_pred=y_pred, k=k), 0.0)

    def test_precision_at_k_empty_list(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = []
        k = 3

        self.assertEqual(utils.precision_at_k(y_true=y_true, y_pred=y_pred, k=k), 0.0)

    def test_precision_at_k_no_list(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = None
        k = 3

        with self.assertRaises(AssertionError):
            utils.precision_at_k(y_true=y_true, y_pred=y_pred, k=k)

    def test_missing_k(self):
        y_true = list(range(10))
        y_pred = list(range(10))

        self.assertEqual(utils.precision_at_k(y_true=y_true, y_pred=y_pred), 1.0)

    def test_gini_equal_distribution(self):
        x = np.array([0, 0, 0, 1, 1, 1])

        self.assertEqual(utils.gini(x), 0.5)

    def test_gini_constant_values(self):
        x = np.array([1, 1, 1, 1, 1, 1])

        self.assertEqual(utils.gini(x), 0.0)

    def test_gini_only_zeros(self):
        x = np.array([0, 0, 0, 0])

        self.assertEqual(utils.gini(x), 0.0)

    def test_gini_not_an_arrayt(self):
        x = 0

        with self.assertRaises(AssertionError):
            utils.gini(x)

    def test_theil_only_zeros(self):
        x = np.array([0, 0, 0, 0])

        self.assertEqual(utils.theil(x), 0.0)

    def test_theil_array(self):
        x = np.array([0, 0, 0, 10])

        self.assertGreater(utils.theil(x), 1.0)

    def test_theil_not_an_array(self):
        x = 0

        with self.assertRaises(AssertionError):
            utils.theil(x)


if __name__ == '__main__':
    unittest.main()
