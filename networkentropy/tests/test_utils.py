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

    def test_normalize_dict(self):
        d = {0: 10, 1: 30, 2: 10}

        self.assertEqual(sum(utils.normalize_dict(d).values()), 1.0)
        self.assertEqual(list(utils.normalize_dict(d).values()), [0.2, 0.6, 0.2])

    def test_normalize_dict_with_factor(self):
        d = {0: 10, 1: 30, 2: 10}

        self.assertEqual(sum(utils.normalize_dict(d, target=2).values()), 2.0)
        self.assertEqual(list(utils.normalize_dict(d, target=2).values()), [0.4, 1.2, 0.4])

    def test_normalize_dict_without_dict(self):
        d = [0, 1, 2]

        with self.assertRaises(AssertionError):
            utils.normalize_dict(d)

    def test_load_iris(self):
        X, y, description = utils.load_iris()

        self.assertEqual(y.shape[0], X.shape[0]) # target vector aligned with the training set
        self.assertTrue(np.alltrue(~np.isnan(X))) # there are no missing values in the dataset

    def test_load_titanic(self):
        X, y, description = utils.load_titanic()

        self.assertEqual(y.shape[0], X.shape[0])  # target vector aligned with the training set

    def test_load_lenses(self):
        X, y, description = utils.load_lenses()

        self.assertEqual(y.shape[0], X.shape[0])  # target vector aligned with the training set
        self.assertTrue(np.alltrue(~np.isnan(X)))  # there are no missing values in the dataset

    def test_load_breast_cancer(self):
        X, y, description = utils.load_breast_cancer_short()

        self.assertEqual(y.shape[0], X.shape[0])  # target vector aligned with the training set
        self.assertTrue(np.alltrue(~np.isnan(X)))  # there are no missing values in the dataset

    def test_load_housing_prices(self):
        X, y, description = utils.load_housing_prices_short()

        self.assertEqual(y.shape[0], X.shape[0])  # target vector aligned with the training set
        self.assertTrue(np.alltrue(~np.isnan(X)))  # there are no missing values in the dataset

    def test_load_wine_quality(self):
        X, y, description = utils.load_wine_quality()

        self.assertEqual(y.shape[0], X.shape[0])  # target vector aligned with the training set
        self.assertTrue(np.alltrue(~np.isnan(X)))  # there are no missing values in the dataset
        self.assertGreater(len(np.unique(y)), 3)

    def test_load_wine_quality_with_binning(self):
        bins = (2, 4.5, 6)
        groups = ['bad','good']

        X, y, description = utils.load_wine_quality(bins=bins, groups=groups)

        self.assertEqual(y.shape[0], X.shape[0])  # target vector aligned with the training set
        self.assertTrue(np.alltrue(~np.isnan(X)))  # there are no missing values in the dataset
        self.assertEqual(len(np.unique(y)), 2)


if __name__ == '__main__':
    unittest.main()
