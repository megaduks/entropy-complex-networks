import unittest
import numpy as np
import os
import shutil
import networkx as nx
from networkentropy import utils

from typing import Iterable

BRUNSON_SOUTH_AFRICA_TAR_BZ = 'brunson_south-africa.tar.bz2'
BRUNSON_SOUTH_AFRICA_NAME = 'brunson_south-africa'
BRUNSON_SOUTH_AFRICA_TSV_URL = 'http://konect.uni-koblenz.de/downloads/tsv/brunson_south-africa.tar.bz2'
TESTS_DATA_PATH = 'networkentropy/tests/data/'


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

        self.assertEqual(y.shape[0], X.shape[0])  # target vector aligned with the training set
        self.assertTrue(np.alltrue(~np.isnan(X)))  # there are no missing values in the dataset

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
        groups = ['bad', 'good']

        X, y, description = utils.load_wine_quality(bins=bins, groups=groups)

        self.assertEqual(y.shape[0], X.shape[0])  # target vector aligned with the training set
        self.assertTrue(np.alltrue(~np.isnan(X)))  # there are no missing values in the dataset
        self.assertEqual(len(np.unique(y)), 2)

    def test_read_avalilable_datasets_konect_cc(self):
        networks = network_utils.read_available_datasets_konect(name='konect.cc').to_list()

        self.assertGreater(len(networks), 0)

    def test_read_avalilable_datasets_konect_uni(self):
        networks = network_utils.read_available_datasets_konect(name='konect.uni').to_list()

        self.assertGreater(len(networks), 0)

    def test_is_iterable_cc(self):
        networks = network_utils.read_available_datasets_konect(name='konect.cc')

        self.assertIsInstance(networks, Iterable)

    def test_is_iterable_uni(self):
        networks = network_utils.read_available_datasets_konect(name='konect.uni')

        self.assertIsInstance(networks, Iterable)

    def test_filter_by_number_of_nodes_cc(self):
        MAX_SIZE = 25
        networks = network_utils.read_available_datasets_konect(name='konect.cc').filter(max_size=MAX_SIZE)
        max_size = max([network_utils.Dataset(*g).num_nodes for g in networks])

        self.assertLessEqual(max_size, MAX_SIZE)

    def test_filter_by_number_of_nodes_uni(self):
        MAX_SIZE = 25
        networks = network_utils.read_available_datasets_konect(name='konect.uni').filter(max_size=MAX_SIZE)
        max_size = max([network_utils.Dataset(*g).num_nodes for g in networks])

        self.assertLessEqual(max_size, MAX_SIZE)

    def test_download_tsv_dataset_konect(self):
        output_file_name = f'{BRUNSON_SOUTH_AFRICA_NAME}-test'
        output_file_name_tar_bz = f'{TESTS_DATA_PATH}{output_file_name}.tar.bz2'
        try:
            network_utils.download_tsv_dataset_konect(output_file_name=output_file_name,
                                                      tsv_url=BRUNSON_SOUTH_AFRICA_TSV_URL,
                                                      dir_name=TESTS_DATA_PATH)
            self.assertTrue(os.path.exists(output_file_name_tar_bz))
        finally:
            shutil.rmtree(output_file_name_tar_bz, ignore_errors=True)

    def test_unpack_tar_bz2_file(self):
        output_dir = f'{TESTS_DATA_PATH}{BRUNSON_SOUTH_AFRICA_NAME}-test'
        try:
            network_utils.unpack_tar_bz2_file(file_name=BRUNSON_SOUTH_AFRICA_TAR_BZ,
                                              dir_name=TESTS_DATA_PATH,
                                              output_dir_name=output_dir)
            self.assertTrue(len(os.listdir(output_dir)) > 0)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_build_network_from_out_konect(self):
        g = network_utils.build_network_from_out_konect(network_name=BRUNSON_SOUTH_AFRICA_NAME,
                                                        tsv_url=BRUNSON_SOUTH_AFRICA_TSV_URL,
                                                        directed=False,
                                                        dir_name=TESTS_DATA_PATH)

        self.assertTrue(g.number_of_nodes() == 6)

    def test_node2vec_embedding(self):
        g = nx.karate_club_graph()
        ge = utils.embed_network(g)

        num_rows, emb_size = ge.shape

        self.assertEqual(num_rows, g.number_of_nodes())
        self.assertEqual(emb_size, 128)  # TODO: move params of node2vec to a settings file


if __name__ == '__main__':
    unittest.main()
