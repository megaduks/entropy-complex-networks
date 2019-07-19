import unittest
import os
import shutil

from .. import network_utils

BRUNSON_SOUTH_AFRICA_TAR_BZ = 'brunson_south-africa.tar.bz2'

BRUNSON_SOUTH_AFRICA_NAME = 'brunson_south-africa'

BRUNSON_SOUTH_AFRICA_TSV_URL = 'http://konect.uni-koblenz.de/downloads/tsv/brunson_south-africa.tar.bz2'

TESTS_DATA_PATH = 'networkentropy/tests/data/'


class UtilsTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_read_avalilable_datasets_konect(self):
        networks = network_utils.read_available_datasets_konect(name='konect.cc')

        self.assertGreater(len(networks), 0)

    def _test_download_tsv_dataset_konect(self):
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

        self.assertTrue(g.number_of_nodes() > 0)

    def test_precision_at_k_full_coverage(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 8, 9]
        k = 3

        self.assertEqual(network_utils.precision_at_k(y_true=y_true, y_pred=y_pred, k=k), 1.0)

    def test_precision_at_k_partial_coverage(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 8, 9]
        k = 4

        self.assertEqual(network_utils.precision_at_k(y_true=y_true, y_pred=y_pred, k=k), 0.75)

    def test_precision_at_k_no_coverage(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [9, 8, 7, 6, 5]
        k = 3

        self.assertEqual(network_utils.precision_at_k(y_true=y_true, y_pred=y_pred, k=k), 0.0)

    def test_precision_at_k_empty_list(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = []
        k = 3

        self.assertEqual(network_utils.precision_at_k(y_true=y_true, y_pred=y_pred, k=k), 0.0)

    def test_precision_at_k_no_list(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = None
        k = 3

        with self.assertRaises(AssertionError):
            network_utils.precision_at_k(y_true=y_true, y_pred=y_pred, k=k)


if __name__ == '__main__':
    unittest.main()
