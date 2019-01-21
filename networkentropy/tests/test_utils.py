import unittest
import os

from .. import utils

class UtilsTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_read_avalilable_datasets_konect(self):

        networks = utils.read_avalilable_datasets_konect()

        self.assertGreater(len(networks), 0)
        
    def test_download_tsv_dataset_konect(self):

        network_name = 'brunson_south-africa'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        utils.download_tsv_dataset_konect(network_name=network_name, dir_name=dir_name)

        self.assertTrue(os.path.exists(dir_name + network_name + '.tar.bz2'))

    def test_unpack_tar_bz2_file(self):

        file_name = 'brunson_south-africa.tar.bz2'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        utils.unpack_tar_bz2_file(file_name=file_name, dir_name=dir_name)

        self.assertTrue(os.path.exists(dir_name + 'network_' + file_name.replace('.tar.bz2','')))

    def test_build_network_from_out_konect(self):

        network_name = 'brunson_south-africa'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        g = utils.build_network_from_out_konect(network_name=network_name, dir_name=dir_name)

        self.assertTrue(g.number_of_nodes() > 0)

    def test_build_network_from_out_konect_with_min_size_limit(self):

        network_name = 'brunson_south-africa'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        g = utils.build_network_from_out_konect(network_name=network_name,
                                                dir_name=dir_name,
                                                num_nodes=11,
                                                num_edges=13,
                                                min_size=30)

        self.assertTrue(g is None)

    def test_build_network_from_out_konect_with_max_size_limit(self):

        network_name = 'brunson_south-africa'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        g = utils.build_network_from_out_konect(network_name=network_name,
                                                dir_name=dir_name,
                                                num_nodes=11,
                                                num_edges=30,
                                                max_size=10)

        self.assertTrue(g is None)

    def test_build_network_from_out_konect_with_max_density_limit(self):

        network_name = 'brunson_south-africa'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        g = utils.build_network_from_out_konect(network_name=network_name,
                                                dir_name=dir_name,
                                                num_nodes=11,
                                                num_edges=13,
                                                max_density=0.1)

        self.assertTrue(g is None)

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


if __name__ == '__main__':

    unittest.main()

