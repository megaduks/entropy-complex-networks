import unittest
import os

from .. import utils

class UtilsTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_read_avalilable_datasets_konect(self):

        network_names = utils.read_avalilable_datasets_konect()

        self.assertGreater(len(network_names), 0)
        
    def test_download_tsv_dataset_konect(self):

        network_name = 'moreno_bison'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        utils.download_tsv_dataset_konect(network_name=network_name, dir_name=dir_name)

        self.assertTrue(os.path.exists(dir_name + network_name + '.tar.bz2'))

    def test_unpack_tar_bz2_file(self):

        file_name = 'moreno_bison.tar.bz2'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        utils.unpack_tar_bz2_file(file_name=file_name, dir_name=dir_name)

        self.assertTrue(os.path.exists(dir_name + 'network_' + file_name.replace('.tar.bz2','')))

    def test_build_network_from_out_konect(self):

        network_name = 'moreno_bison'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        g = utils.build_network_from_out_konect(network_name=network_name, dir_name=dir_name)

        self.assertTrue(g.number_of_nodes() > 0)

    def test_build_network_from_out_konect_with_min_size_limit(self):

        network_name = 'moreno_bison'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        g = utils.build_network_from_out_konect(network_name=network_name, dir_name=dir_name, min_size=30)

        self.assertTrue(g is None)

    def test_build_network_from_out_konect_with_max_size_limit(self):

        network_name = 'moreno_bison'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        g = utils.build_network_from_out_konect(network_name=network_name, dir_name=dir_name, max_size=10)

        self.assertTrue(g is None)

    def test_build_network_from_out_konect_with_max_density_limit(self):

        network_name = 'moreno_bison'
        dir_name = '/home/mikolaj/Research/entropy-complex-networks/networkentropy/data/'

        g = utils.build_network_from_out_konect(network_name=network_name, dir_name=dir_name, max_density=0.1)

        self.assertTrue(g is None)


if __name__ == '__main__':

    unittest.main()

