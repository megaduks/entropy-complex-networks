import unittest
import os
import shutil
import networkx as nx

from networkentropy import network_utils

from typing import Iterable

BRUNSON_SOUTH_AFRICA_TAR_BZ = 'brunson_south-africa.tar.bz2'
BRUNSON_SOUTH_AFRICA_NAME = 'brunson_south-africa'
BRUNSON_SOUTH_AFRICA_TSV_URL = 'http://konect.uni-koblenz.de/downloads/tsv/brunson_south-africa.tar.bz2'
TESTS_DATA_PATH = 'networkentropy/tests/data/'


class NetworkUtilsTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

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
        ge = network_utils.embed_network(g)

        num_rows, emb_size = ge.shape

        self.assertEqual(num_rows, g.number_of_nodes())
        self.assertEqual(emb_size, 128) # TODO: move params of node2vec to a settings file


if __name__ == '__main__':
    unittest.main()
