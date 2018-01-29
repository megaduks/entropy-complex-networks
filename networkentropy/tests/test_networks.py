import unittest
import networkx as nx
import numpy as np

from .. import networks


class NetworkLoadingTests(unittest.TestCase):

    def test_load_python_dependency(self):
        g = networks.load_network('python_dependency')
        number_of_nodes = g.number_of_nodes()
        number_of_edges = g.number_of_edges()

        self.assertTupleEqual((number_of_nodes, number_of_edges), (26211, 72081))

    def test_load_R_dependency(self):
        g = networks.load_network('R_dependency')
        number_of_nodes = g.number_of_nodes()
        number_of_edges = g.number_of_edges()

        self.assertTupleEqual((number_of_nodes, number_of_edges), (2471, 5451))


    def test_load_st_mark_ecosystem(self):
        g = networks.load_network('st_mark_ecosystem')
        number_of_nodes = g.number_of_nodes()
        number_of_edges = g.number_of_edges()

        self.assertTupleEqual((number_of_nodes, number_of_edges), (54, 356))


    def test_load_power_grid(self):
        g = networks.load_network('power_grid')
        number_of_nodes = g.number_of_nodes()
        number_of_edges = g.number_of_edges()

        self.assertTupleEqual((number_of_nodes, number_of_edges), (4941, 6594))


    def test_load_celegans(self):
        g = networks.load_network('celegans')
        number_of_nodes = g.number_of_nodes()
        number_of_edges = g.number_of_edges()

        self.assertTupleEqual((number_of_nodes, number_of_edges), (453, 4596))


    def test_load_cat_brain(self):
        g = networks.load_network('cat_brain')
        number_of_nodes = g.number_of_nodes()
        number_of_edges = g.number_of_edges()

        self.assertTupleEqual((number_of_nodes, number_of_edges), (65, 1139))


    def test_load_bisons(self):
        g = networks.load_network('bisons')
        number_of_nodes = g.number_of_nodes()
        number_of_edges = g.number_of_edges()

        self.assertTupleEqual((number_of_nodes, number_of_edges), (26, 314))

if __name__ == '__main__':
    unittest.main()
