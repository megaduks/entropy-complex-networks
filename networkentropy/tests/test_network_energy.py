import unittest
import networkx as nx
import numpy as np

from .. import network_energy

class NetworkEnergyTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.G = nx.star_graph(3)

    def test_randic_matrix(self):

        expected_matrix = np.array([
            [0.,0.57735027,0.57735027,0.57735027],
            [0.57735027,0.,0.,0.],
            [0.57735027,0.,0.,0.],
            [0.57735027,0.,0.,0.]])

        computed_matrix = network_energy.get_randic_matrix(self.G)

        self.assertTrue(np.isclose(expected_matrix, computed_matrix).all())

    def test_randic_index(self):

        expected_index = 3.464
        computed_index = network_energy.get_randic_index(self.G)

        self.assertAlmostEqual(expected_index, computed_index, places=3)

    def test_randic_energy(self):

        expected_energy = 2
        computed_energy = network_energy.get_randic_energy(self.G)

        self.assertAlmostEqual(expected_energy, computed_energy, places=5)

    def test_randic_centrality_normalization(self):

        expected_sum_centrality = 1
        computed_sum_centrality = sum(network_energy.randic_centrality(self.G, normalized=True).values())

        self.assertEqual(expected_sum_centrality, computed_sum_centrality)

    def test_laplacian_centrality_normalization(self):

        expected_sum_centrality = 1
        computed_sum_centrality = sum(network_energy.laplacian_centrality(self.G, normalized=True).values())

        self.assertEqual(expected_sum_centrality, computed_sum_centrality)

    def test_graph_centrality_normalization(self):

        expected_sum_centrality = 1
        computed_sum_centrality = sum(network_energy.graph_energy_centrality(self.G, normalized=True).values())

        self.assertEqual(expected_sum_centrality, computed_sum_centrality)

    def test_gradient_centrality_normalization(self):

        expected_sum_centrality = 1
        computed_sum_centrality = sum(network_energy.gradient_centrality(self.G, normalized=True).values())

        self.assertEqual(expected_sum_centrality, computed_sum_centrality)


if __name__ == '__main__':

    unittest.main()
