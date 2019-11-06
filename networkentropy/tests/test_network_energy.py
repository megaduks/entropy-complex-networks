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

    def test_random_walk_laplacian(self):

        g = nx.Graph()
        g.add_edges_from([(1,2),(1,3)])

        expected_matrix = np.array([
            [1, -0.5, -0.5],
            [-1, 1, 0],
            [-1, 0, 1]
        ])

        computed_matrix = network_energy.get_random_walk_laplacian_matrix(g)

        self.assertTrue(np.isclose(expected_matrix, computed_matrix).all())

    def test_symmetric_normalized_laplacian(self):

        g = nx.Graph()
        g.add_edges_from([(1,2),(1,3)])

        expected_matrix = np.array([
            [1.4142135, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        computed_matrix = network_energy.get_symmetric_normalized_laplacian_matrix(g)

        self.assertTrue(np.isclose(expected_matrix, computed_matrix).all())


if __name__ == '__main__':

    unittest.main()