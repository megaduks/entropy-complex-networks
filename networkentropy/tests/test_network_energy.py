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

if __name__ == '__main__':

    unittest.main()