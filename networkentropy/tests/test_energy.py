import unittest
import networkx as nx
import numpy as np

from networkentropy import energy

RANDIC = 'randic'
LAPLACIAN = 'laplacian'
GRAPH = 'graph'


class EnergyTests(unittest.TestCase):
    g = None

    def setUp(self):
        graph_matrix = np.array([
            0, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 1,
            0, 0, 0, 0
        ]).reshape((4, 4))
        self.g = nx.from_numpy_array(graph_matrix)

    def test_get_energy_gradients(self):
        actual_graph_gradients = neg.get_energy_gradients(self.g, GRAPH, complete=False)
        actual_laplacian_gradients = neg.get_energy_gradients(self.g, LAPLACIAN, complete=False)
        actual_randic_gradients = neg.get_energy_gradients(self.g, RANDIC, complete=False)
        expected_graph_gradients = {(0, 1): -2.9623886081840283, (0, 2): -0.9623886081840287,
                                    (0, 3): -0.9623886081840287, (2, 3): 0.0}
        expected_laplacian_gradients = {(0, 1): -4.0, (0, 2): -2.000000000000001,
                                        (0, 3): -2.000000000000001, (2, 3): 0.0}
        expected_randic_gradients = {(0, 1): -0.45742710775633855, (0, 2): -0.45742710775633877,
                                     (0, 3): -0.45742710775633877, (2, 3): 0.0}
        assert actual_graph_gradients == expected_graph_gradients
        assert actual_laplacian_gradients == expected_laplacian_gradients
        assert actual_randic_gradients == expected_randic_gradients

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.G = nx.star_graph(3)

    def test_randic_matrix(self):

        expected_matrix = np.array([
            [0.,0.57735027,0.57735027,0.57735027],
            [0.57735027,0.,0.,0.],
            [0.57735027,0.,0.,0.],
            [0.57735027,0.,0.,0.]])

        computed_matrix = energy.get_randic_matrix(self.G)

        self.assertTrue(np.isclose(expected_matrix, computed_matrix).all())

    def test_randic_index(self):

        expected_index = 3.464
        computed_index = energy.get_randic_index(self.G)

        self.assertAlmostEqual(expected_index, computed_index, places=3)

    def test_randic_energy(self):

        expected_energy = 2
        computed_energy = energy.get_randic_energy(self.G)

        self.assertAlmostEqual(expected_energy, computed_energy, places=5)

    def test_random_walk_laplacian(self):

        g = nx.Graph()
        g.add_edges_from([(1,2),(1,3)])

        expected_matrix = np.array([
            [1, -0.5, -0.5],
            [-1, 1, 0],
            [-1, 0, 1]
        ])

        computed_matrix = energy.get_random_walk_laplacian_matrix(g)

        self.assertTrue(np.isclose(expected_matrix, computed_matrix).all())

    def test_symmetric_normalized_laplacian(self):

        g = nx.Graph()
        g.add_edges_from([(1,2),(1,3)])

        expected_matrix = np.array([
            [1.4142135, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        computed_matrix = energy.get_symmetric_normalized_laplacian_matrix(g)

        self.assertTrue(np.isclose(expected_matrix, computed_matrix).all())

    def test_dirichlet_energy(self):

        X = np.array([
            [1,1,1,1],
            [0,0,0,1],
            [1,1,0,0]
        ])

        self.assertAlmostEqual(energy.get_dirichlet_energy(X), 3.5)

    def test_dirichlet_energy_wrong_input(self):

        X = np.array([
            1, 1, 1, 1
        ])

        with self.assertRaises(AssertionError):
            energy.get_dirichlet_energy(X)


if __name__ == '__main__':

    unittest.main()
