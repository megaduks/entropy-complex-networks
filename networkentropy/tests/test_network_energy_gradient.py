from unittest import TestCase
import networkx as nx
import numpy as np

import networkentropy.network_energy_gradient as neg


class NetworkEnergyGradientTest(TestCase):
    G = None

    def setUp(self):
        graph_matrix = np.array([
            0, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 1,
            0, 0, 0, 0
        ]).reshape((4, 4))
        self.G = nx.from_numpy_array(graph_matrix)

    def test_get_energy_gradient(self):
        G1 = nx.ego_graph(self.G, 0, radius=1)
        G2 = nx.ego_graph(self.G, 1, radius=1)
        actual_graph_gradient = neg.get_energy_gradient(G1, G2, method="graph")
        actual_laplacian_gradient = neg.get_energy_gradient(G1, G2, method="laplacian")
        actual_randic_gradient = neg.get_energy_gradient(G1, G2, method="randic")
        assert actual_graph_gradient == -2.9623886081840283
        assert actual_laplacian_gradient == -4.0
        assert actual_randic_gradient == -0.45742710775633855

    def test_get_energy_gradients(self):
        actual_graph_gradients = neg.get_energy_gradients(self.G, "graph")
        actual_laplacian_gradients = neg.get_energy_gradients(self.G, "laplacian")
        actual_randic_gradients = neg.get_energy_gradients(self.G, "randic")
        expected_graph_gradients = {(0, 1): -2.9623886081840283, (0, 2): -0.9623886081840287,
                                    (0, 3): -0.9623886081840287, (2, 3): 0.0}
        expected_laplacian_gradients = {(0, 1): -4.0, (0, 2): -2.000000000000001,
                                        (0, 3): -2.000000000000001, (2, 3): 0.0}
        expected_randic_gradients = {(0, 1): -0.45742710775633855, (0, 2): -0.45742710775633877,
                                     (0, 3): -0.45742710775633877, (2, 3): 0.0}
        assert actual_graph_gradients == expected_graph_gradients
        assert actual_laplacian_gradients == expected_laplacian_gradients
        assert actual_randic_gradients == expected_randic_gradients
