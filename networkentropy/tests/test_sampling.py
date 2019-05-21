import unittest
import networkx as nx

from .. import sampling


class SamplingTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.G = nx.barabasi_albert_graph(n=100, m=2)

    def test_random_nodes(self):
        sample_ratio = 0.1
        sample_graph = sampling.random_node(self.G, sample_ratio)

        self.assertAlmostEqual(nx.number_of_nodes(sample_graph), int(sample_ratio * nx.number_of_nodes(self.G)))

    def test_random_edges(self):
        sample_ratio = 0.1
        sample_graph = sampling.random_edge(self.G, sample_ratio)

        self.assertAlmostEqual(nx.number_of_edges(sample_graph), int(sample_ratio * nx.number_of_edges(self.G)))

if __name__ == '__main__':

    unittest.main()