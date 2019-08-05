import unittest
import networkx as nx

from .. import embed


class EmbedTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.G = nx.barabasi_albert_graph(n=100, m=2)

    def test_embedding(self):
        embedding_size = 200
        node_embeddings = embed.node2vec(self.G, embedding_size=embedding_size)

        vec_number, vec_size = node_embeddings.wv.vectors.shape

        self.assertEqual(vec_number, len(self.G.nodes))
        self.assertEqual(vec_size, embedding_size)


if __name__ == '__main__':

    unittest.main()