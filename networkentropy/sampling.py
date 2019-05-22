import networkx as nx
import numpy as np

import embed

from typing import List, Dict


# TODO: change sample_ratio to accept either an int (absolute) or float (relative)

def random_node(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """"
    Samples a graph by drawing a random sample of nodes and all adjacent edges

    :param graph: input graph
    :param sample_ratio: percentage of the original graph nodes to be sampled
    :return a random subgraph
    """

    assert sample_ratio >= 0, 'sample_ratio must be between [0, 1]'
    assert sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))
    sample_nodes = np.random.choice(graph.nodes, num_nodes, replace=False)

    return nx.subgraph(graph, sample_nodes)


def random_edge(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """
    Samples a graph by drawing a random sample of edges and all adjacent nodes

    :param graph: input graph
    :param sample_ratio: percentage of the original edges to be sampled
    :return: a random subgraph
    """

    assert sample_ratio >= 0, 'sample_ratio must be between [0, 1]'
    assert sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    num_edges = int(sample_ratio * nx.number_of_edges(graph))
    sample_edges_idx = np.random.choice(len(graph.edges), num_edges, replace=False)
    sample_edges = [
        e
        for (i, e) in enumerate(graph.edges)
        if i in sample_edges_idx
    ]

    return nx.Graph(sample_edges)


def random_embedding(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """
    Samples a graph by drawing random vectors from the embedding matrix

    :param graph: input graph
    :param sample_ratio: percentage of the original graph nodes to be sampled
    :return: a random subgraph
    """

    assert sample_ratio >= 0, 'sample_ratio must be between [0, 1]'
    assert sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    embedded_graph = embed.node2vec(graph)
    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))

    sample_nodes = list()

    while len(sample_nodes) < num_nodes:

        random_vector = np.apply_along_axis(np.random.choice, axis=0, arr=embedded_graph.wv.vectors, size=1)[0]
        node, sim = embedded_graph.wv.similar_by_vector(random_vector, topn=1)[0]

        if int(node) not in sample_nodes:
            sample_nodes.append(int(node))

    return nx.subgraph(graph, sample_nodes)


if __name__ == '__main__':

    g = nx.barabasi_albert_graph(100,2)
    gg = random_embedding(g, 0.1)

    print(g)

    print(gg)