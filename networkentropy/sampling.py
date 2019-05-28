import networkx as nx
import numpy as np

from .embed import node2vec

from typing import List, Dict


# TODO: change sample_ratio to accept either an int (absolute) or float (relative)
# TODO: verify if sampling from embeddings is correct (doesn't seem to be the case)

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


def random_degree(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """
    Samples a graph by drawing a sample of nodes with the probability of drawing a node being
    proportional to node's degree

    :param graph: input graph
    :param sample_ratio: percentage of the original nodes to be sampled
    :return: a random subgraph
    """

    assert sample_ratio >= 0, 'sample_ratio must be between [0, 1]'
    assert sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))
    degree_sum = sum(dict(graph.degree).values())
    degree_probs = [d/degree_sum for n,d in graph.degree]
    sample_nodes = np.random.choice(graph.nodes, size=num_nodes, replace=False, p=degree_probs)

    return nx.subgraph(graph, sample_nodes)


def random_pagerank(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """
    Samples a graph by drawing a sample of nodes with the probability of drawing a node being
    proportional to node's pagerank

    :param graph: input graph
    :param sample_ratio: percentage of the original nodes to be sampled
    :return: a random subgraph
    """

    assert sample_ratio >= 0, 'sample_ratio must be between [0, 1]'
    assert sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))
    pagerank_sum = sum(dict(nx.pagerank(graph)).values())
    pagerank_probs = [p/pagerank_sum for p in dict(nx.pagerank(graph)).values()]
    sample_nodes = np.random.choice(graph.nodes, size=num_nodes, replace=False, p=pagerank_probs)

    return nx.subgraph(graph, sample_nodes)


def random_embedding(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """
    Samples a graph by drawing random vectors from the embedding matrix

    :param graph: input graph
    :param sample_ratio: percentage of the original graph nodes to be sampled
    :return: a random subgraph
    """

    assert sample_ratio >= 0, 'sample_ratio must be between [0, 1]'
    assert sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    embedded_graph = node2vec(graph)
    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))

    sample_nodes = list()

    while len(sample_nodes) < num_nodes:

        dimension_sums = np.apply_along_axis(np.sum, arr=np.abs(embedded_graph.wv.vectors), axis=0)
        probabilities = np.abs(embedded_graph.wv.vectors) / dimension_sums

        random_vector = np.array([])

        for dim in range(embedded_graph.wv.vectors.shape[-1]):
            col_values = np.squeeze(np.asarray(embedded_graph.wv.vectors[:, dim]))
            col_probs = np.asarray(probabilities[:,dim])
            # normalize probabilities to make sure they sum up to 1.0
            col_probs /= col_probs.sum()
            random_vector = np.append(random_vector, np.random.choice(col_values, size=1, p=col_probs))

        node, sim = embedded_graph.wv.similar_by_vector(random_vector, topn=1)[0]

        if int(node) not in sample_nodes:
            sample_nodes.append(int(node))

    return nx.subgraph(graph, sample_nodes)


if __name__ == '__main__':

    g = nx.barabasi_albert_graph(1000,2)
    gg = random_edge(g, 0.1)

    print(f'number of nodes: {nx.number_of_nodes(g)} and edges: {nx.number_of_edges(g)}')
    print(f'number of sample nodes: {nx.number_of_nodes(gg)} and edges: {nx.number_of_edges(gg)}')