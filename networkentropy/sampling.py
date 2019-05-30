from typing import Dict

import networkx as nx
import numpy as np

from node2vec import Node2Vec
from scipy.stats import ks_2samp

from networkentropy import network_energy as ne


# TODO: change sample_ratio to accept either an int (absolute) or float (relative)


def random_node(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """"
    Samples a graph by drawing a random sample of nodes and all adjacent edges

    :param graph: input graph
    :param sample_ratio: percentage of the original graph nodes to be sampled
    :return a random subgraph
    """

    assert 0 <= sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

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

    assert 0 <= sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

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

    assert 0 <= sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

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

    assert 0 <= sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))
    pagerank_sum = sum(dict(nx.pagerank(graph)).values())
    pagerank_probs = [p/pagerank_sum for p in dict(nx.pagerank(graph)).values()]
    sample_nodes = np.random.choice(graph.nodes, size=num_nodes, replace=False, p=pagerank_probs)

    return nx.subgraph(graph, sample_nodes)


def random_energy_gradient(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """
    Samples a graph by drawing a sample of nodes with the probability of drawing a node being
    proportional to node's centrality computed from a random walk directed by energy gradients

    :param graph: input graph
    :param sample_ratio: percentage of the original nodes to be sampled
    :return: a random subgraph
    """

    assert 0 <= sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))
    energy_gradient_sum = sum(dict(ne.graph_energy_gradient_centrality(graph)).values())
    energy_gradient_probs = [p/energy_gradient_sum for p in dict(ne.graph_energy_gradient_centrality(graph)).values()]
    sample_nodes = np.random.choice(graph.nodes, size=num_nodes, replace=False, p=energy_gradient_probs)

    return nx.subgraph(graph, sample_nodes)


def random_embedding(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """
    Samples a graph by drawing random vectors from the embedding matrix

    :param graph: input graph
    :param sample_ratio: percentage of the original graph nodes to be sampled
    :return: a random subgraph
    """

    assert 0 <= sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    embedded_graph = Node2Vec(graph).fit()
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


def compare_graphs(g1: nx.Graph, g2: nx.Graph) -> Dict :
    """
    Compares the original graph with the sampled graph in terms of basic graph descriptors
    by performing KS-test on normalized distributions

    :param g1: original graph
    :param g2: sampled graph
    :return: dictionary with the results of multiple comparisons
    """
    #TODO: allow to specify which features are to be used in comparison

    results = {}

    for k,f in {
        'degree': nx.degree_centrality,
        'betweenness': nx.betweenness_centrality,
        'pagerank': nx.pagerank,
        'closeness': nx.closeness_centrality,
        'clustering': nx.clustering
    }.items():

        if k in g1.graph:
            x = g1.graph[k]
        else:
            x = list(f(g1).values())
            g1.graph[k] = x

        y = list(f(g2).values())

        # normalize distributions
        if sum(x) > 0:
            x = [e/sum(x) for e in x]
        if sum(y) > 0:
            y = [e/sum(y) for e in y]

        stat, p_val = ks_2samp(x,y)

        results.update({k: {'stat': stat, 'p_val': p_val}})

    return results


if __name__ == '__main__':

    g = nx.erdos_renyi_graph(100,0.1)

    for i in range(1,100):
        gg = random_energy_gradient(g, sample_ratio=i/100)
        result = compare_graphs(g, gg)

        print(f"{i}: degree: {result['degree']['p_val']:.4f} "
              f"betweenness: {result['betweenness']['p_val']:.4f} "
              f"pagerank: {result['pagerank']['p_val']:.4f} "
              f"closeness: {result['closeness']['p_val']:.4f} "
              f"clustering: {result['clustering']['p_val']:.4f} "
              )