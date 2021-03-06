from typing import Dict
from tqdm import tqdm

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from networkentropy import network_energy as ne
from networkentropy.embed import node2vec


#TODO: change sample_ratio to accept either an int (absolute) or float (relative)
#TODO: add sampling from embeddings computed on walks driven by graph energy gradients


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
    degree_probs = [d / degree_sum for n, d in graph.degree]
    sample_nodes = np.random.choice(graph.nodes, size=num_nodes, replace=True, p=degree_probs)

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
    pagerank_probs = [p / pagerank_sum for p in dict(nx.pagerank(graph)).values()]
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
    energy_gradient_probs = [p / energy_gradient_sum for p in dict(ne.graph_energy_gradient_centrality(graph)).values()]
    sample_nodes = np.random.choice(graph.nodes, size=num_nodes, replace=False, p=energy_gradient_probs)

    return nx.subgraph(graph, sample_nodes)


def random_graph_energy(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """
    Samples a graph by drawing a sample of nodes with the probability of drawing a node being
    proportional to node's centrality computed from graph energy (energy of node's neighborhood)

    :param graph: input graph
    :param sample_ratio: percentage of the original nodes to be sampled
    :return: a random subgraph
    """

    assert 0 <= sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))
    graph_energy_sum = sum(dict(ne.graph_energy_centrality(graph)).values())
    graph_energy_probs = [p / graph_energy_sum for p in dict(ne.graph_energy_centrality(graph)).values()]
    sample_nodes = np.random.choice(graph.nodes, size=num_nodes, replace=False, p=graph_energy_probs)

    return nx.subgraph(graph, sample_nodes)


def random_laplacian_energy(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """
    Samples a graph by drawing a sample of nodes with the probability of drawing a node being
    proportional to node's centrality computed from the Laplacian energy of node's neighborhood

    :param graph: input graph
    :param sample_ratio: percentage of the original nodes to be sampled
    :return: a random subgraph
    """

    assert 0 <= sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))
    laplacian_energy_sum = sum(dict(ne.laplacian_centrality(graph)).values())
    laplacian_energy_probs = [p / laplacian_energy_sum for p in dict(ne.laplacian_centrality(graph)).values()]
    sample_nodes = np.random.choice(graph.nodes, size=num_nodes, replace=False, p=laplacian_energy_probs)

    return nx.subgraph(graph, sample_nodes)


def random_randic_energy(graph: nx.Graph, sample_ratio: float) -> nx.Graph:
    """
    Samples a graph by drawing a sample of nodes with the probability of drawing a node being
    proportional to node's centrality computed from the Randić energy of node's neighborhood

    :param graph: input graph
    :param sample_ratio: percentage of the original nodes to be sampled
    :return: a random subgraph
    """

    assert 0 <= sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))
    randic_energy_sum = sum(dict(ne.randic_centrality(graph)).values())
    randic_energy_probs = [p / randic_energy_sum for p in dict(ne.randic_centrality(graph)).values()]
    sample_nodes = np.random.choice(graph.nodes, size=num_nodes, replace=False, p=randic_energy_probs)

    return nx.subgraph(graph, sample_nodes)


def random_embedding(graph: nx.Graph, sample_ratio: float, walk_type: str = None) -> nx.Graph:
    """
    Samples a graph by drawing random vectors from the embedding matrix

    :param graph: input graph
    :param sample_ratio: percentage of the original graph nodes to be sampled
    :param walk_type: name of energy for embedding computation, allowed values include 'graph', 'laplacian', 'randic',
            defaunt None generates regular random walk embeddings
    :return: a random subgraph
    """

    assert 0 <= sample_ratio <= 1, 'sample_ratio must be between [0, 1]'

    if walk_type:
        embedded_graph = node2vec(graph, walk_number=100, walk_type=walk_type)
    else:
        embedded_graph = node2vec(graph, walk_number=100)

    num_nodes = int(sample_ratio * nx.number_of_nodes(graph))

    dimension_sums = np.apply_along_axis(np.sum, arr=np.abs(embedded_graph.wv.vectors), axis=0)
    probabilities = np.abs(embedded_graph.wv.vectors) / dimension_sums

    random_vector = np.array([])

    for dim in range(embedded_graph.wv.vectors.shape[-1]):
        col_values = np.squeeze(np.asarray(embedded_graph.wv.vectors[:, dim]))
        col_probs = np.asarray(probabilities[:, dim])
        # normalize probabilities to make sure they sum up to 1.0
        col_probs /= col_probs.sum()
        random_vector = np.append(random_vector, col_values[np.random.choice(col_values.size, size=1, p=col_probs)])

    pass

    sample_nodes = [
        int(node)
        for (node, sim)
        in embedded_graph.wv.similar_by_vector(random_vector, topn=num_nodes)
    ]

    return nx.subgraph(graph, sample_nodes)


def compare_graphs(g1: nx.Graph, g2: nx.Graph) -> Dict:
    """
    Compares the original graph with the sampled graph in terms of basic graph descriptors
    by performing KS-test on normalized distributions

    :param g1: original graph
    :param g2: sampled graph
    :return: dictionary with the results of multiple comparisons
    """
    # TODO: allow to specify which features are to be used in comparison
    # TODO: make sure the function works when empty arrays are passed

    results = {}

    for k, f in {
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
            x = [e / sum(x) for e in x]
        if sum(y) > 0:
            y = [e / sum(y) for e in y]

        stat, p_val = ks_2samp(x, y)

        results.update({k: {'stat': stat, 'p_val': p_val}})

    return results


if __name__ == '__main__':

    results = list()

    num_nodes = 1000

    # iterate over graph model main parameter
    for i in tqdm(range(1, 10)):

        graph_models = {
            'random': (nx.erdos_renyi_graph(n=num_nodes, p=i/100)),
            'smallworld': nx.watts_strogatz_graph(n=num_nodes, k=4, p=i/1000),
            'powerlaw': nx.barabasi_albert_graph(n=num_nodes, m=int(np.ceil(np.log2(i)+0.01)))
        }

        # iterate over graph models
        for graph in tqdm(graph_models):

            g = graph_models[graph]

            functions = {
                'energy_gradient': random_energy_gradient,
                'degree': random_degree,
                'pagerank': random_pagerank,
                'edge': random_edge,
                'node': random_node,
                'embedding': random_embedding,
                'graph_embedding_walk': random_embedding,
                'laplacian_embedding_walk': random_embedding,
                'randic_embedding_walk': random_embedding,
                'energy': random_graph_energy,
                'laplacian': random_laplacian_energy,
                'randic': random_randic_energy
            }

            # iterate over graph sampling methods
            for f in tqdm(functions):

                # iterate over the size of graph sample
                for j in range(1, 20):

                    # sample graph according to the sampling function
                    sg = functions[f](g, sample_ratio=j/100)

                    if f == 'graph_embedding_walk':
                        sg = functions[f](g, sample_ratio=j/100, walk_type='graph')
                    elif f == 'laplacian_embedding_walk':
                        sg = functions[f](g, sample_ratio=j/100, walk_type='laplacian')
                    elif f == 'randic_embedding_walk':
                        sg = functions[f](g, sample_ratio=j/100, walk_type='randic')

                    # check if sampling returned any graph
                    if sg:
                        result = compare_graphs(g, sg)

                        results.append(
                            (
                                graph,
                                f,
                                i,
                                j,
                                result['degree']['p_val'],
                                result['betweenness']['p_val'],
                                result['pagerank']['p_val'],
                                result['closeness']['p_val'],
                                result['clustering']['p_val']
                            )
                        )

    pd.DataFrame(results,
                 columns=['model',
                          'function',
                          'param',
                          'sample_ratio',
                          'degree',
                          'betweenness',
                          'pagerank',
                          'closeness',
                          'clustering']).to_csv('results.csv', index=None)
