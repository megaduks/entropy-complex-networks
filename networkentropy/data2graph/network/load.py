import numpy as np
import networkx as nx
from operator import itemgetter


def _count_threshold_by_density(data, density):
    """
    Args:
        data:
            Array with weights
        density:
            Must be from 0 to 1
    """
    v = data.shape[0]

    # number of desired edges
    e = round(density * v * (v - 1) / 2)

    # get indexes for lower triangle without main diagonal(k=-1) and then sort it
    flatten = np.sort(data[np.tril_indices(v, k=-1)], axis=None)
    # sorting descending
    flatten[:] = flatten[::-1]

    return flatten[e]


def load_graph_weight_distance(data, y_label, alg='density', beta=0.1):
    return load_graph(data, y_label, alg, beta, similarity=False)


def load_graph_weight_similarity(data, y_label, alg='density', beta=0.1):
    return load_graph(data, y_label, alg, beta, similarity=True)


def load_graph_weight_distance_no_negative(data, y_label, alg='density', beta=0.1):
    return load_graph_no_negative(data, y_label, alg, beta, similarity=False)


def load_graph_weight_similarity_no_negative(data, y_label, alg='density', beta=0.1):
    return load_graph_no_negative(data, y_label, alg, beta, similarity=True)


def load_graph(data, y_label, alg='density', beta=0.1, similarity=True):
    """

    Args:
        data:
            Main data
        y_label:
            Y data, classes
        alg:
             Algorithm of setting down threshold, pass "denisty" or "percentile"
        beta:
             Parameter used by algorithm "alg" e.g beta=0.1 equals density 0.1
        similarity:
            if true similarity is taken as a weight else it is distance, default True
    Returns:
        Networkx undirected graph with weights based on distances/similarities
    """
    # assert data.shape[0] == data.shape[1]

    if alg == 'avg':
        alpha = np.average(data)
    elif alg == 'percentile':
        alpha = np.percentile(data, beta)
    elif alg == 'density':
        alpha = _count_threshold_by_density(data, beta)

    edges = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[1]):
            if data[i, j] >= alpha:
                # data from 0 to 1, it is similarity
                if similarity:
                    edges[i, j] = data[i, j]
                else:
                    # transform to distance
                    edges[i, j] = 1 - data[i, j]
                if y_label[i] != y_label[j]:
                    edges[i, j] *= -1  # pagerank can not converge
                    # https://stackoverflow.com/questions/19772261/networkx-python-pagerank-numpy-pagerank-fails-but-pagerank-scipy-works

    dt = [('weight', float)]
    A = np.matrix(edges, dtype=dt)
    G = nx.from_numpy_matrix(A)

    # add class labels
    nodes = G.nodes()
    classes = itemgetter(*nodes)(y_label)
    nx.set_node_attributes(G, dict(zip(nodes, classes)), "class")

    return G


def load_graph_no_negative(data, y_label, alg='density', beta=0.1, similarity=True):
    """
    Shifts weights to positive numbers
    Args:
        data:
            Main data
        y_label:
            Y data, classes
        alg:
             Algorithm of setting down threshold, pass "denisty" or "percentile"
        beta:
             Parameter used by algorithm "alg" e.g beta=0.1 equals density 0.1
        similarity:
            if true similarity is taken as a weight else it is distance, default True
    Returns:
        Networkx undirected graph
    """
    G = load_graph(data, y_label, alg, beta, similarity)

    # shift to positive numbers
    # add small constant to avoid division by zero
    w = nx.get_edge_attributes(G, 'weight')
    min_weight = min(w.values())
    if min_weight < 0:
        for u, v, d in G.edges(data=True):
            d['weight'] = d['weight'] - min_weight + 0.00001
    return G
