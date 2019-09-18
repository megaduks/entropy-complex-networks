import numpy as np
from collections import Counter
import math
import networkx as nx
# from scipy.special import exp
from functools import reduce


def _get_list_from_dict(d):
    return np.array([value for key, value in sorted(dict.items(d))])


def _sigmoid(w):
    return expit(w)


def _second_neighbors(G, node):
    """
    Returns second neighbors of node in graph
    Neighbors are uniques!
    """
    snds = []
    for edges_list in [G.edges(n) for n in G.neighbors(node)]:
        snds.extend(edges_list)

    snds = map(lambda x: x if x[0] <= x[1] else (x[1], x[0]), snds)
    snds = set(snds)
    return snds


def weight_by_pagerank(G):
    """
    Pagerank

    Raises:
        Can not converge
    """
    pg = nx.pagerank(G, weight='weight')
    return _get_list_from_dict(pg)


def weight_by_hits_authorities(G):
    # already uses weights when named "weight"
    # see networkx code
    return _get_list_from_dict(nx.hits(G)[1])


def weight_by_hits_hubs(G):
    # already uses weights when named "weight"
    # see networkx code
    return _get_list_from_dict(nx.hits(G)[0])


def weight_by_eigenvector_centrality(G):
    """
    Eigenvector centrality

    Raises:
        Can not converge
    """
    centrality = nx.eigenvector_centrality(G, weight='weight')
    return _get_list_from_dict(centrality)


def weight_by_centrality(G):
    centrality = nx.degree_centrality(G)
    return _get_list_from_dict(centrality)


def weight_by_betweenness_centrality(G):
    centrality = nx.betweenness_centrality(G, weight='weight')
    return _get_list_from_dict(centrality)


def weight_by_entropy(G):
    weights = np.zeros(G.number_of_nodes(), dtype=float)
    for g in G.nodes():
        neighbors = [e for _, e in G.edges(g)]
        classes = [G.node[n]['class'] for n in neighbors]
        if classes:
            entropy = 0
            # count entropy of node
            counter = Counter(classes)
            for cls in counter:
                p = counter[cls] / len(classes)
                entropy += p * math.log(p, 2)
            entropy *= -1
            # get I
            i = 1 if counter.most_common(1)[0][0] == G.node[g]['class'] else -1
            weights[g] = 1 / (i * (entropy + 1))
        else:
            # no neighbours, weight 1
            weights[g] = 1
    return weights


def weight_by_random(G):
    weights = np.zeros(G.number_of_nodes(), dtype=float)
    return np.random.rand(weights.shape[0])


def weight_by_katz(G):
    kz = nx.katz_centrality(G, weight='weight')
    return _get_list_from_dict(kz)


def weight_by_degree(G):
    dg = G.degree(weight='weight')

    return _get_list_from_dict(dict(dg))


def weight_by_degree_k_2(G):
    weights = np.zeros(G.number_of_nodes(), dtype=float)
    for g in G.nodes():
        weight = reduce(lambda curr, n: curr + G[n[0]][n[1]]['weight'], _second_neighbors(G, g), 0.0)
        weights[g] = weight

    return weights



