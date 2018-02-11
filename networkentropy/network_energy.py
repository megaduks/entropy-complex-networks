import numpy as np
import networkx as nx
import scipy
import scipy.stats

from itertools import product

from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt


def get_randic_matrix(G):
    """
    Computes the Randić matrix of a graph

    Elements of the Randić matrix are defined as follows:

               0 if v=w
    R(v,w) =   1/sqrt(D(v)*D(w)) if (v,w) in E(G)
               0 otherwise

    :param G: input graph
    :return: NumPy matrix
    """

    D = nx.degree(G)

    randic_values = [1 / np.sqrt(D[v] * D[w]) if G.has_edge(v, w) else 0 for (v, w) in product(G.nodes, G.nodes)]
    randic_matrix = np.matrix(randic_values).reshape(G.number_of_nodes(), G.number_of_nodes())

    return randic_matrix


def get_randic_index(G):
    """
    Computes the Randić index of a graph

    Randić index is the sum of non-zero elements of the Randić matrix of a graph

    :param G: input graph
    :return: float: Randić index of a graph
    """

    D = nx.degree(G)

    randic_values = [1 / np.sqrt(D[v] * D[w]) if G.has_edge(v, w) else 0 for (v, w) in product(G.nodes, G.nodes)]
    randic_index = sum(randic_values)

    return randic_index


def get_randic_energy(G):
    """
    Computes the Randić energy of a graph

    Randić energy is the sum of all absolute eigenvalues of the Randić matrix of a graph

    :param G: input graph
    :return: float: Randić energy of a graph
    """

    R = get_randic_matrix(G)
    randic_energy = np.abs(scipy.linalg.eigvals(R).real).sum()

    return randic_energy


def get_laplacian_energy(G):
    """
    Computes the energy of the Laplacian of a graph

    :param G: input graph
    :return: float: Laplacian energy of a graph
    """

    L = nx.laplacian_matrix(G).todense()
    laplacian_energy = np.abs(scipy.linalg.eigvals(L).real).sum()

    return laplacian_energy


def get_graph_energy(G):
    """
    Computes the energy of the adjacency matrix of a graph

    :param G: input graph
    :return: float: energy of the adjacency matrix of a graph
    """

    M = nx.adjacency_matrix(G).todense()
    graph_energy = np.abs(scipy.linalg.eigvals(M).real).sum()

    return graph_energy

def get_distance_matrix(G):
    """
    Computes the distance matrix of a graph

    :param G: input graph
    :return: matrix with elements representing shortest paths between nodes
    """

    distance_matrix = np.zeros(G.number_of_nodes() * G.number_of_nodes()).\
        reshape(G.number_of_nodes(), G.number_of_nodes())
    distance_values = [(x,y,d[y]) for (x,d) in nx.all_pairs_dijkstra_path_length(G) for y in d.keys()]

    for (x,y,d) in distance_values:
        distance_matrix[x,y] = d

    return distance_matrix


def get_distance_energy(G):
    """
    Computes the distance energy of a graph

    Distance energy is the sum of all absolute eigenvalues of the distance matrix of a graph

    :param G: input graph
    :return: float: distance energy of a graph
    """

    D = get_distance_matrix(G)
    distance_energy = np.abs(scipy.linalg.eigvals(D).real).sum()

    return distance_energy

# TODO add the retrieval of spectra from a graph

if __name__ == '__main__':

    entropies = []

    for p in tqdm(range(0,100, 10)):

        # G = nx.erdos_renyi_graph(n=100, p=p/100)
        G = nx.watts_strogatz_graph(n=100, k=2, p=p/100)
        # G = nx.powerlaw_cluster_graph(n=100, m=4, p=p/100)
        # G = nx.random_lobster(n=100, p1=p/100, p2=p/100)

        results = []

        for n in G.nodes:
            g = nx.ego_graph(G=G, n=n)

            E_randic = get_randic_energy(g)
            E_graph = get_graph_energy(g)
            E_laplace = get_laplacian_energy(g)

            results.append((E_randic, E_graph, E_laplace))

        e_randic, e_graph, e_laplace = map(list, zip(*results))

        randic_cnt, _ = np.histogram(e_randic)
        graph_cnt, _ = np.histogram(e_graph)
        laplace_cnt, _ = np.histogram(e_laplace)

        randic_entropy = scipy.stats.entropy(randic_cnt)
        graph_entropy = scipy.stats.entropy(graph_cnt)
        laplace_entropy = scipy.stats.entropy(laplace_cnt)

        entropies.append((randic_entropy, graph_entropy, laplace_entropy))

    y1, y2, y3 = map(list, zip(*entropies))

    x = range(0,100,10)

    fig, ax = plt.subplots()
    ax.plot(x, y1, color='red', label='randic')
    ax.plot(x, y2, color='green', label='graph')
    ax.plot(x, y3, color='blue', label='laplace')

    plt.show()

    print(pd.DataFrame({'x':x, 'randic': y1, 'graph': y2, 'laplace': y3}))
