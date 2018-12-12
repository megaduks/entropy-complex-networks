import numpy as np
import networkx as nx
import scipy
import scipy.stats

from typing import Dict

from itertools import product

# TODO: add an option to normalize the distribution of energy centrality
# TODO: add typing


def get_randic_matrix(G: nx.Graph) -> np.matrix:
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


def get_randic_index(G: nx.Graph) -> float:
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


def get_randic_energy(G: nx.Graph) -> float:
    """
    Computes the Randić energy of a graph

    Randić energy is the sum of all absolute eigenvalues of the Randić matrix of a graph

    :param G: input graph
    :return: float: Randić energy of a graph
    """

    R = get_randic_matrix(G)
    randic_energy = np.abs(scipy.linalg.eigvals(R).real).sum()

    return randic_energy


def get_randic_spectrum(G: nx.Graph, radius: int=1) -> np.array:
    """
    Computes the spectrum  (i.e. distribution of egonetwork) Randić energy of a graph

    :param G: input graph
    :param radius: radius of the egocentric network
    :return: NumPy array
    """

    result = [
        get_randic_energy(nx.ego_graph(G, v, radius=radius))
        for v in G.nodes
    ]

    return np.asarray(result)


def randic_centrality(G: nx.Graph, radius: int=1):
    """
    Computes the centrality index for each vertex by computing the Randić energy of that vertex's
    neighborhood of a given radius

    :param G: input graph
    :param radius: radius of the egocentric network
    :return: dictionary with Randić energy centrality for each vertex
    """

    result = {n: get_randic_energy(nx.ego_graph(G=G, n=n, radius=radius)) for n in G.nodes}
    return result


def get_laplacian_energy(G: nx.Graph) -> float:
    """
    Computes the energy of the Laplacian of a graph

    :param G: input graph
    :return: float: Laplacian energy of a graph
    """

    L = nx.laplacian_matrix(G).todense()
    eigvals = scipy.linalg.eigvals(L).real
    const = nx.number_of_edges(G) * 2 / nx.number_of_nodes(G)
    consts = np.full(nx.number_of_nodes(G), const)
    laplacian_energy = np.abs(np.subtract(eigvals, consts)).sum()

    return laplacian_energy


def get_laplacian_spectrum(G: nx.Graph, radius: int=1) -> np.array:
    """
    Computes the spectrum of the Laplacian energy of a graph

    :param G: input graph
    :param radius: size of the egocentric network
    :return: NumPy array
    """

    result = [
        get_laplacian_energy(nx.ego_graph(G, v, radius=radius))
        for v in G.nodes
    ]

    return np.asarray(result)


def laplacian_centrality(G: nx.Graph, radius: int=1) -> Dict:
    """
    Computes the centrality index for each vertex by computing the Laplacian energy of that vertex's
    neighborhood of a given radius

    :param G: input graph
    :param radius: radius of the egocentric network
    :return: dictionary with Laplacian energy centrality for each vertex
    """

    result = {n: get_laplacian_energy(nx.ego_graph(G=G, n=n, radius=radius)) for n in G.nodes}
    return result


def get_graph_energy(G: nx.Graph) -> float:
    """
    Computes the energy of the adjacency matrix of a graph

    :param G: input graph
    :return: float: energy of the adjacency matrix of a graph
    """

    M = nx.adjacency_matrix(G).todense()
    graph_energy = np.abs(scipy.linalg.eigvals(M).real).sum()

    return graph_energy


def get_graph_spectrum(G: nx.Graph, radius: int=1) -> np.array:
    """
    Computes the spectrum of the graph energy of a graph

    :param G: input graph
    :param radius: size of the egocentric network
    :return: NumPy array
    """

    result = [
        get_graph_energy(nx.ego_graph(G, v, radius=radius))
        for v in G.nodes
    ]

    return np.asarray(result)


def graph_energy_centrality(G: nx.Graph, radius: int=1) -> Dict:
    """
    Computes the centrality index for each vertex by computing the graph energy of that vertex's
    neighborhood of a given radius

    :param G: input graph
    :param radius: radius of the egocentric network
    :return: dictionary with graph energy centrality for each vertex
    """

    result = {n: get_graph_energy(nx.ego_graph(G=G, n=n, radius=radius)) for n in G.nodes}
    return result

