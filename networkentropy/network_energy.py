from builtins import object

import numpy as np
import networkx as nx
import scipy
import scipy.stats

from typing import Dict, List

from itertools import product

# TODO: add an option to normalize the distribution of energy centrality
# TODO: add typing to the entire project
# TODO: add functions to compute energy gradients for graph energy
# TODO: add functions to compute energy gradients for Randić energy
# TODO: add functions to compute energy gradients for Laplacian energy
# TODO: add function to compute pagerank-like centrality based on graph energy gradients



def get_randic_matrix(g: object) -> np.matrix:
    """
    Computes the Randić matrix of a graph

    Elements of the Randić matrix are defined as follows:

               0 if v=w
    R(v,w) =   1/sqrt(D(v)*D(w)) if (v,w) in E(g)
               0 otherwise

    :param g: input graph
    :return: NumPy matrix
    """

    D = nx.degree(g)

    randic_values = [1 / np.sqrt(D[v] * D[w]) if g.has_edge(v, w) else 0 for (v, w) in product(g.nodes, g.nodes)]
    randic_matrix = np.matrix(randic_values).reshape(g.number_of_nodes(), g.number_of_nodes())

    return randic_matrix


def get_randic_index(g: object) -> float:
    """
    Computes the Randić index of a graph

    Randić index is the sum of non-zero elements of the Randić matrix of a graph

    :param g: input graph
    :return: float: Randić index of a graph
    """

    D = nx.degree(g)

    randic_values = [1 / np.sqrt(D[v] * D[w]) if g.has_edge(v, w) else 0 for (v, w) in product(g.nodes, g.nodes)]
    randic_index = sum(randic_values)

    return randic_index


def get_randic_energy(g: object) -> float:
    """
    Computes the Randić energy of a graph

    Randić energy is the sum of all absolute eigenvalues of the Randić matrix of a graph

    :param g: input graph
    :return: float: Randić energy of a graph
    """

    R = get_randic_matrix(g)
    randic_energy = np.abs(scipy.linalg.eigvals(R).real).sum()

    return randic_energy


def get_randic_spectrum(g: object, radius: int = 1) -> np.array:
    """
    Computes the spectrum  (i.e. distribution of egonetwork) Randić energy of a graph

    :param g: input graph
    :param radius: radius of the egocentric network
    :return: NumPy array
    """

    result = [
        get_randic_energy(nx.ego_graph(G=g, n=n, radius=radius))
        for n in g.nodes
    ]

    return np.asarray(result)


def randic_centrality(g: object, radius: int = 1):
    """
    Computes the centrality index for each vertex by computing the Randić energy of that vertex's
    neighborhood of a given radius

    :param g: input graph
    :param radius: radius of the egocentric network
    :return: dictionary with Randić energy centrality for each vertex
    """

    result = {n: get_randic_energy(nx.ego_graph(G=g, n=n, radius=radius)) for n in g.nodes}
    return result


def get_laplacian_energy(g: object) -> float:
    """
    Computes the energy of the Laplacian of a graph

    :param g: input graph
    :return: float: Laplacian energy of a graph
    """

    L = nx.laplacian_matrix(g).todense()
    eigvals = scipy.linalg.eigvals(L).real
    const = nx.number_of_edges(g) * 2 / nx.number_of_nodes(g)
    consts = np.full(nx.number_of_nodes(g), const)
    laplacian_energy = np.abs(np.subtract(eigvals, consts)).sum()

    return laplacian_energy


def get_laplacian_spectrum(g: object, radius: int = 1) -> np.array:
    """
    Computes the spectrum of the Laplacian energy of a graph

    :param g: input graph
    :param radius: size of the egocentric network
    :return: NumPy array
    """

    result = [
        get_laplacian_energy(nx.ego_graph(G=g, n=n, radius=radius))
        for n in g.nodes
    ]

    return np.asarray(result)


def laplacian_centrality(g: object, radius: int = 1) -> Dict:
    """
    Computes the centrality index for each vertex by computing the Laplacian energy of that vertex's
    neighborhood of a given radius

    :param g: input graph
    :param radius: radius of the egocentric network
    :return: dictionary with Laplacian energy centrality for each vertex
    """

    result = {n: get_laplacian_energy(nx.ego_graph(G=g, n=n, radius=radius)) for n in g.nodes}
    return result


def get_graph_energy(g: object) -> float:
    """
    Computes the energy of the adjacency matrix of a graph

    :param g: input graph
    :return: float: energy of the adjacency matrix of a graph
    """

    M = nx.adjacency_matrix(g).todense()
    graph_energy = np.abs(scipy.linalg.eigvals(M).real).sum()

    return graph_energy


def get_graph_spectrum(g: object, radius: int = 1) -> np.array:
    """
    Computes the spectrum of the graph energy of a graph

    :param g: input graph
    :param radius: size of the egocentric network
    :return: NumPy array
    """

    result = [
        get_graph_energy(nx.ego_graph(G=g, n=n, radius=radius))
        for n in g.nodes
    ]

    return np.asarray(result)


def graph_energy_centrality(g: object, radius: int = 1) -> Dict:
    """
    Computes the centrality index for each vertex by computing the graph energy of that vertex's
    neighborhood of a given radius

    :param g: input graph
    :param radius: radius of the egocentric network
    :return: dictionary with graph energy centrality for each vertex
    """

    result = {n: get_graph_energy(nx.ego_graph(G=g, n=n, radius=radius)) for n in g.nodes}
    return result


def get_graph_energy_gradients(g: object, energy_dist: List[float] = None) -> Dict:
    """
    Compute gradients of graph energy for all nodes

    :param g: input graph
    :param energy_dist: precomputed distribution of energy in the graph g
    :return: dictionary with graph energy differences for each node
    """

    if energy_dist is None:
        energy_dist = get_graph_spectrum(g)

    result = {
        n: {
            nn: energy_dist[n] - energy_dist[nn]
            for nn
            in nx.ego_graph(G=g, n=n)
            if nn != n
        }
        for n
        in g.nodes
    }

    return result


def get_max_graph_energy_gradient(energy_gradients: Dict) -> List[int]:
    """
    Finds the list of nodes representing the vector of max gradients. For each node the list contains
    the label of node's neighbor with the maximum energy gradient

    :param energy_gradients: dictionary with all energy gradients
    :return: list of nodes which represent the maximum gradient of graph energy
    """

    result = [
        max(energy_gradients[n], key=lambda key: energy_gradients[n][key])
        if (energy_gradients[n] and max(energy_gradients[n].values()) > 0) else None
        for n
        in energy_gradients.keys()
    ]

    return result


def graph_energy_gradient_centrality(g: object) -> Dict:
    """
    Computes the stationary distribution of the random walk directed by the gradient of graph energy

    :param g: input graph
    :return: list of centrality scores for each node
    """

    gs = get_graph_spectrum(g)

    gradients = {
        (u, v): gs[u] - gs[v]
        if (gs[u] - gs[v]) > 0 else 0.0
        for (u, v)
        in g.edges
    }

    nx.set_edge_attributes(g, gradients, 'gradients')
    result = nx.pagerank(g, weight='gradients')

    return result
