import numpy as np
import networkx as nx
import scipy
from itertools import product

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

    randic_values = [ 1/np.sqrt(D[v]*D[w]) if G.has_edge(v,w) else 0 for (v,w) in product(G.nodes, G.nodes) ]
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

    Randić energy is the sum of all absolute eigenvalues of the Randić matriź of a graph

    :param G: input graph
    :return: float: Randić energy of a graph
    """

    rm = get_randic_matrix(G)
    randic_energy = np.abs(scipy.linalg.eigvals(rm).real).sum()

    return randic_energy