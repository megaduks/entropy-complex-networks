from typing import Dict, Callable, Tuple
import networkx as nx

from networkentropy import network_energy as ne


def get_energy_method(method: str) -> Callable[[nx.Graph, int], Dict]:
    """
    Returns one of methods for computing graph energy

    :param method: name of a method for computing graph energy. Possible values are: randic, laplacian, graph
    :return: method for computing graph energy
    """
    if method == "randic":
        return ne.randic_centrality
    elif method == "laplacian":
        return ne.laplacian_centrality
    elif method == "graph":
        return ne.graph_energy_centrality
    else:
        raise ValueError("Method: {} doesn't exist".format(method))


def __compute_gradient(energy1: float, energy2: float) -> float:
    return energy2 - energy1


def get_energy_gradients(g: nx.Graph, method: str, complete: bool = True, radius: int = 1) -> Dict[Tuple, float]:
    """
    Computes gradient between every two connected nodes.

    :param g: input graph
    :param method: name of a method for computing graph energy. Possible values are: randic, laplacian, graph
    :param complete: indicates if the result should contain every pair of connected nodes twice in two orders
    :param radius: radius of the egocentric network
    :return: returns Dict with edges ad keys and gradients as values
    """
    get_energies = get_energy_method(method)
    energies = get_energies(g, radius)
    result = {}
    for edge in g.edges:
        node1 = edge[0]
        node2 = edge[1]
        gradient = __compute_gradient(energies[node1], energies[node2])
        result[edge] = gradient
        if complete:
            result[edge[::-1]] = -gradient
    return result


def get_graph_with_energy_data(g: nx.Graph, methods: Tuple, radius: int = 1, copy: bool = True):
    """
    Computes energies and gradients and stores them in a graph as node attributes and edge attributes.
    Energies are stored in node attributes. The format of attribute names is: <METHOD>_energy
    Gradients are stored in edge attributes. The format of attribute names is: <METHOD>_gradient

    :param g: input graph
    :param methods: list of names of methods for computing graph energy. Possible values are: randic, laplacian, graph
    :param radius: radius of the egocentric network
    :param copy: if True the input graph in copied, if False the input graph is modified
    :return: Graph with energies and gradients stored as node and edge attributes
    """
    energy_methods = {}
    for m in methods:
        energy_methods[m] = get_energy_method(m)
    if copy:
        g = g.copy()
    for method, get_energy in energy_methods.items():
        energies = get_energy(g, radius)
        for node, energy in energies.items():
            g.node[node]["{}_energy".format(method)] = energy
        for edge in g.edges:
            node1 = edge[0]
            node2 = edge[1]
            energy_g1 = energies[node1]
            energy_g2 = energies[node2]
            g[node1][node2]["{}_gradient".format(method)] = __compute_gradient(energy_g1, energy_g2)
    return g
