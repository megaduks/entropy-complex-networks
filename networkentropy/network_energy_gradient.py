from typing import Dict, Callable, Tuple
import networkx as nx

from networkentropy import network_energy as ne


def get_energy_method(method: str) -> Callable[[nx.Graph], float]:
    """
    Returns one of methods for computing graph energy

    :param method: name of a method for computing graph energy. Possible values are: randic, laplacian, graph
    :return: method for computing graph energy
    """
    if method == "randic":
        return ne.get_randic_energy
    elif method == "laplacian":
        return ne.get_laplacian_energy
    elif method == "graph":
        return ne.get_graph_energy
    else:
        raise ValueError("Method: {} doesn't exist".format(method))


def __compute_gradient(energy1: float, energy2: float) -> float:
    return energy2 - energy1


def get_energy_gradient(G1: nx.Graph, G2: nx.Graph, method: str) -> float:
    """
    Computes energy gradient between two graphs using a given method. Energy of G2 graph is subtracted from
    energy of G1 graph.

    :param G1: first graph
    :param G2: second graph
    :param method: name of a method for computing graph energy. Possible values are: randic, laplacian, graph
    :return: energy gradient between the two graphs
    """
    get_energy = get_energy_method(method)
    return __compute_gradient(get_energy(G1), get_energy(G2))


def get_energy_gradient_of_edge(G: nx.Graph, edge: Tuple, method: str, radius: int = 1):
    G1 = nx.ego_graph(G, edge[0], radius=radius)
    G2 = nx.ego_graph(G, edge[1], radius=radius)
    return get_energy_gradient(G1, G2, method)


def get_energy_gradients(G: nx.Graph, method: str, complete: bool = True, radius: int = 1) -> Dict[Tuple, float]:
    result = {}
    for edge in G.edges:
        gradient = get_energy_gradient_of_edge(G, edge, method, radius)
        result[edge] = gradient
        if complete:
            result[edge[::-1]] = -gradient
    return result


def get_graph_with_energy_data(G: nx.Graph, methods: Tuple, radius: int = 1, copy: bool = True):
    energy_functions = {}
    for m in methods:
        energy_functions[m] = get_energy_method(m)
    if copy:
        G = G.copy()
    for edge in G.edges:
        node1 = edge[0]
        node2 = edge[1]
        G1 = nx.ego_graph(G, node1, radius=radius)
        G2 = nx.ego_graph(G, node2, radius=radius)
        for method, get_energy in energy_functions.items():
            energyG1 = get_energy(G1)
            energyG2 = get_energy(G2)
            G.node[node1]["{}_energy".format(method)] = energyG1
            G.node[node2]["{}_energy".format(method)] = energyG2
            G[node1][node2]["{}_gradient".format(method)] = __compute_gradient(energyG1, energyG2)
    return G
