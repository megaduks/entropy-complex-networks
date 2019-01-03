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


def get_energy_gradients(G: nx.Graph, method: str, complete: bool = True, radius: int = 1) -> Dict[Tuple, float]:
    get_energies = get_energy_method(method)
    energies = get_energies(G, radius)
    result = {}
    for edge in G.edges:
        node1 = edge[0]
        node2 = edge[1]
        gradient = __compute_gradient(energies[node1], energies[node2])
        result[edge] = gradient
        if complete:
            result[edge[::-1]] = -gradient
    return result


def get_graph_with_energy_data(G: nx.Graph, methods: Tuple, radius: int = 1, copy: bool = True):
    energy_methods = {}
    for m in methods:
        energy_methods[m] = get_energy_method(m)
    if copy:
        G = G.copy()
    for method, get_energy in energy_methods.items():
        energies = get_energy(G, radius)
        for edge in G.edges:
            node1 = edge[0]
            node2 = edge[1]
            energyG1 = energies[node1]
            energyG2 = energies[node2]
            G.node[node1]["{}_energy".format(method)] = energyG1
            G.node[node2]["{}_energy".format(method)] = energyG2
            G[node1][node2]["{}_gradient".format(method)] = __compute_gradient(energyG1, energyG2)
    return G
