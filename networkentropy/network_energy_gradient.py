import networkx as nx

from networkentropy import network_energy as ne


def get_energy_method(method: str):
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
    return get_energy(G2) - get_energy(G1)
