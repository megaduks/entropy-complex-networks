from typing import Dict, Callable, Tuple, Sequence
import networkx as nx

from networkentropy import network_energy as ne


def _get_energy_method(method: str) -> Callable[[nx.Graph, int], Dict]:
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


def _get_energy_method_name(method):
    return "{}_energy".format(method)


def _get_gradient_method_name(method):
    return "{}_gradient".format(method)


def _compute_gradient(energy1: float, energy2: float) -> float:
    return energy2 - energy1


def get_energy_gradients(g: nx.Graph, method: str, complete: bool = True, radius: int = 1) -> Dict[Tuple, float]:
    """
    Computes gradient between every two connected nodes.

    :param g: input graph
    :param method: name of a method for computing graph energy. Possible values are: randic, laplacian, graph
    :param complete: indicates if the result should contain every pair of connected nodes twice (in each order)
    :param radius: radius of the egocentric network
    :return: returns Dict with edges ad keys and gradients as values
    """
    get_energies = _get_energy_method(method)
    energies = get_energies(g, radius)
    result = {}
    for edge in g.edges:
        node1 = edge[0]
        node2 = edge[1]
        gradient = _compute_gradient(energies[node1], energies[node2])
        result[edge] = gradient
        if complete:
            result[edge[::-1]] = -gradient
    return result


class DecoratedGraph(nx.Graph):

    def __init__(self, supported_methods):
        self.supported_methods = supported_methods

    def get_gradient(self, node1, node2, method: str):
        if not (method in self.supported_methods):
            raise ValueError
        node1_energy = self.node[node1][_get_energy_method_name(method)]
        node2_energy = self.node[node2][_get_energy_method_name(method)]
        return _compute_gradient(node1_energy, node2_energy)

    def get_path_energy(self, path, method):
        energy_sum = 0
        for node in path:
            energy = self.node[node][_get_energy_method_name(method)]
            energy_sum += energy
        return energy_sum

    @classmethod
    def from_graph(cls, graph: nx.Graph, supported_methods: Sequence[str]) -> "DecoratedGraph":
        graph.__class__ = cls
        graph.supported_methods = supported_methods
        return graph


def get_graph_with_energy_data(g: nx.Graph, methods: Sequence[str], radius: int = 1, copy: bool = True) -> \
        DecoratedGraph:
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
        energy_methods[m] = _get_energy_method(m)
    if copy:
        g = g.copy()
    for method, get_energy in energy_methods.items():
        energies = get_energy(g, radius)
        for node, energy in energies.items():
            g.node[node][_get_energy_method_name(method)] = energy
        for edge in g.edges:
            node1 = edge[0]
            node2 = edge[1]
            energy_g1 = energies[node1]
            energy_g2 = energies[node2]
            g[node1][node2][_get_gradient_method_name(method)] = _compute_gradient(energy_g1, energy_g2)
    return DecoratedGraph.from_graph(g, methods)


def get_energy_gradient_centrality(g: nx.Graph, method: str, radius: int = 1, copy: bool = True):
    g_with_data = get_graph_with_energy_data(g, [method], radius, copy)
    return nx.pagerank(g_with_data, weight=_get_gradient_method_name(method))
