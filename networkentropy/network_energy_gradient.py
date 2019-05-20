import operator
from typing import Dict, Callable, Tuple, Iterable, Optional
from types import MethodType
from networkx.algorithms.community.centrality import girvan_newman
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
        raise ValueError(f"Method: {method} doesn't exist")


def _get_energy_method_name(method):
    return f"{method}_energy"


def _get_gradient_method_name(method):
    return f"{method}_gradient"


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
        if complete and not g.is_directed():
            result[edge[::-1]] = -gradient
    return result


def _decorate_graph(graph: nx.Graph, supported_methods: Iterable[str]):
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

    graph.supported_methods = supported_methods
    graph.get_gradient = MethodType(get_gradient, graph)
    graph.get_path_energy = MethodType(get_path_energy, graph)
    return graph


def _get_supported_methods(graph: nx.Graph):
    if hasattr(graph, 'supported_methods'):
        return graph.supported_methods
    else:
        return []


def get_graph_with_energy_data(g: nx.Graph, methods: Iterable[str], radius: int = 1, copy: bool = True) -> \
        nx.Graph:
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
    methods = set(methods).difference(_get_supported_methods(g))
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
    return _decorate_graph(g, methods)


def get_energy_gradient_centrality(g: nx.Graph, method: str, radius: int = 1, alpha=0.85, personalization=None,
                                   max_iter=100, tol=1.0e-6, nstart=None, dangling=None,
                                   copy: bool = True) -> Optional[dict]:
    g_with_data = get_graph_with_energy_data(g, [method], radius, copy)
    try:
        result = nx.pagerank(g_with_data,
                             weight=_get_gradient_method_name(method),
                             alpha=alpha,
                             personalization=personalization,
                             max_iter=max_iter,
                             tol=tol,
                             nstart=nstart,
                             dangling=dangling)
    except nx.PowerIterationFailedConvergence:
        result = None
    return result


def _get_centrality_name(method):
    return f'{method}_gradient_centrality'


def get_graph_with_energy_gradient_centrality(g: nx.Graph, methods: Iterable[str], radius: int = 1, alpha=0.85,
                                              personalization=None, max_iter=100, tol=1.0e-6, nstart=None,
                                              dangling=None, copy: bool = True):
    if copy:
        g = g.copy()
    for method in methods:
        pagerank = get_energy_gradient_centrality(g,
                                                  method=method,
                                                  radius=radius,
                                                  alpha=alpha,
                                                  personalization=personalization,
                                                  max_iter=max_iter,
                                                  tol=tol,
                                                  nstart=nstart,
                                                  dangling=dangling,
                                                  copy=False)
        if pagerank is None:
            return None
        for node, pr in pagerank.items():
            g.node[node][_get_centrality_name(method)] = pr
    return g


def girvan_newman_energy_gradient(graph: nx.Graph, method: str):
    def most_central_edge(g):
        gradients = get_energy_gradients(g, method, complete=False)
        return max(gradients.items(), operator.itemgetter(1))[0]
    return girvan_newman(graph, most_valuable_edge=most_central_edge)
