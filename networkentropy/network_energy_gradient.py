import operator
from typing import Dict, Callable, Tuple, Iterable, Optional
from types import MethodType, MappingProxyType
from networkx.algorithms.community.centrality import girvan_newman
from functools import lru_cache
import networkx as nx

from networkentropy import network_energy as ne

_EDGES_DECORATORS_ATTR_NAME = 'edges_decorators'

_NODES_DECORATORS_ATTR_NAME = 'nodes_decorators'

_EMPTY_DICT = MappingProxyType({})

decorating_function = lru_cache(maxsize=10)

_ENERGY_METHODS_CACHING = {
    'randic': decorating_function(ne.randic_centrality),
    'laplacian': decorating_function(ne.laplacian_centrality),
    'graph': decorating_function(ne.graph_energy_centrality)
}


def _get_energy_method(method: str) -> Callable[[nx.Graph, int], Dict]:
    """
    Returns one of methods for computing graph energy

    :param method: name of a method for computing graph energy. Possible values are: randic, laplacian, graph
    :return: method for computing graph energy
    """
    if method not in _ENERGY_METHODS_CACHING:
        raise ValueError(f"Method: {method} doesn't exist")
    else:
        return _ENERGY_METHODS_CACHING[method]


def _get_energy_method_name(method):
    return f"{method}_energy"


def _get_gradient_method_name(method):
    return f"{method}_gradient"


def _compute_gradient(energy1: float, energy2: float) -> float:
    return energy2 - energy1


def get_energy_gradients(g: nx.Graph, method: str, complete: bool = False, radius: int = 1) -> Dict[Tuple, float]:
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


def _is_decorated_helper(graph: nx.Graph, attribute_name: str, decorator_name: str) -> bool:
    return decorator_name in getattr(graph, attribute_name, [])


def has_nodes_decorated(graph: nx.Graph, decorator_name: str) -> bool:
    return _is_decorated_helper(graph, _NODES_DECORATORS_ATTR_NAME, decorator_name)


def has_edges_decorated(graph: nx.Graph, decorator_name: str) -> bool:
    return _is_decorated_helper(graph, _EDGES_DECORATORS_ATTR_NAME, decorator_name)


def _add_decorator_helper(graph: nx.Graph, attribute_name: str, decorator_name: str) -> list:
    storing_attribute = getattr(graph, attribute_name, [])
    storing_attribute.append(decorator_name)
    setattr(graph, attribute_name, storing_attribute)
    return storing_attribute


def add_nodes_decorator(graph: nx.Graph, decorator_name: str) -> list:
    return _add_decorator_helper(graph, _NODES_DECORATORS_ATTR_NAME, decorator_name)


def add_edges_decorator(graph: nx.Graph, decorator_name: str) -> list:
    return _add_decorator_helper(graph, _EDGES_DECORATORS_ATTR_NAME, decorator_name)


def clear_all_nodes_attrs(g: nx.Graph):
    for n, data in g.nodes(data=True):
        data.clear()
    if hasattr(g, _NODES_DECORATORS_ATTR_NAME):
        delattr(g, _NODES_DECORATORS_ATTR_NAME)


def clear_all_edges_attrs(g: nx.Graph):
    for n1, n2, data in g.edges(data=True):
        data.clear()
    if hasattr(g, _EDGES_DECORATORS_ATTR_NAME):
        delattr(g, _EDGES_DECORATORS_ATTR_NAME)


def decorate_graph(graph: nx.Graph,
                   nodes_decorators: dict = _EMPTY_DICT,
                   edges_decorators: dict = _EMPTY_DICT,
                   methods: dict = _EMPTY_DICT,
                   copy: bool = False,
                   clear: bool = False):
    if copy:
        graph = graph.copy()
    if clear:
        clear_all_nodes_attrs(graph)
        clear_all_edges_attrs(graph)
    for name, function in nodes_decorators.items():
        if not has_nodes_decorated(graph, name):
            result = function(graph)
            if result:
                for node, value in result.items():
                    graph.nodes[node][name] = value
                add_nodes_decorator(graph, name)
    for name, function in edges_decorators.items():
        if not has_edges_decorated(graph, name):
            result = function(graph)
            if result:
                for edge, value in function(graph).items():
                    graph[edge[0]][edge[1]][name] = value
                add_edges_decorator(graph, name)
    for name, method in methods.items():
        setattr(graph, name, MethodType(method, graph))
    return graph


def _get_gradient(graph, node1, node2, method: str):
    if not (method in graph.supported_methods):
        raise ValueError
    node1_energy = graph.nodes[node1][_get_energy_method_name(method)]
    node2_energy = graph.nodes[node2][_get_energy_method_name(method)]
    return _compute_gradient(node1_energy, node2_energy)


def _get_path_energy(graph, path, method):
    energy_sum = 0
    for node in path:
        energy = graph.nodes[node][_get_energy_method_name(method)]
        energy_sum += energy
    return energy_sum


def get_graph_with_energy_data(g: nx.Graph, methods: Iterable[str], radius: int = 1, copy: bool = False,
                               clear: bool = False) -> \
        nx.Graph:
    """
    Computes energies and gradients and stores them in a graph as node attributes and edge attributes.
    Energies are stored in node attributes. The format of attribute names is: <METHOD>_energy
    Gradients are stored in edge attributes. The format of attribute names is: <METHOD>_gradient

    :param clear: if True, all graph attributes are first deleted
    :param g: input graph
    :param methods: list of names of methods for computing graph energy. Possible values are: randic, laplacian, graph
    :param radius: radius of the egocentric network
    :param copy: if True the input graph in copied, if False the input graph is modified
    :return: Graph with energies and gradients stored as node and edge attributes
    """
    energy_methods = {}
    for m in methods:
        energy_methods[m] = _get_energy_method(m)
    nodes_decorators = {}
    edges_decorators = {}
    for method, get_energy in energy_methods.items():
        nodes_decorators[_get_energy_method_name(method)] = lambda graph: get_energy(graph, radius=radius)
        edges_decorators[_get_gradient_method_name(method)] = lambda graph: get_energy_gradients(graph, method,
                                                                                                 radius=radius)
    return decorate_graph(g,
                          nodes_decorators=nodes_decorators,
                          edges_decorators=edges_decorators,
                          methods={
                              'get_gradient': _get_gradient,
                              'get_path_energy': _get_path_energy
                          },
                          copy=copy,
                          clear=clear)


@lru_cache(maxsize=10)
def get_energy_gradient_centrality(g: nx.Graph, method: str, radius: int = 1, alpha=0.85, personalization=None,
                                   max_iter=100, tol=1.0e-6, nstart=None, dangling=None,
                                   copy: bool = True, clear=True) -> Optional[dict]:
    g_with_data = get_graph_with_energy_data(g, [method], radius, copy, clear=clear)
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
                                              dangling=None, copy: bool = False, clear: bool = False):
    nodes_decorators = {}
    for method in methods:
        name = _get_centrality_name(method)
        nodes_decorators[name] = lambda graph: get_energy_gradient_centrality(graph,
                                                                              method=method,
                                                                              radius=radius,
                                                                              alpha=alpha,
                                                                              personalization=personalization,
                                                                              max_iter=max_iter,
                                                                              tol=tol,
                                                                              nstart=nstart,
                                                                              dangling=dangling,
                                                                              copy=False)
    return decorate_graph(g, nodes_decorators=nodes_decorators, copy=copy, clear=clear)


def girvan_newman_energy_gradient(graph: nx.Graph, method: str):
    def most_central_edge(g):
        gradients = get_energy_gradients(g, method, complete=False)
        return max(gradients.items(), operator.itemgetter(1))[0]
    return girvan_newman(graph, most_valuable_edge=most_central_edge)


def clear_cache():
    get_energy_gradient_centrality.cache_clear()
    for function in _ENERGY_METHODS_CACHING.values():
        function.cache_clear()
