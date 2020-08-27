import operator

import numpy as np
import scipy
import scipy.stats
import torch

from typing import Dict, List, Iterable, Callable, Tuple, Optional
from itertools import product
from types import MethodType, MappingProxyType
from functools import lru_cache
import networkx as nx
from networkx.algorithms.community import girvan_newman

from node2vec import Node2Vec
from networkentropy.utils import node_attribute_setter

_EDGES_DECORATORS_ATTR_NAME = 'edges_decorators'

_NODES_DECORATORS_ATTR_NAME = 'nodes_decorators'

_EMPTY_DICT = MappingProxyType({})

decorating_function = lru_cache(maxsize=10)


def get_randic_matrix(g: nx.Graph) -> np.array:
    """
    Computes the Randić matrix of a graph

    Elements of the Randić matrix are defined as follows:

               0 if v=w
    R(v,w) =   1/sqrt(D(v)*D(w)) if (v,w) in E(g)
               0 otherwise

    :param g: input graph
    :return: NumPy array
    """

    D = nx.degree(g)

    randic_values = [1 / np.sqrt(D[v] * D[w]) if g.has_edge(v, w) else 0 for (v, w) in product(g.nodes, g.nodes)]
    randic_matrix = np.asarray(randic_values).reshape(g.number_of_nodes(), g.number_of_nodes())

    return randic_matrix


def get_randic_index(g: nx.Graph) -> float:
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


def get_randic_energy(g: nx.Graph) -> float:
    """
    Computes the Randić energy of a graph

    Randić energy is the sum of all absolute eigenvalues of the Randić matrix of a graph

    :param g: input graph
    :return: float: Randić energy of a graph
    """

    R = get_randic_matrix(g)
    randic_energy = np.abs(scipy.linalg.eigvals(R).real).sum()

    return randic_energy


def get_randic_spectrum(g: nx.Graph, radius: int = 1) -> np.array:
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


@node_attribute_setter(name='randic')
def randic_centrality(graph: nx.Graph, radius: int = 1, normalized: bool = False) -> Dict:
    """
    Computes the centrality index for each vertex by computing the Randić energy of that vertex's
    neighborhood of a given radius

    :param graph: input graph
    :param radius: radius of the egocentric network
    :param normalized: if True, the result is normalized to sum to 1
    :return: dictionary with Randić energy centrality for each vertex
    """

    result = {n: get_randic_energy(nx.ego_graph(G=graph, n=n, radius=radius)) for n in graph.nodes}

    if normalized:
        s = sum(result.values())
        result = {n: v / s for n, v in result.items()}

    return result


def get_laplacian_energy(g: nx.Graph) -> float:
    """
    Computes the energy of the Laplacian of a graph

    :param g: input graph
    :return: float: Laplacian energy of a graph
    """

    if nx.is_directed(g):
        g = nx.to_undirected(g)

    L = nx.laplacian_matrix(g).todense()
    eigvals = scipy.linalg.eigvals(L).real
    const = nx.number_of_edges(g) * 2 / nx.number_of_nodes(g)
    consts = np.full(nx.number_of_nodes(g), const)
    laplacian_energy = np.abs(np.subtract(eigvals, consts)).sum()

    return laplacian_energy


def get_random_walk_laplacian_matrix(g: object) -> np.ndarray:
    """
    Extracts random walk Laplacian matrix of a graph

    Args:
        g: input graph

    Returns: matrix representing random walk Laplacian
    """
    A = nx.adjacency_matrix(g).todense()
    d = np.array([-(1 / v) if v > 0 else 0 for k, v in nx.degree(g)]).reshape((-1, 1))
    D = np.diag(np.array([1 if v > 0 else 0 for k, v in nx.degree(g)]))

    return np.multiply(d, A) + D


def get_symmetric_normalized_laplacian_matrix(g: object) -> np.ndarray:
    """
    Extracts symmetric normalized Laplacian matrix of a graph

    Args:
        g: input graph

    Returns: matrix representing the symmetric normalized Laplacian of a graph
    """
    L = nx.laplacian_matrix(g).todense()
    d = np.array([1 / np.sqrt(v) if v > 0 else 0 for k, v in nx.degree(g)])

    return np.multiply(np.diag(d), L, np.diag(d))


def get_laplacian_spectrum(g: object, radius: int = 1) -> np.array:
    """
    Computes the spectrum of the Laplacian energy of a graph

    :rtype: object
    :param g: input graph
    :param radius: size of the egocentric network
    :return: NumPy array
    """

    result = [
        get_laplacian_energy(nx.ego_graph(G=g, n=n, radius=radius))
        for n in g.nodes
    ]

    return np.asarray(result)


@node_attribute_setter(name='laplacian')
def laplacian_centrality(graph: nx.Graph, radius: int = 1, normalized: bool = False) -> Dict:
    """
    Computes the centrality index for each vertex by computing the Laplacian energy of that vertex's
    neighborhood of a given radius

    :param graph: input graph
    :param radius: radius of the egocentric network
    :param normalized: if True, the result is normalized to sum to 1
    :return: dictionary with Laplacian energy centrality for each vertex
    """

    result = {n: get_laplacian_energy(nx.ego_graph(G=graph, n=n, radius=radius)) for n in graph.nodes}

    if normalized:
        s = sum(result.values())
        result = {n: v / s for n, v in result.items()}

    return result


def get_graph_energy(g: nx.Graph) -> float:
    """
    Computes the energy of the adjacency matrix of a graph

    :param g: input graph
    :return: float: energy of the adjacency matrix of a graph
    """

    M = nx.adjacency_matrix(g).todense()
    graph_energy = np.abs(scipy.linalg.eigvals(M).real).sum()

    return graph_energy


def get_graph_spectrum(g: nx.Graph, radius: int = 1) -> np.array:
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


@node_attribute_setter(name='energy')
def graph_energy_centrality(graph: nx.Graph, radius: int = 1, normalized: bool = False) -> Dict:
    """
    Computes the centrality index for each vertex by computing the graph energy of that vertex's
    neighborhood of a given radius

    :param graph: input graph
    :param radius: radius of the egocentric network
    :param normalized: if True, the result is normalized to sum to 1
    :return: dictionary with graph energy centrality for each vertex
    """

    result = {n: get_graph_energy(nx.ego_graph(G=graph, n=n, radius=radius)) for n in graph.nodes}

    if normalized:
        s = sum(result.values())
        result = {n: v / s for n, v in result.items()}

    return result


def graph_energy_pagerank(g: nx.Graph,
                          radius: int = 1,
                          normalized: bool = False,
                          mode: str = 'graph',
                          alpha: float = 0.85,
                          max_iter: int = 10000,
                          tol: float = 1.0e-6
                          ) -> Dict:
    """
    Computes the eigenvector centrality index for each vertex by computing the pagerank of each vertex
    using vertex energy as the value of the personalization parameter

    :param g: input graph
    :param radius: radius of the egocentric network
    :param normalized: if True, the result is normalized to sum to 1
    :param mode: string representing type of graph energy, possible values include 'graph', 'randic', 'laplacian'
    :param alpha: probability of continuing the random walk
    :param max_iter: maximum number of iterations for the random walk computation
    :param tol: tolerance parameter for random walk divergence

    :return: dictionary with eigenvalue centralities for each vertex
    """

    assert mode in ['graph', 'randic', 'laplacian'], "supported modes are: 'graph', 'randic', 'laplacian'"

    if mode == 'graph':
        energy_dist = graph_energy_centrality(g, radius=radius)
    elif mode == 'randic':
        energy_dist = randic_centrality(g, radius=radius)
    elif mode == 'laplacian':
        energy_dist = laplacian_centrality(g, radius=radius)

    if sum(energy_dist.values()) == 0:
        return energy_dist

    result = nx.pagerank(g, personalization=energy_dist, alpha=alpha, max_iter=max_iter, tol=tol)

    if normalized:
        s = sum(result.values())
        result = {n: v / s for n, v in result.items()}

    return result


def get_energy_gradients(g: nx.Graph, energy_dist: List[float] = None, mode: str = 'graph') -> Dict:
    """
    Compute gradients of a given graph energy for all nodes

    :param g: input graph
    :param energy_dist: precomputed distribution of energy in the graph g
    :param mode: string representing type of graph energy, possible values include 'graph', 'randic', 'laplacian'

    :return: dictionary with energy differences for each node
    """

    if energy_dist is None:
        if mode == 'graph':
            energy_dist = get_graph_spectrum(g)
        elif mode == 'randic':
            energy_dist = get_randic_spectrum(g)
        elif mode == 'laplacian':
            energy_dist = get_laplacian_spectrum(g)

    result = {
        n: {
            nn: energy_dist[nn] - energy_dist[n]
            for nn
            in nx.ego_graph(G=g, n=n)
            if nn != n
        }
        for n
        in g.nodes
    }

    return result


def get_max_energy_gradient(energy_gradients: Dict) -> List[int]:
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


def gradient_centrality(g: nx.Graph,
                        normalized: bool = False,
                        radius: int = 1,
                        mode: str = 'graph',
                        alpha: float = 0.85,
                        max_iter: int = 10000,
                        tol: float = 1.0e-6
                        ) -> Dict:
    """
    Computes the stationary distribution of the random walk directed by the gradient of graph energy

    :param g: input graph
    :param normalized: if True, the result is normalized to sum to 1
    :param radius: radius of the egocentric network
    :param mode: string representing type of graph energy, possible values include 'graph', 'randic', 'laplacian'
    :param alpha: probability of continuing the random walk
    :param max_iter: maximum number of iterations for the random walk computation
    :param tol: tolerance parameter for random walk divergence

    :return: list of centrality scores for each node
    """

    assert mode in ['graph', 'randic', 'laplacian'], "supported modes are: 'graph', 'randic', 'laplacian'"

    delta = 1.0e-4  # small value to be used instead of negative gradients

    if mode == 'graph':
        energy = graph_energy_centrality(g, radius=radius)
    elif mode == 'randic':
        energy = randic_centrality(g, radius=radius)
    elif mode == 'laplacian':
        energy = laplacian_centrality(g, radius=radius)

    gradients = {
        (u, v): energy[u] - energy[v]
        if (energy[u] - energy[v]) > 0 else 0.0
        for (u, v)
        in g.edges
    }

    nx.set_edge_attributes(g, gradients, 'gradients')
    result = nx.pagerank(g, weight='gradients')

    return result


def get_dirichlet_energy(gradients: List[torch.Tensor]) -> float:
    """
    Computes the Dirichlet energy of the gradient vector field

    Args:
        gradients: array-like list of gradients at each graph vertex

    Returns: positive float

    """

    return 0.5 * sum([torch.norm(gradient) ** 2 for gradient in gradients])


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


def compute_degree_gradients(g: nx.Graph, embedding_name: str, criterion: Callable) -> np.array:
    """Computes the gradients of degree for the current node embedding"""
    gradients = []
    scalars = criterion(g)
    embeddings = nx.get_node_attributes(g, embedding_name)

    for n in g.nodes:
        neighbours = {k: v for k, v in scalars.items() if k in nx.neighbors(g, n)}
        max_neighbor = max(neighbours.items(), key=operator.itemgetter(1))[0]

        if scalars[max_neighbor] > scalars[n]:
            n_emb = torch.Tensor(embeddings[n])
            m_emb = torch.Tensor(embeddings[max_neighbor])
            gradients.append(m_emb - n_emb)
        else:
            gradients.append(torch.zeros(embeddings[n].shape[0]))

    return gradients


_ENERGY_METHODS_CACHING = {
    'randic': decorating_function(randic_centrality),
    'laplacian': decorating_function(laplacian_centrality),
    'graph': decorating_function(graph_energy_centrality)
}
