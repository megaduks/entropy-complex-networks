import gensim
import numpy as np
import multiprocessing
import networkx as nx

from typing import Dict, List
from networkentropy import network_energy as ne


# TODO: unify walk generation procedures by providing only the criterion for next node selection
# TODO: unify this file with network_utils.py implementation of node2vec

def generate_random_walk(graph: object, walk_length: int, beta: float = 0.15) -> List:
    """
    Finds a sequence of nodes forming a random walk in the graph

    :param graph: input graph
    :param walk_length: fixed length of each random walk
    :param beta: probability of making a random jump instead of continuing the walk

    :return: List of nodes forming a random walk
    """
    walk = list()

    current_node = np.random.choice(graph.nodes)

    # nodes are stored as strings instead of ints because of the gensim's word2vec implementation
    walk.append(str(current_node))

    for i in range(walk_length):
        if np.random.random() <= beta:
            next_node = np.random.choice(graph.nodes)
        else:
            neighborhood = list(nx.neighbors(graph, current_node))
            if neighborhood:
                next_node = np.random.choice(neighborhood)
            else:
                next_node = np.random.choice(graph.nodes)

        walk.append(str(next_node))
        current_node = next_node

    return walk


def generate_gradient_walk(graph: object,
                           walk_length: int,
                           beta: float = 0.15,
                           energy_type: str = 'graph',
                           energy_gradient_radius: int = 1) -> List:
    """
    Finds a sequence of nodes forming a walk in the graph where the criterion of the next node selection
    is the increasing energy gradient

    :param graph: input graph
    :param walk_length: fixed length of each gradient walk
    :param beta: probability of making a random jump instead of continuing the walk
    :param energy_type: name of node energy, possible values include 'graph', 'laplacian', 'randic'
    :param energy_gradient_radius: radius of ego network for which node energy is computed

    :return: List of nodes forming a gradient walk
    """
    walk = list()

    energy_distribution = {
        'graph': ne.get_graph_spectrum,
        'laplacian': ne.get_laplacian_spectrum,
        'randic': ne.get_randic_spectrum
    }

    energy_gradients = ne.get_graph_energy_gradients(graph,
                                                     energy_distribution[energy_type](graph, energy_gradient_radius))

    current_node = np.random.choice(graph.nodes)

    # nodes are stored as strings instead of ints because of the gensim's word2vec implementation
    walk.append(str(current_node))

    for i in range(walk_length):

        # get positive gradients from the current node
        current_gradients = [
            (k, v)
            for k, v in energy_gradients[current_node].items()
            if v > 0
        ]
        sum_current_gradients = sum([v for k, v in current_gradients])

        if np.random.random() <= beta:
            next_node = np.random.choice(graph.nodes)
        else:
            if sum_current_gradients > 0:
                neighbors = [k for (k, v) in current_gradients]
                probs = [v / sum_current_gradients for k, v in current_gradients]
                next_node = np.random.choice(neighbors, size=1, p=probs)[0]
            else:
                next_node = np.random.choice(graph.nodes)

        walk.append(str(next_node))
        current_node = next_node

    return walk


def node2vec(graph: object, walk_length: int = 10, walk_number: int = 1000, embedding_size: int = 100,
             walk_type: str = 'random', energy_type: str = 'graph') -> List:
    """
    Generates node embeddings based on random walks

    :param graph: input graph
    :param walk_length: length of the random walk
    :param walk_number: number of random walks
    :param embedding_size: size of resulting embeddings
    :param walk_type: criterion for next node selection, possible values include 'random' and 'gradient'
    :param energy_type: name of node energy to be used for gradient computation

    :return: list of node embeddings
    """

    walks = list()
    num_cpu = multiprocessing.cpu_count() - 1

    for i in range(walk_number):
        if walk_type == 'random':
            walks.append(generate_random_walk(graph=graph, walk_length=walk_length))
        elif walk_type == 'gradient':
            walks.append(generate_gradient_walk(graph=graph, walk_length=walk_length, energy_type=energy_type))

    # train a word2vec model on random walks as if they were sentences
    model = gensim.models.Word2Vec(walks, min_count=5, workers=num_cpu, size=embedding_size, batch_words=10)

    return model
