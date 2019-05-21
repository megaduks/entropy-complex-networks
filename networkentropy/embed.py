import gensim
import numpy as np
import multiprocessing
import networkx as nx

from typing import Dict, List


def generate_random_walk(graph: object, walk_length: int, beta:float = 0.15) -> List:
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
            next_node = np.random.choice(list(nx.neighbors(graph, current_node)))

        walk.append(str(next_node))
        current_node = next_node

    return walk


def node2vec(graph: object, walk_length: int = 10, walk_number: int = 10000, embedding_size: int = 300) -> List:
    """
    Generates node embeddings based on random walks

    :param graph: input graph
    :param walk_length: length of the random walk
    :param walk_number: number of random walks
    :param embedding_size: size of resulting embeddings

    :return: list of node embeddings
    """

    walks = list()
    num_cpu = multiprocessing.cpu_count() - 1

    for i in range(walk_number):
        walks.append(generate_random_walk(graph, walk_length=walk_length))

    # train a word2vec model on random walks as if they were sentences
    model = gensim.models.Word2Vec(walks, min_count=1, workers=num_cpu, size=embedding_size)

    return model