from abc import ABC
from itertools import starmap, islice, tee, count, repeat

import gensim
import numpy as np
import multiprocessing
import networkx as nx

from typing import List, Tuple, Callable, Dict
from node2vec import Node2Vec
from torch import nn as nn
from torch.nn import functional as F

import torch
import torch.autograd as autograd
import torch.optim as optim

# TODO: unify walk generation procedures by providing only the criterion for next node selection
# TODO: unify this file with utils.py implementation of node2vec

import networkentropy as ne
from networkentropy.utils import node_attribute_setter
from networkentropy.energy import get_dirichlet_energy, compute_degree_gradients


def generate_random_walk(graph: nx.Graph,
                         node: object,
                         walk_length: int,
                         beta: float = 0.15) -> List:
    """
    Finds a sequence of nodes forming a random walk in the graph

    :param graph: input graph
    :param node: a node from which the random walk starts
    :param walk_length: fixed length of each random walk
    :param beta: probability of making a random jump instead of continuing the walk

    :return: List of nodes forming a random walk
    """
    walk = [node]

    for i in range(walk_length):
        if np.random.random() <= beta:
            next_node = np.random.choice(graph.nodes)
        else:
            neighborhood = list(nx.neighbors(graph, node))
            if neighborhood:
                next_node = np.random.choice(neighborhood)
            else:
                next_node = np.random.choice(graph.nodes)

        walk.append(next_node)
        node = next_node

    return walk


# FIXME: change the generate_gradient_walk function to start from a given node (similar to above)

def generate_gradient_walk(graph: nx.Graph,
                           node: object,
                           walk_length: int,
                           beta: float = 0.15,
                           energy_type: str = 'graph',
                           energy_gradient_radius: int = 1) -> List:
    """
    Finds a sequence of nodes forming a walk in the graph where the criterion of the next node selection
    is the increasing energy gradient

    Args:
        graph: input graph
        node: a node from which the random walk starts
        walk_length: fixed length of each gradient walk
        beta: probability of making a random jump instead of continuing the walk
        energy_type: name of node energy, possible values include 'graph', 'laplacian', 'randic'
        energy_gradient_radius: radius of ego network for which node energy is computed

    :return: List of nodes forming a gradient walk

    """
    walk = [node]

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


def node2vec(graph: object,
             walk_length: int = 10,
             walk_number: int = 1000,
             embedding_size: int = 100,
             walk_type: str = 'random',
             energy_type: str = 'graph') -> List:
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


def embed_network(g: nx.Graph) -> np.ndarray:
    """Uses Node2Vec algorithm to embed network nodes in a continuous vector space"""

    node2vec = Node2Vec(g, dimensions=128, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=5)

    return model.wv.vectors


def extract_context_pairs(path: List, context_size: int = 2) -> List[Tuple]:
    """Transforms a list of nodes into a list of context node pairs"""

    def sliding(iterable, n):
        """Creates an iterator with sliding windows of size n over the iterable"""
        return zip(*starmap(islice, zip(tee(iterable, n), count(0), repeat(None))))

    context_pairs = []

    window_width = 2 * context_size + 1
    contexts = sliding(path, n=window_width)

    for context in contexts:
        for i in range(context_size):
            context_pairs.append((context[i], context[context_size]))
        for i in range(context_size + 1, window_width):
            context_pairs.append((context[context_size], context[i]))

    return context_pairs


class DirichletCrossEntropy:
    def __init__(self, graph: nx.Graph, criterion: Callable, alpha: float = 0.001):
        self.graph = graph
        self.criterion = criterion
        self.alpha = alpha

    def compute(self, output, target):
        log_prob = -1.0 * F.log_softmax(output, 1)
        loss = log_prob.gather(1, target.unsqueeze(1))
        loss = loss.mean()

        dirichlet_energy = get_dirichlet_energy(compute_degree_gradients(self.graph, 'uno', self.criterion))

        return loss + self.alpha * dirichlet_energy


class CrossEntropy:
    def __init__(self):
        pass

    def compute(self, output, target):
        log_prob = -1.0 * F.log_softmax(output, 1)
        loss = log_prob.gather(1, target.unsqueeze(1))
        loss = loss.mean()

        return loss


def random_walk_generator(graph: nx.Graph, walk_length: int, num_walks: int) -> List[Tuple]:
    """Generates context pairs of nodes based on random walks"""
    walks = list()

    for _ in range(num_walks):
        for node in graph.nodes:
            walks.append(generate_random_walk(graph=graph, node=node, walk_length=walk_length))

    context_tuple_list = []

    for walk in walks:
        context_tuple_list += extract_context_pairs(walk)

    return context_tuple_list


class UnoEmbedding(nn.Module, ABC):

    def __init__(self, embedding_size: int, network_size: int, walk_length: int, num_walks: int,
                 walk_generator: Callable, loss: object):
        super(UnoEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = network_size
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embeddings = nn.Embedding(network_size, embedding_size)
        self.linear = nn.Linear(embedding_size, network_size)
        self.walk_generator = walk_generator
        self.loss = loss

    def forward(self, context_word):
        emb = self.embeddings(context_word)
        hidden = self.linear(emb)
        out = F.log_softmax(hidden)
        return out

    def fit(self, graph: nx.Graph) -> List:
        """Trains a node2vec embedding using the list of pairs of context nodes"""

        @node_attribute_setter(name='uno')
        def get_uno_embeddings(graph: nx.Graph, mapping: Dict) -> Dict:
            return {n: self.embeddings.weight.data[mapping[n]].data.numpy() for n in graph.nodes}

        node_to_index = {n: idx for (idx, n) in enumerate(graph.nodes)}
        optimizer = optim.Adam(self.parameters())
        early_stopping = EarlyStopping(min_percent_gain=0.05)

        context_tensor_list = []

        for target, context in self.walk_generator(graph, walk_length=self.walk_length, num_walks=self.num_walks):
            target_tensor = autograd.Variable(torch.LongTensor([node_to_index[target]]))
            context_tensor = autograd.Variable(torch.LongTensor([node_to_index[context]]))
            context_tensor_list.append((target_tensor, context_tensor))

        epoch_loss = []

        while True:
            losses = []
            for target_tensor, context_tensor in context_tensor_list:
                self.zero_grad()
                log_probs = self(context_tensor)
                get_uno_embeddings(graph=graph, mapping=node_to_index)
                loss = self.loss.compute(log_probs, target_tensor)
                loss.backward()
                optimizer.step()
                losses.append(loss.data)

            print("Loss: ", np.mean(losses))
            epoch_loss.append(np.mean(losses))

            early_stopping.update_loss(np.mean(losses))
            if early_stopping.stop_training():
                break

        return epoch_loss


class EarlyStopping():
    """Implements an early stopping criterion"""

    def __init__(self, patience=10, min_percent_gain=0.1):
        self.patience = patience
        self.loss_list = []
        self.min_percent_gain = min_percent_gain / 100.

    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]

    def stop_training(self):
        if len(self.loss_list) == 1:
            return False
        gain = (max(self.loss_list) - min(self.loss_list)) / max(self.loss_list)

        if gain < self.min_percent_gain:
            return True
        elif self.loss_list[-1] > max(self.loss_list[:-1]):
            return True
        else:
            return False

