from abc import ABC
from typing import List, Tuple
from itertools import *

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

import networkx as nx
import numpy as np


def generate_random_walk(g: nx.Graph, walk_length: int, beta: float = 0.15):
    """
    Finds a sequence of nodes forming a random walk in the graph

    Args:
        g: input graph
        walk_length: fixed length of each random walk
        beta: probability of making a random jump instead of continuing the walk

    Returns:
        List of nodes forming a random walk
    """
    walk = []

    current_node = np.random.choice(g.nodes)
    walk.append(current_node)

    for i in range(walk_length):
        if np.random.random() <= beta:
            next_node = np.random.choice(g.nodes)
        else:
            next_node = np.random.choice(list(nx.neighbors(g, current_node)))

        walk.append(str(next_node))
        current_node = next_node

    return walk


def sliding(iterable, n):
    """Creates an iterator with sliding windows of size n over the iterable"""
    return zip(*starmap(islice, zip(tee(iterable, n), count(0), repeat(None))))


def path_to_bon(path: List[int], context_size: int = 2) -> List[Tuple[int, int]]:
    """Transforms a list of nodes into a list of tuples with context node pairs"""

    context_pairs = []

    window_width = 2 * context_size + 1
    contexts = sliding(path, n=window_width)

    for context in contexts:
        for i in range(context_size):
            context_pairs.append((context[i], context[context_size]))
        for i in range(context_size + 1, window_width):
            context_pairs.append((context[context_size], context[i]))

    return context_pairs


class Node2Vec(nn.Module, ABC):

    def __init__(self, embedding_size, vocab_size):
        super(Node2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, context_word):
        emb = self.embeddings(context_word)
        hidden = self.linear(emb)
        out = F.log_softmax(hidden)
        return out


def cross_entropy_loss(output, target):
    log_prob = -1.0 * F.log_softmax(output, 1)
    loss = log_prob.gather(1, target.unsqueeze(1))
    loss = loss.mean()
    return loss


class EarlyStopping():
    def __init__(self, patience=5, min_percent_gain=0.1):
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
        # print(f"Loss gain: {round(100 * gain, 2)}%")
        if gain < self.min_percent_gain:
            return True
        else:
            return False


if __name__ == "__main__":

    g = nx.florentine_families_graph()
    walks = list()

    for i in range(100):
        walks.append(generate_random_walk(g, walk_length=5))

    vocabulary_size = len(g.nodes)

    context_tuple_list = []
    for walk in walks:
        context_tuple_list += path_to_bon(walk)

    node_to_index = {n: idx for (idx, n) in enumerate(g.nodes)}
    index_to_node = {idx: n for (idx, n) in enumerate(g.nodes)}

    net = Node2Vec(embedding_size=2, vocab_size=vocabulary_size)
    # loss_function = nn.CrossEntropyLoss()
    loss_function = cross_entropy_loss
    optimizer = optim.Adam(net.parameters())
    early_stopping = EarlyStopping()
    context_tensor_list = []

    for target, context in context_tuple_list:
        target_tensor = autograd.Variable(torch.LongTensor([node_to_index[target]]))
        context_tensor = autograd.Variable(torch.LongTensor([node_to_index[context]]))
        context_tensor_list.append((target_tensor, context_tensor))

    while True:
        losses = []
        for target_tensor, context_tensor in context_tensor_list:
            net.zero_grad()
            log_probs = net(context_tensor)
            loss = loss_function(log_probs, target_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
        print("Loss: ", np.mean(losses))
        early_stopping.update_loss(np.mean(losses))
        if early_stopping.stop_training():
            break

    for n in g.nodes:
        print(f"{n}: {net.embeddings.weight.data[node_to_index[n]]}")
