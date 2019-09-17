import numpy as np
import math
from . import helpers


# overlap
def overlap_single(x_data, a, b):
    similarity = 0.0
    for xa, xb in zip(x_data[a], x_data[b]):
        if xa == xb:
            similarity += 1
    return similarity / x_data.shape[1]


# overlap
def overlap(x_data):
    similarities = np.ones((x_data.shape[0], x_data.shape[0]))
    for i in range(x_data.shape[0]):
        for j in range(i + 1, x_data.shape[0]):
            similarities[i][j] = overlap_single(x_data, i, j)
            similarities[j][i] = similarities[i][j]
    return similarities


# occurrences frequency
def of_single(x_data, a, b, uniques):
    similarity = 0.0
    for xa, xb, unique in zip(x_data[a], x_data[b], uniques):
        if xa == xb:
            similarity += 1
        else:
            N = x_data.shape[0]
            m = 1 / (1 + math.log(N / unique[xa]) * math.log(N / unique[xb]))
            similarity += m
    return similarity / x_data.shape[1]


# occurrences frequency
def of(x_data):
    similarities = np.ones((x_data.shape[0], x_data.shape[0]))
    uniques = helpers.count_uniques_per_attribute(x_data)
    for i in range(x_data.shape[0]):
        for j in range(i + 1, x_data.shape[0]):
            similarities[i][j] = of_single(x_data, i, j, uniques)
            similarities[j][i] = similarities[i][j]
    return similarities


# inverted occurrences frequency
def iof_single(x_data, a, b, uniques):
    similarity = 0.0
    for xa, xb, unique in zip(x_data[a], x_data[b], uniques):
        if xa == xb:
            similarity += 1
        else:
            m = 1 / (1 + math.log(unique[xa]) * math.log(unique[xb]))
            similarity += m
    return similarity / x_data.shape[1]


# inverted occurrences frequency
def iof(x_data):
    similarities = np.ones((x_data.shape[0], x_data.shape[0]))
    uniques = helpers.count_uniques_per_attribute(x_data)
    for i in range(x_data.shape[0]):
        for j in range(i + 1, x_data.shape[0]):
            similarities[i][j] = iof_single(x_data, i, j, uniques)
            similarities[j][i] = similarities[i][j]
    return similarities


# goodall 3
def goodall_3_single(x_data, a, b, uniques):
    similarity = 0.0
    for xa, xb, unique in zip(x_data[a], x_data[b], uniques):
        if xa == xb:
            N = x_data.shape[0]
            similarity += 1 - (unique[xa] * (unique[xa] - 1) / (N * (N - 1)))
    return similarity / x_data.shape[1]


# goodall 3
def goodall_3(x_data):
    similarities = np.ones((x_data.shape[0], x_data.shape[0]))
    uniques = helpers.count_uniques_per_attribute(x_data)
    for i in range(x_data.shape[0]):
        for j in range(i + 1, x_data.shape[0]):
            similarities[i][j] = goodall_3_single(x_data, i, j, uniques)
            similarities[j][i] = similarities[i][j]
    return similarities

