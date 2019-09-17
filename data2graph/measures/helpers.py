import numpy as np
import math
from collections import Counter


def count_uniques_per_attribute(data):
    uniques = list()
    for column in data.T:
        counter = Counter(column)
        uniques.append(counter)
    return uniques


def normalize_attributes_z_score(data):
    normalized = np.empty(data.shape)
    for i, column in enumerate(data.T):
        avg = np.mean(column)
        norm = column - avg
        # use unbiased std with division by N - 1
        std = np.std(column, ddof=1)
        if not math.isclose(std, 0.0):
            norm /= std
        normalized[:, i] = norm
    return normalized


def normalize_attributes_min_max(data):
    normalized = np.empty(data.shape)
    for i, column in enumerate(data.T):
        norm = (column - column.min()) / (column.max() - column.min())
        normalized[:, i] = norm
    return normalized

