import numpy as np
from typing import List


def precision_at_k(y_true: List, y_pred: List, k: int=1) -> float:
    """
    Computes precision@k metric for ranking lists

    params:
    :param y_true: list of real ranking of items
    :param y_pred: list of predicted ranking of items
    :param k: cut off value

    :returns the value of the precision@k metric
    """

    assert isinstance(k, int), 'k must be an integer'
    assert (k > 0), 'k must be positive'
    assert isinstance(y_pred, List), 'y_pred must be a list'

    common = set(y_pred[:k]).intersection(set(y_true[:k]))

    return len(common) / k


def gini(x: np.array) -> float:
    """
    Computes the value of the Gini index of a distribution

    params:
    :param X: array with the distribution

    :returns: the value of the Gini index
    """

    assert isinstance(x, np.ndarray), 'x must by an array'

    if x.sum() == 0:
        gini_index = 0
    else:

        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()

        # Relative mean absolute difference
        rmad = mad / np.mean(x)

        # Gini coefficient
        gini_index = 0.5 * rmad

    return gini_index