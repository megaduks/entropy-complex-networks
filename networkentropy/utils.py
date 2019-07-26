import numpy as np
from typing import List, Dict


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
    :param x: array with the distribution

    :returns: the value of the Gini index
    """

    assert isinstance(x, np.ndarray), 'x must be an array'

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


def theil(x: np.array) -> float:
    """
    Computes the Theil index of the inequality of distribution (https://en.wikipedia.org/wiki/Theil_index)

    params:
    :param x: array with the distribution

    :returns: the value of the Theil index of the distribution
    """

    assert isinstance(x, np.ndarray), 'x must be an array'

    mi = x.mean()
    N = len(x)

    if mi == 0:
        theil_index = 0
    else:
        theil_index = (1 / N) * np.nansum((x / mi) * np.log(x / mi))

    return theil_index

def normalize_dict(d: Dict, target: float = 1.0) -> Dict:
    """
    Normalizes the values in the dictionary so that they sum up to factor

    :params
    :param d: dict to be normalized
    :param factor: value to which all values in the dictionary should sum up to
    :returns: normalized dictionary
    """

    assert isinstance(d, Dict), 'd must be a dictionary'
    raw = sum(d.values())

    if raw > 0:
        factor = target / raw
    else:
        factor = target

    return {key: value * factor for key, value in d.items()}