from .helpers import normalize_attributes_z_score
from scipy.spatial.distance import pdist, squareform


def euclidean(x_data):
    # norm = normalize_attributes_z_score(x_data)
    return squareform(pdist(x_data, 'euclidean'))


def manhattan(x_data):
    # norm = normalize_attributes_z_score(x_data)
    return squareform(pdist(x_data, 'cityblock'))


def cosine(x_data):
    """
    1 - cosine

    Returns
        Values from 0 to 1 when data is positive, unless from 0 to 2
    """
    return squareform(pdist(x_data, 'cosine'))


def correlation(x_data):
    """
    1 - Pearson correlation

    Returns:
        Values from 0 to 2
    """
    return squareform(pdist(x_data, 'correlation'))


def mahalanobis(x_data):
    # no normalization, z score would transform it to euclidean distance
    return squareform(pdist(x_data, 'mahalanobis'))
