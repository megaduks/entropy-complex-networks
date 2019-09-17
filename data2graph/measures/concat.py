import numpy as np


def _get_mask(desc, mask_word):
    mask = np.zeros(desc.shape, dtype=bool)
    for i, d in enumerate(desc):
        if d == mask_word:
            mask[i] = True
    return mask


def weighted_average_algorithm(x_data, column_descriptions, num_alg, cat_alg):
    num_mask = _get_mask(column_descriptions, "numerical")
    cat_mask = _get_mask(column_descriptions, "categorical")

    # similarity stays
    cat_matrix = cat_alg(x_data[:, cat_mask].astype(int))

    # normalize numerical attributes to <0;1>
    num_matrix = num_alg(x_data[:, num_mask])
    num_matrix = (num_matrix - np.min(num_matrix)) / (np.max(num_matrix) - np.min(num_matrix))
    # convert distance to simalarity
    num_matrix = 1 - num_matrix

    # weighted sum
    num_number = np.sum(num_mask)
    cat_number = np.sum(cat_mask)
    return ((num_matrix * num_number) + (cat_matrix * cat_number)) / (num_number + cat_number)
