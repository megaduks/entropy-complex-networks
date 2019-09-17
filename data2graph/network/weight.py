import numpy as np


class Weight(object):
    def __init__(self, measure_strategy=None, network_strategy=None, invert_weights=False):
        self.measure_algorithm = measure_strategy
        self.network_algorithm = network_strategy
        self.invert_weights = invert_weights

    def count(self, X, y, types):
        measures = self.measure_algorithm.compute(X, types)
        G = self.network_algorithm.load(measures, y)
        if self.invert_weights:
            w = self.network_algorithm.weight(G)
            w = (w - np.min(w)) / (np.max(w) - np.min(w))
            return 1 - w
        else:
            # return self.network_algorithm.weight(G)
            w = self.network_algorithm.weight(G)
            w = (w - np.min(w)) / (np.max(w) - np.min(w))
            return w
