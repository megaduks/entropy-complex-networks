
class Network(object):
    """
    Network:
        Objects of class Network are responsible for creating graph and counting weights
    Args:
        load_strategy:
            use load.load_graph_inverted_weight_no_negative in case e.g. pagerank because it may not
            converge with negative distances, otherwise you can use load.load_graph_inverted_weight.
            To pass function with more args use e.g. "partial" module
        alg:
             algorithm of setting down threshold, pass "density" or "percentile"
        beta:
             parameter used by algorithm "alg" e.g beta=0.1 equals density 0.1
        weight_strategy:
            weight_strategy, you can pass different like pagerank, betweenness etc.
            methods like Katz centrality, eigenvector centrality, pagerank may not converge
    """
    def __init__(self, load_strategy=None, weight_strategy=None):
        self.load_algorithm = load_strategy
        self.weight_algorithm = weight_strategy

    def load(self, data, y_label):
        return self.load_algorithm(data, y_label)

    def weight(self, G):
        weights = self.weight_algorithm(G)
        return weights
