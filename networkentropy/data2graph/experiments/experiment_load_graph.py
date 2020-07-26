# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example of transforming relational data to similarity graph
#
# ## Load libraries

# %% pycharm={"is_executing": false}
import networkx as nx

from functools import partial

from data2graph.measuree import measure, caterogical, numerical
from data2graph.measures.measure import Measure
from data2graph.network.network import Network
from data2graph.network.weight import Weight
from data2graph.network import network, load, weight, algorithm

from data2graph.datasets import loader

# %% [markdown]
# ## Load data

# %% pycharm={"is_executing": false}
X, y, types = loader.load_breast_cancer_short()

# %% [markdown]
# ## Choose method of measuring distance

# %% pycharm={"is_executing": false}
ms = Measure(numerical_strategy=numerical.euclidean, categorical_strategy=caterogical.goodall_3)

# %% [markdown]
# ## Choose way of  constructing graph, its density/threshold and method of weighting

# %% pycharm={"is_executing": false}
ns = Network(load_strategy=partial(load.load_graph_weight_distance_no_negative, alg="density", beta=0.1),
             weight_strategy=algorithm.weight_by_pagerank)

# %% pycharm={"name": "#%%\n", "is_executing": false}
measure_strategy = measure.Measure(numerical_strategy=numerical.manhattan, categorical_strategy=caterogical.iof)
measures = measure_strategy.compute(X, types)
network_strategy = network.Network(load_strategy=partial(load.load_graph_weight_similarity),
                                    weight_strategy=algorithm.weight_by_random)
G = network_strategy.load(measures, y)

# %% [markdown]
# ## Investigation of data similarity graph

# %% pycharm={"is_executing": false, "name": "#%%\n"}
betweenness = nx.betweenness_centrality(G)

# %% pycharm={"name": "#%%\n", "is_executing": false}
node_size = [betweenness[n]*1000 for n in G.nodes]
nx.draw(G, node_size=node_size, edge_color='gray', layout=nx.fruchterman_reingold_layout)
