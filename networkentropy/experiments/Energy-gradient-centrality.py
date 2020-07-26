# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:entropy]
#     language: python
#     name: conda-env-entropy-py
# ---

# %% [markdown]
# # Energy gradient pagerank centrality

# %%
from tqdm.auto import tqdm
from collections import OrderedDict, namedtuple
from ggplot import *
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid

import itertools
import pandas as pd
import networkx as nx
import numpy as np
import sys
import qgrid

sys.path.append("../..")
from networkentropy import network_energy_gradient as neg, network_utils as nu

tqdm.pandas()

# %% [markdown]
# ## Synthetic networks

# %%
SEED = 10
NUM_NODES = 90

PARAM1_MAX = 5
PARAMS_N_P = ParameterGrid({'n': [NUM_NODES], 'p': [p/PARAM1_MAX for p in range(1, PARAM1_MAX)]})
PARAMS_N = ParameterGrid({'n': [NUM_NODES]})
PARAMS_N_K_ALPHA = ParameterGrid({'n': [NUM_NODES], 'k': [3], 'alpha': range(10, PARAM1_MAX*10, 10)})

Gen = namedtuple('Gen', ['generator', 'params'])
GENERATORS = {
    'random':     Gen(lambda n, p: nx.erdos_renyi_graph(n=n, p=p, directed=True, seed=SEED), PARAMS_N_P),
    'k_out':      Gen(lambda n, k, alpha: nx.DiGraph(nx.random_k_out_graph(
                      n=n, k=k, alpha=alpha, self_loops=False, seed=SEED)),                  PARAMS_N_K_ALPHA),
    'scale_free': Gen(lambda n: nx.DiGraph(nx.scale_free_graph(n=n,seed=SEED)),              PARAMS_N)
                 
}

RANDOM, K_OUT, SCALE_FREE = GENERATORS.keys()

METHODS = ['randic', 'laplacian', 'graph']
RANDIC, LAPLACIAN, GRAPH = METHODS


# %%
def create_graphs(generators: dict):
    graphs = []
    for name, (generator, params) in tqdm(list(generators.items())):
        for i, params_row in enumerate(params):
            row = {**params_row, 'generator': name, 'params_num': i, 'graph': generator(**params_row), 
                   'id': f'{name} {params_row}'}
            graphs.append(row)
    return pd.DataFrame(graphs)


# %%
graphs_df = create_graphs(GENERATORS)

# %%
graphs_df


# %% [markdown]
# ### Calculating energy gradient centralities
#
# Firsly pagerank is calculated for every graph and then histograms of energyu gradient centrality are ploted

# %%
def calculate_pagerank(graphs_df: pd.DataFrame, methods, max_iter=1000, tol=0.001) -> pd.DataFrame:
    for method in methods:
        pagerank_df = graphs_df.progress_apply(lambda s: neg.get_energy_gradient_centrality(s['graph'], method, 
                                                                                               max_iter=max_iter, 
                                                                                               tol=tol),
                                     axis=1)
        graphs_df = graphs_df.assign(**{f'{method}_pagerank': pagerank_df})
    return graphs_df


# %%
methods = set(METHODS).difference([LAPLACIAN])
graph_pagerank_df = calculate_pagerank(graphs_df, methods, max_iter=1000, tol=1.0e-4)

# %%
graph_pagerank_df


# %% [markdown]
# ### Pagerank calculations results
#
# For some graphs number of iterations have been exceeded before reaching tolerance. Those graphs will be further ignored.

# %%
def flatten_pagerank_df(pagerank_df: pd.DataFrame, methods, additional_attrs = []):
    pagerank_attrs = methods
    if len(additional_attrs) <= 0:
        additional_attrs = list(set(pagerank_df.columns).difference(pagerank_attrs))
    flatten_rows = []
    for _, row in pagerank_df.loc[:, pagerank_attrs + additional_attrs].iterrows():
        additional_attrs_values = [(attr, val) for attr, val in zip(additional_attrs, row[len(pagerank_attrs):])]
        for name, pagerank in zip(pagerank_attrs, row):
            if pagerank:
                for node, value in pagerank.items():
                    flatten_rows.append(OrderedDict(additional_attrs_values +
                                                    [('node', node), ('method', name), ('pagerank', value)]))
    return pd.DataFrame(flatten_rows)


# %%
pagerank_methods = [f'{m}_pagerank' for m in methods]

# %%
flat_graph_pagerank_df = flatten_pagerank_df(graph_pagerank_df, pagerank_methods)

# %%
flat_graph_pagerank_df.head()

# %% [markdown]
# ### Histograms

# %%
BINS = 30
p = ggplot(aes(x='pagerank'), data=flat_graph_pagerank_df) + \
geom_histogram(alpha=0.4, color='blue', size=3, bins=BINS) + \
facet_grid('id','method', scales = "free")

p.make()
p.fig.set_size_inches(15, 120, forward=True)
p.fig.set_dpi(100)
p.fig

plt.show()


# %%
def test_normality(flat_pagerank_df: pd.DataFrame, grouping_attrs):
    return flat_pagerank_df.loc[:, grouping_attrs + ['pagerank']] \
                                              .groupby(grouping_attrs) \
                                              .agg(lambda series: stats.normaltest(series))


# %% [markdown]
# ### Normality test
#
# Only for graph: random {'n': 90, 'p': 0.8} gradient centralities are normally distributed 

# %%
test_normality(flat_graph_pagerank_df, ['id', 'method'])


# %% [markdown]
# ### Visualisation of energy grdaient centrality in comparison to energy

# %%
def get_graph_with_energy_data(graph: nx.Graph, methods, max_iter, tol): 
    graph = neg.get_graph_with_energy_data(graph, methods=methods, clear=True)
    graph = neg.get_graph_with_energy_gradient_centrality(graph, methods=methods, max_iter=max_iter, tol=tol)
    return graph


# %%
synthetic_graphs_with_data_series = graph_pagerank_df.progress_apply(
    lambda s: get_graph_with_energy_data(s['graph'], methods, max_iter=2000, tol=1.0e-4), axis=1)
graph_pagerank_df = graph_pagerank_df.assign(graph=synthetic_graphs_with_data_series)


# %%
def draw_network(graph, node_attribute, edge_attribute, title, nodes_bar_title, edges_bar_title):
    nodes_colors = list(nx.get_node_attributes(graph, node_attribute).values())
    edges_colors = list(nx.get_edge_attributes(graph, edge_attribute).values())
    if nodes_colors and edges_colors:
        plt.figure(figsize=(18, 12))
        pos = nx.layout.kamada_kawai_layout(graph)
        nodes = nx.draw_networkx_nodes(graph, pos, node_color=nodes_colors, cmap=plt.cm.coolwarm, 
                                       with_labels=False)
        edges = nx.draw_networkx_edges(graph, pos, edge_color=edges_colors, edge_cmap=plt.cm.Reds, width=2, 
                                       with_labels=False)
    #     edges_cbar = plt.colorbar(edges)
    #     edges_cbar.ax.set_title(edges_bar_title)
        nodes_cbar = plt.colorbar(nodes)
        nodes_cbar.ax.set_title(nodes_bar_title)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        print(f'Graph {title} does not contain {node_attribute} or {edge_attribute}')


# %%
def draw_synthetic_networks(graph_pagerank_df: pd.DataFrame, method: str):
    for _, (graph, graph_id) in tqdm(list(graph_pagerank_df.loc[:,['graph', 'id']].iterrows())):
        title = f'Graph energy centrality, {graph_id}'
        nodes_bar_title = 'energy gradient centrality'
        edges_bar_title = 'gradient'
        draw_network(graph, neg._get_centrality_name(method), neg._get_gradient_method_name(method), title, 
                     nodes_bar_title, edges_bar_title)
        title = f'Graph energy, generator: {graph_id}'
        nodes_bar_title = 'energy'
        draw_network(graph, neg._get_energy_method_name(method), neg._get_gradient_method_name(method), title, 
                     nodes_bar_title, edges_bar_title)


# %%
draw_synthetic_networks(graph_pagerank_df, GRAPH)

# %%
draw_synthetic_networks(graph_pagerank_df, RANDIC)

# %% [markdown]
# ## Empirical networks

# %%
datasets = nu.create_datasets('konect.uni').filter(directed=True, min_size=50, max_size=1000, max_density=0.5)
networks_df = datasets.download_and_build_networks('data/')

# %% [markdown]
# ### Calculating energy gradient centralities

# %%
empirical_graph_pagerank_df = calculate_pagerank(networks_df, methods, max_iter=2000, tol=1.0e-2)

# %%
empirical_graph_pagerank_df

# %%
flat_empirical_graph_pagerank_df = flatten_pagerank_df(empirical_graph_pagerank_df, pagerank_methods)

# %% [markdown]
# ### Histograms

# %%
p = ggplot(aes(x='pagerank'), data=flat_empirical_graph_pagerank_df) + \
geom_histogram(alpha=0.4, color='blue', size=3, bins=BINS) + \
facet_grid('name', 'method', scales = "free")

p.make()
p.fig.set_size_inches(15, 40, forward=True)
p.fig.set_dpi(100)
p.fig

plt.show()

# %% [markdown]
# ### Normality test
#
# None of the graphs has normally distributed energy gradient centrality

# %%
test_normality(flat_empirical_graph_pagerank_df, ['name', 'method'])

# %% [markdown]
# ###  Visualisation of energy grdaient centrality in comparison to energy

# %%
get_graph_with_gradient_centrality = lambda s: get_graph_with_energy_data(s['graph'], methods, max_iter=2000, tol=1.0e-2)
empirical_graphs_with_data_series = empirical_graph_pagerank_df.progress_apply(get_graph_with_gradient_centrality, axis=1)
empirical_graph_pagerank_df = empirical_graph_pagerank_df.assign(graph=empirical_graphs_with_data_series)

# %%
empirical_graph_pagerank_df

# %%
for _, (graph, name) in tqdm(list(empirical_graph_pagerank_df.loc[:,['graph', 'name']].iterrows())):
    if graph:
        title = f'Graph energy centrality, name: {name}'
        nodes_bar_title = 'energy gradient centrality'
        edges_bar_title = 'gradient'
        draw_network(graph, neg._get_centrality_name(GRAPH), neg._get_gradient_method_name(GRAPH), title, 
                     nodes_bar_title, edges_bar_title)
        title = f'Graph energy, name: {name}'
        nodes_bar_title = 'energy'
        draw_network(graph, neg._get_energy_method_name(GRAPH), neg._get_gradient_method_name(GRAPH), title,
                     nodes_bar_title, edges_bar_title)


# %% [markdown]
# ## Correlation between energy gradient centralities and other nodes descriptors

# %%
def sums_helper(length,total_sum, max_sum):
    if length < 1:
        raise ValueError("Length must be positive")
    if length == 1:
        yield (total_sum/max_sum,)
    else:
        for value in range(1, total_sum):
            for permutation in sums_helper(length - 1,total_sum - value, max_sum):
                yield (value/max_sum,) + permutation
                
def sums(length,total_sum):
    return sums_helper(length, total_sum, total_sum)


# %%
ABG = list(sums(3, 14))
PARAM2_MAX = len(ABG) + 1
NUM_NODES = 100
PARAMS_N_P = ParameterGrid({'n': [NUM_NODES], 'p': [p/PARAM1_MAX for p in range(1, PARAM2_MAX)]})
PARAMS_N = ParameterGrid({'n': [NUM_NODES]})
PARAMS_N_K_ALPHA = ParameterGrid({'n': [NUM_NODES], 'k': [3], 'alpha': range(1, PARAM2_MAX)})
PARAMS_N_A_B_G = [{'n': NUM_NODES, 'a': a, 'b': b, 'g': g} for a, b, g in ABG]

GENERATORS2 = {
    'random':     Gen(lambda n, p: 
                      nx.erdos_renyi_graph(n=n, p=p, directed=True, seed=SEED),                 PARAMS_N_P),
    'k_out':      Gen(lambda n, k, alpha: nx.DiGraph(nx.random_k_out_graph(
                      n=n, k=k, alpha=alpha, self_loops=False, seed=SEED)),                     PARAMS_N_K_ALPHA),
    'scale_free': Gen(lambda n, a, b, g: 
                      nx.DiGraph(nx.scale_free_graph(n=n,alpha=a, beta=b, gamma=g, seed=SEED)), PARAMS_N_A_B_G)
                 
}

# %%
synthetic_graphs_df = create_graphs(GENERATORS2)

# %%
CENTRALITIES = {
    'eigencentrality': nx.pagerank,
    'betweenness': nx.betweenness_centrality,
    'closeness': nx.closeness_centrality,
    'degree': nx.degree_centrality,
    'graph_gradient': lambda g: neg.get_energy_gradient_centrality(g, method=GRAPH, max_iter=1000, tol=1.0e-3),
    'randic_gradient': lambda g: neg.get_energy_gradient_centrality(g, method=RANDIC, max_iter=1000, tol=1.0e-3),
}

def calculate_centralities(graph_df, centralities):
    for name, function in tqdm(list(centralities.items())):
        centrality_series = graph_df.progress_apply(lambda s: function(s['graph']), axis=1)
        graph_df = graph_df.assign(**{name: centrality_series})
    return graph_df


# %%
synthetic_graphs_centralities_df = calculate_centralities(synthetic_graphs_df, CENTRALITIES)

# %%
synthetic_graphs_centralities_df.isnull().sum()


# %%
def flatten_centralities_df(centralities_df, centrality_names):
    other_attrs = list(set(centralities_df.columns).difference(centrality_names))
    centrs_length = len(centrality_names)
    rows = []
    for _, row in centralities_df.loc[:, centrality_names + other_attrs].iterrows():
        other_attrs_dict = {name: value for name, value in zip(other_attrs, row[centrs_length:])}
        centrs = {name: centr for name, centr in zip(centrality_names, row[:centrs_length]) if centr}
        for r in zip(*[r.items() for r in centrs.values()]):
            nodes, values = zip(*r)
            assert len(set(nodes)) == 1
            rows.append({**{name: value for name, value in zip(centrality_names, values)}, **other_attrs_dict})
    return pd.DataFrame(rows)


# %%
flat_synthetic_graphs_centralities_df2 = flatten_centralities_df(synthetic_graphs_centralities_df, list(CENTRALITIES.keys()))

# %%
synthethic_centralities_corrs_df = flat_synthetic_graphs_centralities_df2.loc[:, ['id', 'params_num', 'generator'] + list(CENTRALITIES.keys())] \
.groupby(['params_num', 'id', 'generator']).corr()

# %%
reset_synthethic_centralities_corrs_df = synthethic_centralities_corrs_df.reset_index()

# %%
reset_synthethic_centralities_corrs_df.head()

# %%
id_vars = ['params_num','generator', 'level_3', 'id']
value_vars = list(set(reset_synthethic_centralities_corrs_df.columns).difference(id_vars))

melted_synthethic_centralities_corrs_df = reset_synthethic_centralities_corrs_df.melt(id_vars=id_vars,
                                             value_vars=value_vars)

# %%
plots = {}
for centrality in ['graph_gradient', 'randic_gradient']:
    df = melted_synthethic_centralities_corrs_df.query('level_3 == @centrality and variable != level_3')

    p = ggplot(aes(x='params_num', y='value', color='variable'), data=df) + \
            geom_line() + facet_grid('variable', 'generator') + \
            ggtitle(f'Correlation of {centrality} with other node descriptors')
    plots[centrality] = p

# %%
plots.values()

# %%
qgrid.show_grid(melted_synthethic_centralities_corrs_df)

# %%
