# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %%
# %matplotlib inline

import networkx as nx
import pandas as pd
import numpy as np
import scipy as sp
import scipy as spy
import qgrid

from tqdm import tqdm_notebook as tqdm
from collections import OrderedDict
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

import itertools
import sys

sys.path.append("../..")
from networkentropy import network_energy as ne, network_energy_gradient as neg, network_utils as nu

# %% [markdown]
# ## Correlaction between energy gradient and length of the shortest path for all node pairs
# ### For sythetic networks

# %% [markdown]
# #### Define constants

# %%
GENERATORS = ['random', 'smallworld', 'waxman', 'powerlaw']
RANDOM, SMALLWORLD, WAXMAN, POWERLAW = GENERATORS

METHODS = ['randic', 'laplacian', 'graph']
RANDIC, LAPLACIAN, GRAPH = METHODS

PARAMETER_MAX = 4
PARAMETERS = list(range(1, PARAMETER_MAX)) #parameters for synthetic networks models


# %% [markdown]
# #### Create synthetic networks

# %%
def create_graph(p, generator, p_max, num_nodes=100):
    if generator == RANDOM:
        return nx.erdos_renyi_graph(n=num_nodes, p=p/(p_max*10))
    elif generator == SMALLWORLD:
        return nx.watts_strogatz_graph(n=num_nodes, k=4, p=p/p_max)
    elif generator == WAXMAN:
        return nx.waxman_graph(n=num_nodes, alpha=p/p_max, beta=0.1)
    elif generator == POWERLAW:
        return nx.powerlaw_cluster_graph(n=num_nodes, m=3, p=p/(p_max*10))
    else:
        raise ValueError('Generator: {} does not exist'.format(generator))


# %%
def create_graphs(paramters, parameter_max, generators):
    graphs = []
    parameters_generators = list(itertools.product(paramters, generators))
    for p, generator in tqdm(parameters_generators):
        graph = create_graph(p, generator, parameter_max)
        graphs.append(OrderedDict([('param', p), 
                                   ('generator', generator), 
                                   ('graph', graph),]))
    return pd.DataFrame(graphs)


# %%
graphs_df = create_graphs(PARAMETERS, PARAMETER_MAX, GENERATORS)

# %%
graphs_df.head()


# %% [markdown]
# #### Coalculate shortest paths and gradients for the networks

# %%
def compute_shortest_paths_and_gradients_for_graph(graph: nx.Graph, additional_attrs: dict, methods):
    shortest_paths = nx.shortest_path(graph)
    decorated_graph = neg.get_graph_with_energy_data(graph, methods, copy=False)
    results = []
    for source, paths in shortest_paths.items():
        for target, path in paths.items():
            length = len(path) - 1
            if length > 0:
                for method in methods:
                    gradient = decorated_graph.get_gradient(source, target, method)
                    path_energy = decorated_graph.get_path_energy(path, method)
                    row = OrderedDict(additional_attrs)
                    row.update(OrderedDict([('method', method), 
                                            ('source', source), 
                                            ('target', target), 
                                            ('length', length), 
                                            ('gradient', gradient),
                                            ('abs_gradient', np.abs(gradient)),
                                            ('path_energy', path_energy),
                                            ('avg_path_energy', path_energy/len(path)),]))
                    results.append(row)
    return results
    

def compute_shortest_paths_and_gradients_for_synthetic_networks(graphs_df, methods):
    results = []
    for _, (p, generator, graph) in tqdm(list(graphs_df.iterrows())):
        additional_attrs = OrderedDict([('param', p),
                                        ('generator', generator),])
        results.extend(compute_shortest_paths_and_gradients_for_graph(graph, additional_attrs, methods))
    return pd.DataFrame(results)


# %%
paths_data_df = compute_shortest_paths_and_gradients_for_synthetic_networks(graphs_df, METHODS)

# %%
qgrid.show_grid(paths_data_df)


# %% [markdown]
# #### Check is data destribution is normal

# %%
def draw_qqplot(data):
    qqplot(data, line='s')
    pyplot.show()


# %%
draw_qqplot(paths_data_df.length)

# %%
spy.stats.normaltest(paths_data_df.length)

# %%
draw_qqplot(paths_data_df.gradient)

# %%
spy.stats.normaltest(paths_data_df.gradient)

# %%
draw_qqplot(paths_data_df.path_energy)

# %%
spy.stats.normaltest(paths_data_df.path_energy)

# %%
draw_qqplot(paths_data_df.avg_path_energy)

# %%
spy.stats.normaltest(paths_data_df.avg_path_energy)


# %% [markdown]
# ##### Conclusion
#
# Results clearly show the data is not normally distributed for all of the analyzed columns. It is visible on qqplot and also p value of statistical test is zero or very near to zero for all the columns. 

# %% [markdown]
# #### Calculate correlations between shortest path lengths and energy gradients

# %%
def compute_correlation(grouped_df, column1, column2, methods):
    corr_series = []    
    for method in methods:
        df = grouped_df.corr(method=method)
        index_len = len(df.index.names)
        index_slices = (index_len - 1) * (slice(None),) + (column1,)
        series = df.loc[index_slices, column2]
        series.name = method
        corr_series.append(series)
    corr_df =  pd.concat(corr_series, axis=1)
    corr_df.index = corr_df.index.droplevel(-1)
    return corr_df

def compute_correlation_for_synthetic_networks(paths_gradients_df, column1, column2, methods=['pearson', 'kendall', 'spearman']):
    grouped_df = paths_gradients_df\
                    .query('target > source')\
                    .loc[:, ['method', 'generator', 'param', column1, column2]]\
                    .groupby(['method', 'generator', 'param'])
    return compute_correlation(grouped_df, column1, column2, methods)


# %%
length_abs_gradient_corr_df = compute_correlation_for_synthetic_networks(paths_data_df, 'length', 'abs_gradient')
qgrid.show_grid(length_abs_gradient_corr_df)

# %%
path_energy_length_corr_df = compute_correlation_for_synthetic_networks(paths_data_df, 'length', 'path_energy')
qgrid.show_grid(path_energy_length_corr_df)

# %%
avg_path_energy_length_corr_df = compute_correlation_for_synthetic_networks(paths_data_df, 'length', 'avg_path_energy')
qgrid.show_grid(avg_path_energy_length_corr_df)


# %% [markdown]
# ##### Conclusion
#
# We can observe small correlation for gradient and average path energy, however its maximum values are about -0.35.
# There is a very high correlation between path energy and length of the shortest path but these measures are dependent as sum for longer element has more elements so this probably doesn't bring much information.

# %% [markdown]
# #### Calculate it again using custom function

# %% [markdown]
# It is redundant repetition but I leave it for now

# %%
def compute_correlations(parameters, generators, methods, paths_gradients_df):
    correlations = []
    for p, generator, method in tqdm(list(itertools.product(parameters, generators, methods))):
        sub_df = paths_gradients_df.query("param=='{}' and generator=='{}' and method=='{}'".format(p, generator, method))
        #leave out negative equivalents
        sub_df = sub_df.query("target > source")
        sub_df = sub_df.assign(abs_gradient=np.abs(sub_df.gradient))
        x = list(sub_df.length)
        y = list(sub_df.abs_gradient)
        corr = spy.stats.pearsonr(x, y)
        correlations.append(OrderedDict([('p', p), 
                                         ('generator', generator), 
                                         ('method', method), 
                                         ('corr', corr)]))
    return pd.DataFrame(correlations)


# %%
corr_df2 = compute_correlations(PARAMETERS, GENERATORS, METHODS, paths_data_df)
qgrid.show_grid(corr_df2.sort_values('corr').head())

# %% [markdown]
# ### For empirical networks

# %%
datasets = nu.create_datasets('konect.uni').filter(min_size=50, max_size=500, max_density=0.1)

# %%
datasets.to_df().head()

# %%
networks_df = datasets.download_and_build_networks('data/')


# %%
def compute_shortest_paths_and_gradients_for_empirical_networks(networks_df: pd.DataFrame, methods):
    results = []
    for _, (name, category, _, _, _, graph) in tqdm(list(networks_df.iterrows())):
        print(name)
        additional_attrs = OrderedDict([('name', name),                                        
                                        ('category', category),])
        results.extend(compute_shortest_paths_and_gradients_for_graph(graph, additional_attrs, methods))
    return pd.DataFrame(results)


# %%
paths_data_empirical_df = compute_shortest_paths_and_gradients_for_empirical_networks(networks_df, METHODS)

# %%
paths_data_empirical_df.head()

# %%
draw_qqplot(paths_data_empirical_df.length)

# %%
spy.stats.normaltest(paths_data_empirical_df.length)

# %%
draw_qqplot(paths_data_empirical_df.gradient)

# %%
spy.stats.normaltest(paths_data_empirical_df.gradient)

# %%
draw_qqplot(paths_data_empirical_df.abs_gradient)

# %%
spy.stats.normaltest(paths_data_empirical_df.abs_gradient)

# %%
draw_qqplot(paths_data_empirical_df.path_energy)

# %%
spy.stats.normaltest(paths_data_empirical_df.path_energy)

# %%
draw_qqplot(paths_data_empirical_df.avg_path_energy)

# %%
spy.stats.normaltest(paths_data_empirical_df.avg_path_energy)


# %%
def compute_correlation_for_empirical_networks(paths_gradients_df, column1, column2, methods=['pearson', 'kendall', 'spearman']):
    grouped_df = paths_gradients_df\
                    .query('target > source')\
                    .loc[:, ['category', 'name', 'method', column1, column2,]]\
                    .groupby(['category', 'name', 'method',])
    return compute_correlation(grouped_df, column1, column2, methods)


# %%
length_abs_gradient_corr_empirical_df = \
compute_correlation_for_empirical_networks(paths_data_empirical_df,  'length', 'abs_gradient')
qgrid.show_grid(length_abs_gradient_corr_empirical_df)

# %%
length_path_energy_corr_empirical_df = \
compute_correlation_for_empirical_networks(paths_data_empirical_df,  'length', 'path_energy')
qgrid.show_grid(length_path_energy_corr_empirical_df)

# %%
length_avg_path_energy_corr_empirical_df = \
compute_correlation_for_empirical_networks(paths_data_empirical_df,  'length', 'avg_path_energy')
qgrid.show_grid(length_avg_path_energy_corr_empirical_df)

# %%
