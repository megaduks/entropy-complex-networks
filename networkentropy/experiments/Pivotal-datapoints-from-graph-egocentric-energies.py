# -*- coding: utf-8 -*-
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

# %% [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Discovery-of-pivotal-data-instances-using-vertex-energy-and-data-similarity-graph" data-toc-modified-id="Discovery-of-pivotal-data-instances-using-vertex-energy-and-data-similarity-graph-1">Discovery of pivotal data instances using vertex energy and data similarity graph</a></span><ul class="toc-item"><li><span><a href="#Graph-energy" data-toc-modified-id="Graph-energy-1.1">Graph energy</a></span></li><li><span><a href="#Randić-energy" data-toc-modified-id="Randić-energy-1.2">Randić energy</a></span></li><li><span><a href="#Laplacian-energy" data-toc-modified-id="Laplacian-energy-1.3">Laplacian energy</a></span></li><li><span><a href="#Matrix-energies-for-various-topologies-of-small-egocentric-networks" data-toc-modified-id="Matrix-energies-for-various-topologies-of-small-egocentric-networks-1.4">Matrix energies for various topologies of small egocentric networks</a></span></li></ul></li><li><span><a href="#Fine-tuning-of-data-for-unsupervised-clustering" data-toc-modified-id="Fine-tuning-of-data-for-unsupervised-clustering-2">Fine-tuning of data for unsupervised clustering</a></span></li></ul></div>

# %%
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

# %% pycharm={"is_executing": false}
import sys
sys.path.append("..")
sys.path.append("...")

# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm_notebook as tqdm
from ggplot import *

import pandas as pd
import numpy as np
import networkx as nx
import scipy, scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import logging
import requests
import itertools

import network_energy as ne

from functools import partial

from data2graph.measures import measure, caterogical, numerical
from data2graph.measures.measure import Measure
from data2graph.network.network import Network
from data2graph.network.weight import Weight
from data2graph.network import network, load, weight, algorithm

from data2graph.datasets import loader


# %% [markdown] cell_style="center"
# # Discovery of pivotal data instances using vertex energy and data similarity graph
#
# In this experiment we verify the usefulness of the concept of graph energy in discovering pivotal data points.
# We transform relational dataset into graph representation where each data instance is represented by a single vertex
# and an edge exists between vertices if the similarity between data instances exceeds a given threshold.
#
# We use this data similarity graph to estimate the relative importance of data instances. An instance is important
# if its characterized by high betweenness in data similarity graph. Unfortunately, exact computation of betweenness 
# for large graphs is prohibitively expensive. We present a simple method which allows to estimate the betweenness
# of vertices with high precision using a novel concept of vertex energy.

# %% [markdown]
# ## Graph energy
#
# Graph energy of a graph is defined as $E_G(G) = \sum\limits_{i=1}^n |\mu_i|$, where $\mu_1, \ldots, \mu_n$ are the eigenvalues of the adjacency matrix $M_A$ (also known as the *spectrum* of the graph).
#
# ## Randić energy
#
# Randić matrix of the graph $G=\left<V, E\right>$ is defined as:
#
# $$
# M_R(i,j)=
# \begin{cases}
# 0 & \mathit{if} & i=j\\
# \frac{1}{\sqrt{d_i d_j}} & \mathit{if} & (i,j) \in E\\
# 0 & \mathit{if} & (i,j) \notin E
# \end{cases}
# $$
#
# Randić energy of a graph is defined as $E_R(G) = \sum\limits_{i=1}^n |\rho_i|$, where $\rho_1, \ldots, \rho_n$ are the eigenvalues of the Randić matrix $M_R$.
#
# ## Laplacian energy
#
# Laplacian matrix of the graph $G=\left<V, E\right>$ is defined as:
#
# $$
# M_L(i,j)=
# \begin{cases}
# d_i & \mathit{if} & i=j\\
# -1 & \mathit{if} & (i,j) \in E\\
# 0 & \mathit{otherwise}
# \end{cases}
# $$
#
# Laplacian energy of a graph is defined as $E_L(G) =  \sum\limits_{i=1}^n |\lambda_i - \frac{2m}{n}|$, where $\lambda_1, \ldots, \lambda_n$ are the eigenvalues of the Laplacian matrix $M_L$, $n$ is the number of vertices and $m$ is the number of edges in the graph $G$.

# %% pycharm={"is_executing": false}
def chunks(lst, n):
    """
    Divide a list of vertices `lst` into chunks consisting of `n` vertices
    
    Tests:
    >>> list(chunks([1,2,3,4,5,6], 2))
    [(1, 2), (3, 4), (5, 6)]

    >>> list(chunks([1,2,3,4,5,6], 4))
    [(1, 2, 3, 4), (5, 6)]

    >>> list(chunks([], 2))
    []

    """
    _lst = iter(lst)
    while 1:
        x = tuple(itertools.islice(_lst, n))
        if not x:
            return
        yield x
        
def normalize_df_column(df_column):
    """
    Normalize a dataframe column to the range [0,1]
    
    Tests:
    >>> normalize_df_column(pd.Series([1,2,3,4,5]))
    array([[0.  ],
           [0.25],
           [0.5 ],
           [0.75],
           [1.  ]])
    """
    x = df_column.values.astype(float)
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.reshape(-1,1))
    
    return x_scaled


# %% pycharm={"is_executing": false}
import doctest
doctest.testmod()

# %% [markdown]
# ## Matrix energies for various topologies of small egocentric networks
#
# Firstly, let us examine the relationship between the topology of a small egocentric network and its energies. We generate five different egocentric networks representing possible small scale configurations and compute all three types of matrix energies. The results are somehow surprising, graph energy tends to correlate with the degree of connectivity of the egocentric network, Randic energy remains practically constant, and Laplacian energy behaves unpredictably, receiving the maximum value for a custom topology. 

# %% pycharm={"is_executing": false}
g_custom = nx.star_graph(n=5)
g_custom.add_edge(1,2)
g_custom.add_edge(4,5)

graphs = [
    {'name': 'path', 'graph': nx.path_graph(n=3)},
    {'name': 'star', 'graph': nx.star_graph(n=5)},
    {'name': 'custom', 'graph': g_custom},
    {'name': 'wheel', 'graph': nx.wheel_graph(n=5)},
    {'name': 'complete', 'graph': nx.complete_graph(n=5)}
]

plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set(rc={'figure.figsize': (15, 4)})
fig, ax = plt.subplots(5, 1)

df = pd.DataFrame(
    {'name': [ g['name'] for g in graphs],
     'graph energy': [ne.get_graph_energy(g['graph']) for g in graphs],
     'randic energy': [ne.get_randic_energy(g['graph']) for g in graphs],
     'laplacian energy': [ne.get_laplacian_energy(g['graph']) for g in graphs]
    }
)

plt.subplot(151)
nx.draw(graphs[0]['graph'], node_color=['black','white','black'], edgecolors='black')
plt.title(graphs[0]['name'])

plt.subplot(152)
nx.draw(graphs[1]['graph'], node_color=['white','black', 'black', 'black', 'black', 'black'], edgecolors='black')
plt.title(graphs[1]['name'])

plt.subplot(153)
nx.draw(graphs[2]['graph'], node_color=['white','black','black', 'black', 'black', 'black'], edgecolors='black')
plt.title(graphs[2]['name'])

plt.subplot(154)
nx.draw(graphs[3]['graph'], node_color=['white','black','black', 'black', 'black'], edgecolors='black')
plt.title(graphs[3]['name'])

plt.subplot(155)
nx.draw(graphs[4]['graph'], node_color=['white','black','black', 'black', 'black'], edgecolors='black')
plt.title(graphs[4]['name'])

plt.show()

print(df[['name', 'graph energy', 'randic energy', 'laplacian energy']])

# %% [markdown]
# Next, we start with a star configuration of an egocentric network consisting of an ego and additional $n$ vertices, 
# and we gradually add all remaining edges, until we form a full $K_5$ graph. 
# For each intermediate graph we compute all its energies. We can clearly see that each of matrix energies 
# is measuring a different "aspect" of the egocentric network:
#
# * randic energy is maximized for topologies very close to the original star-like structure and diminishes as more and more edges are added to the egocentric network
# * laplacian energy strongly resembles the entropy of adjacency matrix, being maximized half-way between the star structure and the clique structure of the egocentric network
# * graph energy steadily grows as the density of the egocentric network increases.

# %% pycharm={"is_executing": false}
from itertools import combinations
from random import shuffle

g = nx.star_graph(n=25)

results = []

edges = list(combinations(range(1, len(g.nodes)), r=2))

# comment if you want to add edges in an ordered way
shuffle(edges)

for (idx, (i, j)) in enumerate(edges):
    results.append((idx, ne.get_graph_energy(g), ne.get_randic_energy(g),
                    ne.get_laplacian_energy(g)))

    g.add_edge(i, j)

# %% pycharm={"is_executing": false}
df = pd.DataFrame(
    data=results,
    columns=[
        'complexity', 'graph energy', 'randic energy', 'laplacian energy'
    ])

dfn = df[['graph energy','randic energy','laplacian energy']].apply(lambda s: s/s.max(), axis=0)
dfn['complexity'] = df['complexity']

# %% pycharm={"is_executing": false}
dfm = pd.melt(
    dfn,
    value_vars=['graph energy', 'randic energy', 'laplacian energy'],
    id_vars='complexity')

plt.style.use(['seaborn-white', 'seaborn-paper'])

sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(data=dfm, x='complexity', y='value', style='variable')

# %% [markdown]
# # Fine-tuning of data for unsupervised clustering
#
# In this experiment we create data similarity graphs for a few well-known datasets. 
# For each graph we collect detailed statistics on every vertex:
#
# - its betweenness,
# - its Randić energy,
# - its Laplacian energy,
# - and its graph energy.
#
# We normalize these variables using MinMax scaling to the range of [0-1]. 
#
# Finally, we compute the correlation between vertex betweenness and vertex energies.

# %%
datasets = {
    'inflammation': loader.load_diagnosis_inflammation,
    'diagnosis_nephritis': loader.load_diagnosis_nephritis,
    'iris': loader.load_iris,
    'titanic': loader.load_titanic,
    'lenses': loader.load_lenses,
    'mushrooms': loader.load_mushrooms,
    'breast_cancer': loader.load_breast_cancer_short,
    'wine_quality': loader.load_wine_quality_classification,
    'pima_diabetes': loader.load_pima_diabetes,
    'internet_ads': loader.load_internet_ads_pca,
    'housing_prices': loader.load_housing_prices_short,
    'ionosphere': loader.load_ionosphere,
    'monks1': loader.load_monks_1,
    'monks2': loader.load_monks_2,
    'monks3': loader.load_monks_3,
    'yeast': loader.load_yeast,
    'heart_statlog': loader.load_heart_statlog,
    'haberman': loader.load_haberman,
    'hepatitis': loader.load_hepatitis,
    'dermatology': loader.load_dermatology,
    'glass': loader.load_glass,
    'ecoli': loader.load_ecoli,
    'cmc': loader.load_cmc,
    'zoo': loader.load_zoo,
    'balance_scale': loader.load_balance_scale,
    'segmentation': loader.load_segmentation,
    'car': loader.load_car,
    'house_voting': loader.load_house_voting
}

# %%
for d in datasets:
    X, y, types = datasets[d]()
    
    if len(X) > 250:
        print(f'{d}: {len(X)} rows')

# %%
from sklearn.cluster import KMeans

def _compute_difference_clustering_scores(X, vals, n_clusters, score_function, top_k=10):
    
    model_full = KMeans(n_clusters=n_clusters).fit(X)

    original_score = score_function(X, model_full.labels_)
    
    if score_function == 'Calinski-Harabasz index':
        original_score /= len(X)

    top_order_idx = [ 
        idx
        for idx, val
        in sorted(vals, key=lambda x: x[1], reverse=True)[top_k:]
    ]

    X_lim = X[top_order_idx]

    model_reduced = KMeans(n_clusters=n_clusters).fit(X_lim)

    reduced_score = score_function(X_lim, model_reduced.labels_)
    
    if score_function == 'Calinski-Harabasz index':
        reduced_score /= len(X_lim)
    
    delta_score = (reduced_score - original_score) / original_score
    
    model_applied = model_reduced.predict(X)
    
    applied_score = score_function(X, model_applied)
    
    if score_function == 'Calinski-Harabasz index':
        applied_score /= len(X)

    applied_delta_score = (applied_score - original_score) / original_score
    
    # check if results hold in comparison with random order as well
    random_order_idx = [n for n in G.nodes]
    np.random.shuffle(random_order_idx)
    X_rand = X[random_order_idx[top_k:]]
    
    model_random = KMeans(n_clusters=n_clusters).fit(X_rand)

    random_score = score_function(X_rand, model_random.labels_)
    
    if score_function == 'Calinski-Harabasz index':
        random_score /= len(X)

    random_delta_score = (reduced_score - random_score) / random_score
    
    return delta_score, applied_delta_score, random_delta_score


# %%
from operator import itemgetter

def _find_optimum_number_clusters(X, scoring_function, n_values=10):
    
    scores = []
    
    k_min = 2
    k_max = np.sqrt(len(X)).astype(int)
    
    for k in np.linspace(k_min, k_max, n_values).astype(int).tolist():
        
        model = KMeans(n_clusters=k).fit(X)
        score = scoring_function(X, model.labels_)
        
        scores.append((k, score))
        
    if scoring_function == 'Davies-Bouldin index':
        return min(scores, key=itemgetter(1))[0] # get the index of the smallest value
    else:
        return max(scores, key=itemgetter(1))[0] # get the index of the largest value


# %% pycharm={"is_executing": false}
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

results = pd.DataFrame(columns=[
    'dataset',
    'order',
    'nodes',
    'betweenness',
    'randic_energy',
    'graph_energy',
    'score_name',
    'score_diff',
    'score_improve_applied',
    'score_improve_over_random',
    'top_k'
])

# TODO: repeat experiment for another size of neighborhood
radius = 1

# simple function to find quantiles
thresholds = lambda x: [ int(x*p) for p in np.arange(0.00, 0.11, 0.01)]

for dataset in tqdm(datasets):

    X, y, types = datasets[dataset]()

    if len(X) > 2000:
        continue
    if len(X) < 250:
        continue
        
    _thresholds = thresholds(len(X))

    measure_strategy = measure.Measure(numerical_strategy=numerical.mahalanobis,
                                       categorical_strategy=caterogical.goodall_3)

    measures = measure_strategy.compute(X, types)

    network_strategy = network.Network(load_strategy=partial(load.load_graph_weight_similarity, beta=0.15),
                                       weight_strategy=algorithm.weight_by_degree)

    G = network_strategy.load(measures, y)

    be_order = nx.betweenness_centrality(G, k=None, normalized=True)
    re_order = ne.randic_centrality(G, radius=radius, normalized=True)
    ge_order = ne.graph_energy_centrality(G, radius=radius, normalized=True)
    le_order = ne.laplacian_centrality(G, radius=radius, normalized=True)

    orders = {
        'betweenness': be_order,
        'randic': re_order,
        'graph': ge_order,
        'laplacian': le_order,
    }
    
    score_functions = {
        'Silhouette score': silhouette_score,
        'Calinski-Harabasz index': calinski_harabasz_score,
        'Davies-Bouldin index': davies_bouldin_score
    }

    for score_function in score_functions:
        
        n_clusters = _find_optimum_number_clusters(X, scoring_function=score_functions[score_function], n_values=5)
        
        for order in orders:

            for top_k in _thresholds:

                score_diff, score_applied_diff, score_random_diff = _compute_difference_clustering_scores(
                    X, orders[order].items(), n_clusters=n_clusters, score_function=score_functions[score_function], top_k=top_k)
                
                # davis-bouldin values are the other way around
                if score_function == 'Davies-Bouldin index':
                    score_diff *= -1
                    score_random_diff *= -1
                    score_random_diff *= -1

                _dict = {
                    'dataset': dataset,
                    'order': order,
                    'nodes': list(G.nodes),
                    'betweenness': list(be_order.values()),
                    'randic_energy': list(re_order.values()),
                    'graph_energy': list(ge_order.values()),
                    'laplacian_energy': list(le_order.values()),
                    'score_function': score_function,
                    'score_diff': score_diff,
                    'score_improve_applied': score_applied_diff,
                    'score_improve_over_random': score_random_diff,
                    'top_k': top_k,
                }

                _result = pd.DataFrame.from_dict(_dict)

                results = pd.concat([results, _result], axis=0)

# %%
df_plot = results.groupby(['order','top_k', 'score_function'])[[
    'score_diff',
    'score_improve_applied',
    'score_improve_over_random'
]].mean().reset_index()

# %%
df_plot.groupby(['score_function','order']).agg(['mean'])

# %% pycharm={"name": "#%%\n"}
for score_function in score_functions:
    
    for y in ['score_diff', 'score_improve_applied', 'score_improve_over_random']:

        plt.figure()
        
        g = sns.barplot(
            data=df_plot[df_plot.score_function == score_function], 
            x='order',
            y=y
        ).set_title(f'{score_function} {y}')

        g

        g = sns.barplot(
            data=df_plot[df_plot.score_function == score_function], 
            x='order',
            y=y
        ).set_title(f'{score_function} {y}')

        g

        g = sns.barplot(
            data=df_plot[df_plot.score_function == score_function], 
            x='order',
            y=y
        ).set_title(f'{score_function} {y}')

        g

# %%
