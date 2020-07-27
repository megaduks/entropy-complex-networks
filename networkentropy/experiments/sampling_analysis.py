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
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set(rc={'figure.figsize':(20,16)})

# %%
# %matplotlib inline

# %%
file = 'sampling_results.csv'

df = pd.read_csv(file)

# %%
models = df.model.unique()
functions = df.function.unique()
parameters = df.param.unique()
sample_ratios = df.sample_ratio.unique()

# %%
df_plot = pd.melt(df, 
                  id_vars=['model','function','param','sample_ratio'], 
                  value_vars=['degree','betweenness','pagerank','closeness','clustering'])

df_plot.head()

# %%
models

# %%
df_idx = df_plot.model == 'smallworld'

dfm = df_plot[df_idx]

# %%
sns.lineplot(data=dfm, x='sample_ratio', y='value', hue='variable', estimator='mean')

# %%
g = sns.FacetGrid(dfm, col='function', col_wrap=3)
g.map(sns.lineplot, 'sample_ratio', 'value', 'variable')

# %%
pd.options.display.float_format = '{:,.4f}'.format

dfm[['function', 'sample_ratio', 'variable','value']].groupby(['function', 'sample_ratio', 'variable']).agg([np.mean, np.min, np.max])
