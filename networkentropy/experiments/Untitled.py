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

# %%
import numpy as np

# leaky rectified linear unit
lreLU = lambda x: x if x >= 0 else np.log10(np.abs(x)+1)

X = np.arange(-10, 10, 0.001)
Y = [ lreLU(x) for x in X ]

# %%
# %matplotlib inline
import matplotlib.pyplot as plt

plt.plot(X,Y)


# %%
def calculate_energy_gradients(g, energies, radius=1):
    result = {}
    for edge in g.edges:
        node1 = edge[0]
        node2 = edge[1]
        gradient = energies[node2] - energies[node1]
        result[edge] = gradient
    return result

    result = {
        edge: energies[node2] - energies[node1]
        for node1, node2 = 
    }
