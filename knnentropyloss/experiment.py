from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

import numpy as np
from math import log, e

from model import Net
from loss import NeighborhoodEntropyLoss

from tqdm import tqdm
import matplotlib.pyplot as plt


X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    weights=[0.5, 0.5],
    flip_y=0.01,
    class_sep=1.0,
    random_state=42
)


def entropy(labels, base=None):
    """ Computes the entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    base = e if base is None else base

    for i in probs:
        ent -= i * log(i, base)

    return ent


if __name__ == "__main__":

    n_neighbors = 5
    epochs = 10000
    lr = 0.0001

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.FloatTensor)

    knn = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors, weights='distance').fit(X_train, y_train)

    nbr_idx = knn.kneighbors(X_train, n_neighbors=n_neighbors, return_distance=False)

    true_entropies = list()

    for nbr in nbr_idx:
        y_true = [y_train[i] for i in nbr]
        y_true_entropy = entropy(y_true, base=2)
        true_entropies.append(y_true_entropy)

    model = Net()

    # criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    criterion = NeighborhoodEntropyLoss(entropies=true_entropies)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for i in tqdm(range(epochs)):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(losses)
    plt.show()

    print(f'Accuracy: {accuracy_score(y_test, model.predict(X_test))}')

    print(model.predict(X_test))