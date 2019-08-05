import torch
import torch.nn as nn
from torch import tanh, relu_, sigmoid


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = tanh(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        pred = self.forward(x)
        binarize = lambda x: 0 if x <= 0.5 else 1
        ans = [ binarize(p) for p in pred ]

        return torch.tensor(ans)