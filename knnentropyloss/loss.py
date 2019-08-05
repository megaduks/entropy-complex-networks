import torch
import torch.nn as nn
from torch.nn import _reduction as _Reduction


def knn_entropy_loss(pred, targets, entropies):
    """
    Defines a custom loss function where the L1 loss is weighted by the inverse of the exponent
    of the entropy of a given instance's neighborhood

    :param pred: list of predicted values
    :param targets: list of target values
    :param entropies: list of entropies of instance neighborhoods
    :return: loss value
    """

    loss = torch.mean((1/torch.exp(torch.FloatTensor(entropies)))*torch.abs(pred - targets))

    return loss


class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class NeighborhoodEntropyLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, entropies, size_average=None, reduce=None, reduction='mean'):
        super(NeighborhoodEntropyLoss, self).__init__(size_average, reduce, reduction)
        self.entropies = entropies

    def forward(self, input, target):
        return knn_entropy_loss(input, target, entropies=self.entropies)


