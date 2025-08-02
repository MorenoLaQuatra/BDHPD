import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, loss_fns, weights=None):
        """
        Initialize the CombinedLoss class.
        
        :param loss_fns: List of loss functions (each should be a callable).
        :param weights: List of weights for each loss function. 
                        If None, all loss functions are equally weighted.
        """
        super(CombinedLoss, self).__init__()
        self.loss_fns = loss_fns
        if weights is None:
            self.weights = [1.0] * len(loss_fns)
        else:
            assert len(weights) == len(loss_fns), "Weights and loss functions must be of the same length."
            self.weights = weights

    def forward(self, inputs, targets):
        """
        Compute the combined loss.

        :param inputs: The inputs to the loss functions (typically model predictions).
        :param targets: The targets for the loss functions.
        :return: Weighted sum of the losses.
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.loss_fns, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        return total_loss