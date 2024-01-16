import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight=1.8, neg_weight=1.):
        """
        Args:
            pos_weight (float, optional): A weight for the positive class. Defaults to None.
            neg_weight (float, optional): A weight for the negative class. Defaults to None.
        """
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor of the same shape as input.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Compute binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # Apply class-specific weights
        if self.pos_weight is not None:
            pos_mask = target == 1.
            bce_loss[pos_mask] *= self.pos_weight

        if self.neg_weight is not None:
            neg_mask = target == 0.
            bce_loss[neg_mask] *= self.neg_weight

        return torch.mean(bce_loss) # Reduction is always 'mean'
