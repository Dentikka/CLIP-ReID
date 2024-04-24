import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropyAttributes(nn.Module):
    """Cross entropy loss adjusted by attribute information.

    Args:
        attributes_data: DataFrame with attributes annotations for each class
    """

    def __init__(self, attributes_data, use_gpu=True):
        super(CrossEntropyAttributes, self).__init__()
        self.attributes_data = attributes_data
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        N = len(inputs)
        num_classes, num_attributes = self.attributes_data.shape

        log_probs = self.logsoftmax(inputs) 
        
        attributes_target = self.attributes_data.loc[targets.detach().cpu()].values
        attributes_target = np.broadcast_to(attributes_target[:, None, :], (N, num_classes, num_attributes))
        attributes_all = self.attributes_data.values
        attributes_all = np.broadcast_to(attributes_all[None, :, :], (N, num_classes, num_attributes))

        attributes_mask = (attributes_target == attributes_all).sum(axis=-1) / num_attributes
        attributes_mask = torch.tensor(attributes_mask, requires_grad=False)

        if self.use_gpu: attributes_mask = attributes_mask.cuda()
        loss = (- attributes_mask * log_probs).mean(0).sum()
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()