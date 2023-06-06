import torch
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        # self.metric = metric
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, out0, out1, label):
        gt = label.float()
        D = self.distance(out0, out1).float().squeeze()
        loss = (1-gt) * 0.5 * torch.pow(D, 2) + (gt) * 0.5 * torch.pow(torch.clamp(self.margin - D, min=0.0), 2)
        return loss