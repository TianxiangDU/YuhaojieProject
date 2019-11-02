import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def bended_l1_loss(pred, target, alpha=0.5):
    assert alpha > 0 and alpha < 1.0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    beta = 1 / (alpha+1) - 1
    loss = torch.where(diff < 1.0, torch.pow(diff, alpha+1) / (alpha+1),
                       diff + beta)
    return loss


@LOSSES.register_module
class BendedL1Loss(nn.Module):

    def __init__(self, alpha=0.5, reduction='mean', loss_weight=1.0):
        super(BendedL1Loss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * bended_l1_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
