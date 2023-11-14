import torch
import torch.distributed.nn
from torch import nn as nn


class CLIPBCELoss(nn.Module):
    def __init__(self, clip_loss, bce_loss, clip_loss_weight=None, bce_loss_weight=None):
        super().__init__()
        self._clip_text_loss = clip_loss
        self._bce_text_loss = bce_loss
        self._clip_loss_weight = clip_loss_weight
        self._bce_loss_weight = bce_loss_weight
        self._num_tasks = 2

    def forward(self, text_features, label_features, logit_scale):
        clip_text_loss = self._clip_text_loss(
            text_features=text_features,
            label_features=label_features,
            logit_scale=logit_scale
        )
        bce_text_loss = self._bce_text_loss(
            text_features=text_features,
            label_features=label_features,
            logit_scale=logit_scale
        )

        weights = torch.softmax(torch.randn(self._num_tasks), dim=-1)
        clip_loss_weight = weights[0].item() if self._clip_loss_weight is None else self._clip_loss_weight
        bce_loss_weight = weights[0].item() if self._bce_loss_weight is None else self._bce_loss_weight

        return (clip_text_loss * clip_loss_weight) + (bce_text_loss * bce_loss_weight)
