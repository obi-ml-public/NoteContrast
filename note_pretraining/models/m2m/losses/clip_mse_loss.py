import torch
import torch.distributed.nn
from torch import nn as nn


class CLIPMSELoss(nn.Module):
    def __init__(self, clip_loss, mse_loss, clip_loss_weight=None, mse_loss_weight=None):
        super().__init__()
        self._clip_loss = clip_loss
        self._mse_loss = mse_loss
        self._clip_loss_weight = clip_loss_weight
        self._mse_loss_weight = mse_loss_weight
        self._num_tasks = 2

    def forward(self, text_features, label_features, logit_scale):
        clip_text_loss = self._clip_loss(
            text_features=text_features,
            label_features=label_features,
            logit_scale=logit_scale
        )
        mse_text_loss = self._mse_loss(
            text_features=text_features,
            label_features=label_features,
            logit_scale=logit_scale
        )

        weights = torch.softmax(torch.randn(self._num_tasks), dim=-1)
        clip_loss_weight = weights[0].item() if self._clip_loss_weight is None else self._clip_loss_weight
        mse_loss_weight = weights[0].item() if self._mse_loss_weight is None else self._mse_loss_weight

        return (clip_text_loss * clip_loss_weight) + (mse_text_loss * mse_loss_weight)
