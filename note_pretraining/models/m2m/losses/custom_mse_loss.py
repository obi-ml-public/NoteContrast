from torch import nn
from torch.nn import MSELoss


class CustomMSELoss(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, text_features, label_features, logit_scale):
        loss_fct = MSELoss()
        if self.num_labels == 1:
            return loss_fct(text_features.squeeze(), label_features.squeeze()) * (logit_scale / logit_scale)
        else:
            return loss_fct(text_features, label_features) * (logit_scale / logit_scale)
