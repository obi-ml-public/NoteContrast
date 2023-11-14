from torch import nn
from torch.nn import BCEWithLogitsLoss

class BCELoss(nn.Module):

    def __int__(self):
        super().__init__()

    def forward(self, text_features, label_features, logit_scale):
        loss_fct = BCEWithLogitsLoss()
        return loss_fct(input=text_features, target=label_features)  * (logit_scale / logit_scale)
