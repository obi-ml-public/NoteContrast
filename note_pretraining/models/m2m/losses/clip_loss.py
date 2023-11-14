import torch
from torch import nn

class CLIPLoss(nn.Module):

    def __init__(self, logit_scale):
        super().__init__()
        self._logit_scale = logit_scale

    @staticmethod
    def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        text_loss = self.contrastive_loss(similarity)
        label_loss = self.contrastive_loss(similarity.t())
        return (text_loss + label_loss) / 2.0

    def forward(self, text_features, label_features):
        # Normalized features
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        label_features = label_features / label_features.norm(p=2, dim=-1, keepdim=True)

        if self._logit_scale is not None:
            logits_per_text = self._logit_scale(torch.matmul(text_features, label_features.t()))
        else:
            logits_per_text = torch.matmul(text_features, label_features.t())

        return self.clip_loss(logits_per_text)

    def get_logit_scale(self):
        return self._logit_scale.get_value()
