from torch import nn
from transformers.activations import ACT2FN
from .utils import get_model_embedding

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, projection, token_from_end):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, projection)
        self.config = config
        self.token_from_end = token_from_end

    def forward(self, features, attention_mask=None, **kwargs):
        x = get_model_embedding(features=features, attention_mask=attention_mask, token_from_end=self.token_from_end)
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
