from torch import nn
from .utils import get_model_embedding

class ProjectionHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, projection, token_from_end):
        super().__init__()
        self.config = config
        self.token_from_end = token_from_end
        self.final_layer_norm = nn.LayerNorm(self.config.hidden_size)
        self.projection = nn.Linear(config.hidden_size, projection, bias=False)

    def forward(self, features, attention_mask=None, **kwargs):
        x = get_model_embedding(features=features, attention_mask=attention_mask, token_from_end=self.token_from_end)
        x = self.final_layer_norm(x)
        x = self.projection(x)
        return x

    def init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                std=self.config.hidden_size ** -0.5,
            )
