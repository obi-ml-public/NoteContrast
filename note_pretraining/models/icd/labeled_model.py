from dataclasses import dataclass

import torch
from transformers.file_utils import ModelOutput
from typing import Optional


@dataclass
class LabeledModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
    """
    last_hidden_state: Optional[torch.FloatTensor] = None


class LabeledModel(object):

    def __call__(self, embedding):
        # Un-squeeze and return because the M2M framework expects the label model
        # to have an output of the form (batch_size, sequence_len, hidden_size)
        # Since fixed labels are of the shape (batch_size, hidden_size)
        # we un-squeeze to return a vector of shape (batch_size, 1, hidden_size)
        # so that the M2M framework can then select the 0th position in the sequence

        return LabeledModelOutput(
            last_hidden_state=embedding.unsqueeze(dim=1)
        )
