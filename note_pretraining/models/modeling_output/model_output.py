from dataclasses import dataclass

import torch
from transformers.file_utils import ModelOutput
from typing import Optional


@dataclass
class M2MModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
    """
    loss: Optional[torch.FloatTensor] = None
    text_features: Optional[torch.FloatTensor] = None
    label_features: Optional[torch.FloatTensor] = None
    logit_scale: Optional[torch.FloatTensor] = None


@dataclass
class M2MLMModelOutput(M2MModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
    """
    text_lm_loss: Optional[torch.FloatTensor] = None
    m2m_loss: Optional[torch.FloatTensor] = None
    text_logits: Optional[torch.FloatTensor] = None
    label_logits: Optional[torch.FloatTensor] = None
