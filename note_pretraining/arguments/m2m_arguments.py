from dataclasses import dataclass, field
from typing import Optional


@dataclass
class M2MArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    loss_function: str = field(
        default=None, metadata={"help": "The loss function to use for training"}
    )
    lm_objective: Optional[str] = field(
        default=None, metadata={"help": "Whether to use a language modeling objective as well"}
    )
    gensim_model: Optional[str] = field(
        default=None, metadata={"help": "The w2v icd code model"},
    )
    text_projection: int = field(
        default=None, metadata={"help": "The projection size of the text model"},
    )
    label_projection: int = field(
        default=None, metadata={"help": "The projection size of the label model"},
    )
    multi_label: bool = field(
        default=False, metadata={"help": "Flag to use multi label classification"},
    )
    alpha: Optional[float] = field(
        default=None, metadata={"help": "The alpha parameter for label smoothing. Range: [0, 1]"},
    )
    beta: Optional[float] = field(
        default=None, metadata={"help": "The beta parameter for label smoothing."},
    )
    distance_metric: Optional[str] = field(
        default=None,
        metadata={"help": "The distance metric for label smoothing. Possible values are cosine and euclidean"},
    )
    reduction: Optional[str] = field(
        default=None, metadata={"help": "The reduction for label smoothing. Possible values are min and mean"},
    )
    loss_weighting: Optional[str] = field(
        default=None, metadata={"help": "Method to weight the CLIP and LM loss"},
    )
    load_state_dict: bool = field(
        default=None, metadata={"help": "Whether to load a pre-trained state dict"},
    )
