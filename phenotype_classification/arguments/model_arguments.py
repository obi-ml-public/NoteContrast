from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
                    "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    spacy_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "The spacy model to use for tokenization"
        },
    )
    problem_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Problem type for XxxForSequenceClassification models. Can be one of 'regression', "
                    "'single_label_classification' or 'multi_label_classification'. "
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    use_probing_classifier: bool = field(
        default=False,
        metadata={
            "help": "Whether to run experiments as a probing classifier"
        },
    )
    use_prompting_classifier: bool = field(
        default=False,
        metadata={
            "help": "Whether to run experiments as a prompt fine-tuned classifier"
        },
    )

    classification_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": ""
        },
    )
    optimal_f1_threshold: Optional[bool] = field(
        default=None,
        metadata={
            "help": ""
        },
    )
