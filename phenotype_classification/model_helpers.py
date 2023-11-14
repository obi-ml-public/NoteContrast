from typing import List, Optional, Union

import spacy
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
    AutoModel,
    AutoModelForTokenClassification
)

from classification_datasets.labels import LabelInfo
from classification_datasets.text_tokenizers import SpacyTokenizer


class ModelHelpers(object):
    """
    A collection of helper functions to load SpaCy or HuggingFace objects/models.
    """

    @staticmethod
    def get_config(
            label_list: List[str],
            problem_type: str,
            config_name: str,
            config_overrides: Optional[str],
            cache_dir: Optional[str],
            model_revision: str,
            use_auth_token: bool,
            classifier_dropout: float = 0.1
    ) -> PretrainedConfig:
        """
        Get the HuggingFace config object.

        Args:
            label_list (List[str]): The list of labels.
            problem_type (str): Binary/multi-label/multi-class classification or regression
            config_name (str): The name of the config to be loaded.
            config_overrides(Optional[str]): Override some existing default config settings when a model
            is trained from scratch.
            cache_dir (Optional[str]): Where do you want to store the pretrained models downloaded from huggingface.co.
            model_revision (str): The specific model version to use (can be a branch name, tag name or commit id).
            use_auth_token (bool): Will use the token generated when running `transformers-cli login`
            (necessary to use this script with private models).
            classifier_dropout (float, default to `0.1`): The percentage of dropout for the linear classification layer.

        Returns:
            config (PretrainedConfig): The HuggingFace config object.
        """

        # This object will have label information
        if label_list is not None:
            label_info = LabelInfo(label_list=label_list)
        else:
            label_info = None

        # Load the model config
        config = AutoConfig.from_pretrained(
            config_name,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
            classifier_dropout=classifier_dropout,
            problem_type=problem_type,
        )

        # Update the config with any overrides
        if config_overrides is not None:
            config.update_from_string(config_overrides)
        if label_info is not None:
            config.num_labels = len(label_info.get_label_list())

        return config

    @staticmethod
    def get_tokenizer(
            tokenizer_name: str,
            model_type: str,
            cache_dir: Optional[str],
            model_revision: str,
            use_auth_token: bool,
            use_fast: bool = True
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """
        Get the HuggingFace tokenizer object.

        Args:
            tokenizer_name (str): The name of the tokenizer to be loaded.
            model_type (str): The model type - whether to add prefix space or not.
            cache_dir (Optional[str]): Where do you want to store the pretrained models downloaded from huggingface.co.
            model_revision (str): The specific model version to use (can be a branch name, tag name or commit id).
            use_auth_token (bool): Will use the token generated when running `transformers-cli login`
            (necessary to use this script with private models).
            use_fast (bool, defaults to `True`): Whether to load PreTrainedTokenizerFast.

        Returns:
            (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The HuggingFace tokenizer object.
        """
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
            use_fast=use_fast,
        )

    @staticmethod
    def get_model(
            model_name_or_path: str,
            from_tf: bool,
            config: PretrainedConfig,
            cache_dir: Optional[str],
            model_revision: str,
            use_auth_token: bool
    ) -> PreTrainedModel:
        """
        Get the HuggingFace model

        Args:
            model_name_or_path (str): The model checkpoint for weights initialization.
            from_tf (bool): Whether to load the tensorflow checkpoint
            config (PretrainedConfig): The HuggingFace config object
            cache_dir (Optional[str]): Where do you want to store the pretrained models downloaded from huggingface.co.
            model_revision (str): The specific model version to use (can be a branch name, tag name or commit id).
            use_auth_token (bool): Will use the token generated when running `transformers-cli login`
            (necessary to use this script with private models).

        Returns:
            (PreTrainedModel): The HuggingFace model object
        """

        return AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
        )

    @staticmethod
    def get_prompting_model(
            model_name_or_path: str,
            from_tf: bool,
            config: PretrainedConfig,
            cache_dir: Optional[str],
            model_revision: str,
            use_auth_token: bool
    ) -> PreTrainedModel:
        """
        Get the HuggingFace model

        Args:
            model_name_or_path (str): The model checkpoint for weights initialization.
            from_tf (bool): Whether to load the tensorflow checkpoint
            config (PretrainedConfig): The HuggingFace config object
            cache_dir (Optional[str]): Where do you want to store the pretrained models downloaded from huggingface.co.
            model_revision (str): The specific model version to use (can be a branch name, tag name or commit id).
            use_auth_token (bool): Will use the token generated when running `transformers-cli login`
            (necessary to use this script with private models).

        Returns:
            (PreTrainedModel): The HuggingFace model object
        """

        return AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
        )

    @staticmethod
    def get_embedding_model(
            model_name_or_path: str,
            from_tf: bool,
            config: PretrainedConfig,
            cache_dir: Optional[str],
            model_revision: str,
            use_auth_token: bool
    ) -> PreTrainedModel:
        """
        Get the HuggingFace model

        Args:
            model_name_or_path (str): The model checkpoint for weights initialization.
            from_tf (bool): Whether to load the tensorflow checkpoint
            config (PretrainedConfig): The HuggingFace config object
            cache_dir (Optional[str]): Where do you want to store the pretrained models downloaded from huggingface.co.
            model_revision (str): The specific model version to use (can be a branch name, tag name or commit id).
            use_auth_token (bool): Will use the token generated when running `transformers-cli login`
            (necessary to use this script with private models).

        Returns:
            (PreTrainedModel): The HuggingFace model object
        """

        return AutoModel.from_pretrained(
            model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
        )

    @staticmethod
    def get_text_tokenizer(spacy_model: str) -> SpacyTokenizer:
        """
        Get the SpacyTokenizer object. This is the spacy tokenizer wrapped with some helper functions

        Args:
            spacy_model (str): The spacy model to use

        Returns:
            (SpacyTokenizer): The object
        """

        # Create the spacy tokenizer
        nlp = spacy.load(spacy_model)
        return SpacyTokenizer(nlp=nlp)
