from typing import Optional, Union

from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PretrainedConfig,
)

from models import ICDRobertaForMaskedLM


class ModelHelpers(object):
    """
    A collection of helper functions to load SpaCy or HuggingFace objects/models.
    """

    @staticmethod
    def get_config(
            tokenizer,
            config_name: str,
            config_overrides: Optional[str],
            cache_dir: Optional[str],
            model_revision: str,
            use_auth_token: bool,
            classifier_dropout: float = 0.1
    ) -> PretrainedConfig:

        config = AutoConfig.from_pretrained(
            config_name,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
            classifier_dropout=classifier_dropout,
        )
        # Set the special token IDS in the config based on how the special tokens are defined by the tokenizer
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.sep_token_id = tokenizer.sep_token_id
        config.pad_token_id = tokenizer.pad_token_id
        # Set the vocab size to match the tokenizer vocab_size
        config.vocab_size = tokenizer.vocab_size
        # We use token type ids to distinguish between the current encounter
        # and the other encounters. 1 indicates current encounter and 0 indicates
        # the other encounters
        config.type_vocab_size = 2
        # Override any config information
        if config_overrides is not None:
            config.update_from_string(config_overrides)
        # Add label2id and id2label and num_labels?
        return config

    @staticmethod
    def get_pretrained_config(
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

        # Load the model config
        config = AutoConfig.from_pretrained(
            config_name,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
            classifier_dropout=classifier_dropout,
        )

        # Update the config with any overrides
        if config_overrides is not None:
            config.update_from_string(config_overrides)
        return config

    @staticmethod
    def get_tokenizer(
            tokenizer_path: str,
            name: str,
            truncation_side: str = 'right',
            model_max_length: int = 514,
            cache_dir: Optional[str] = None,
            model_revision: str = 'main',
            use_auth_token: bool = False,
    ) -> PreTrainedTokenizerFast:
        """
        Use the tokenizer present in tokenizer_path and load the tokenizer with the name
        specified by name_or_path. This loads the tokenizer that is present in
        tokenizer_path (file where the tokenizer is stored). Refer to create_tokenizer.py
        for more information on creating the tokenizer and saving the tokenizer to a file.

        Args:
            tokenizer_path (str): The path where the tokenizer is present
            name (str): The name we want to assign to this tokenizer
            truncation_side (str, defaults to `left`): The side that is used for truncation
            model_max_length (int, defaults to `514`): The maximum length of the tokenizer - truncate beyond this
            cache_dir (Optional[str], defaults to `None`): Where do you want to store the pretrained models/data
            downloaded from huggingface.co
            model_revision (str, defaults to `main`): The specific token version to use
            (can be a branch name, tag name or commit id).
            use_auth_token (bool, defaults to `False`): Will use the token generated when running
            `transformers-cli login` (necessary to use this script with private models).

        Returns:
            (PreTrainedTokenizerFast): A PreTrainedTokenizerFast object that can be used to tokenize text

        """
        return PreTrainedTokenizerFast(
            name_or_path=name,
            tokenizer_file=tokenizer_path,
            truncation_side=truncation_side,
            model_max_length=model_max_length,
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
        )

    @staticmethod
    def get_pretrained_tokenizer(
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

        if model_type in {"gpt2", "roberta", "deberta"}:
            return AutoTokenizer.from_pretrained(
                tokenizer_name,
                cache_dir=cache_dir,
                revision=model_revision,
                use_auth_token=True if use_auth_token else None,
                use_fast=use_fast,
                add_prefix_space=True
            )
        else:
            return AutoTokenizer.from_pretrained(
                tokenizer_name,
                cache_dir=cache_dir,
                revision=model_revision,
                use_auth_token=True if use_auth_token else None,
                use_fast=use_fast,
            )

    @staticmethod
    def get_model(
            config: PretrainedConfig,
    ) -> ICDRobertaForMaskedLM:
        """
        Get the HuggingFace model

        Args:
            config (PretrainedConfig): The HuggingFace config object

        Returns:
            (ICDRobertaForMaskedLM): The HuggingFace ICD Roberta model object
        """

        return ICDRobertaForMaskedLM(config=config)

    @staticmethod
    def get_pretrained_model(
            model_name_or_path,
            config: PretrainedConfig,
    ) -> ICDRobertaForMaskedLM:
        """
        Get the HuggingFace model

        Args:
            model_name_or_path (str): The model checkpoint for weights initialization.
            config (PretrainedConfig): The HuggingFace config object

        Returns:
            (ICDRobertaForMaskedLM): The HuggingFace ICD Roberta model object
        """

        return ICDRobertaForMaskedLM.from_pretrained(model_name_or_path, config=config)
