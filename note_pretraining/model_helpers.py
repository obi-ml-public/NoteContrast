from typing import Optional, Union

from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM
)

from models.icd import ICDRobertaModel, LabeledModel


class ModelHelpers(object):
    """
    A collection of helper functions to load SpaCy or HuggingFace objects/models.
    """

    @staticmethod
    def get_config(
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

    def get_icd_config(
            self,
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
        config.type_vocab_size = 2
        # Override any config information
        if config_overrides is not None:
            config.update_from_string(config_overrides)
        # Add label2id and id2label and num_labels?
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
    def get_text_model(
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
    def get_icd_model(
            config: PretrainedConfig,
    ) -> ICDRobertaModel:
        """
        Get the HuggingFace model

        Args:
            config (PretrainedConfig): The HuggingFace config object

        Returns:
            (ICDRobertaModel): The HuggingFace ICD Roberta model object
        """

        return ICDRobertaModel(config=config)

    @staticmethod
    def get_icd_pretrained_model(
            model_name_or_path,
            config: PretrainedConfig,
    ) -> ICDRobertaModel:
        """
        Get the HuggingFace model

        Args:
            model_name_or_path (str): The model checkpoint for weights initialization.
            config (PretrainedConfig): The HuggingFace config object

        Returns:
            (ICDRobertaModel): The HuggingFace ICD Roberta model object
        """

        return ICDRobertaModel.from_pretrained(model_name_or_path, config=config)

    @staticmethod
    def get_clm_model(model_name_or_path, config, cache_dir, model_revision, use_auth_token):
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
        )

    @staticmethod
    def get_mlm_model(model_name_or_path, config, cache_dir, model_revision, use_auth_token):
        return AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
        )

    @staticmethod
    def get_clm_head(model_name_or_path, config, cache_dir, model_revision, use_auth_token):
        if config.model_type == 'bigbird':
            return ModelHelpers.get_clm_model(
                model_name_or_path,
                config,
                cache_dir,
                model_revision,
                use_auth_token
            ).cls
        elif config.model_type in ['roberta', 'longformer']:
            return ModelHelpers.get_clm_model(
                model_name_or_path,
                config,
                cache_dir,
                model_revision,
                use_auth_token
            ).lm_head
        else:
            raise NotImplementedError()

    @staticmethod
    def get_mlm_head(model_name_or_path, config, cache_dir, model_revision, use_auth_token):
        if config.model_type in ['bigbird', 'big_bird']:
            return ModelHelpers.get_mlm_model(
                model_name_or_path,
                config,
                cache_dir,
                model_revision,
                use_auth_token
            ).cls
        elif config.model_type in ['roberta', 'longformer']:
            return ModelHelpers.get_mlm_model(
                model_name_or_path,
                config,
                cache_dir,
                model_revision,
                use_auth_token
            ).lm_head
        else:
            raise NotImplementedError()

    @staticmethod
    def get_labeled_model():
        return LabeledModel()
