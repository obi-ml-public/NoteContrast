import logging
from typing import Union, Optional, Sequence

from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast, PretrainedConfig, default_data_collator, DataCollatorWithPadding
)

from classification_datasets.labels import LabelInfo
from classification_datasets.text_tokenizers import SpacyTokenizer
from classification_datasets.text_transformations import (
    TextTransform,
    PromptLabelTransform,
    PromptClassificationTransformation
)
from metrics import ClassificationMetrics, MultiLabelEvaluation
from model_outputs import PromptArgmaxProcess
from classifier import Classifier

logger = logging.getLogger(__name__)
HF_DATASETS_CACHE = '/mnt/obi0/phi/ehr_projects/phenotype_classification/cache/huggingface/datasets/'


class PromptTextClassifier(Classifier):
    """
    The SequenceTagger class can be used to train sequence tagging models, evaluate models, and
    run predictions. The class can also be used to de-identify and augment text.
    TODO: Add a more detailed description
    """

    def __init__(
            self,
            text_tokenizer: Optional[SpacyTokenizer],
            config: PretrainedConfig,
            subword_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            model: PreTrainedModel,
            label_list: Sequence[str],
            ignore_labels: Sequence[str],
            prompt_label_list: Optional[Sequence[str]],
            fp16: bool,
            pad_to_max_length: bool = False,
            max_seq_length: Optional[int] = None,
            is_split_into_words: bool = False,
            text_column_name: str = 'text',
            label_column_name: str = 'label',
            seed: int = 41,
            classification_threshold=None,
            optimal_f1_threshold=None
    ):
        """
        Initialize the variables and load the relevant objects.

        Args:
            text_tokenizer (SpacyTokenizer): The SpacyTokenizer object for word tokenization of text.
            config (PretrainedConfig): The HuggingFace model config object.
            subword_tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The Huggingface tokenizer object.
            model (PreTrainedModel): The HuggingFace model object.
            label_list (Sequence[str]):
            fp16 (bool): Whether to use mixed-precision.
            pad_to_max_length (bool, defaults to `True`): Whether to pad all samples to `max_seq_length`. If False,
            will pad the samples dynamically when batching to the maximum length in the batch.
            max_seq_length (Optional[int], defaults to `None`): The maximum total input sequence length after
            tokenization. Sequences longer than this will be truncated.
            seed (int, defaults to `41): Reproducible seed.
        """

        self._prompt_label_list = prompt_label_list
        self._prompt_label_info = LabelInfo(label_list=self._prompt_label_list)

        super().__init__(
            text_tokenizer=text_tokenizer,
            config=config,
            subword_tokenizer=subword_tokenizer,
            model=model,
            label_list=label_list,
            ignore_labels=ignore_labels,
            fp16=fp16,
            pad_to_max_length=pad_to_max_length,
            max_seq_length=max_seq_length,
            is_split_into_words=is_split_into_words,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            seed=seed,
            classification_threshold=classification_threshold,
            optimal_threshold=optimal_f1_threshold
        )

    def _check_multi_class(self):
        return False

    def _check_multi_label(self):
        return False

    def get_post_process(self):
        return PromptArgmaxProcess(
            label_list=self._label_list,
            ignore_label=-100,
            threshold=self._classification_threshold
        )

    def get_data_collator(
            self,
            pad_to_max_length,
            fp16: bool
    ) -> Optional[Union[default_data_collator, DataCollatorWithPadding, DataCollatorForTokenClassification]]:
        """
        Get the data collator object.

        Args:
            pad_to_max_length (bool, defaults to `True`): Whether to pad all samples to `max_seq_length`. If False,
            will pad the samples dynamically when batching to the maximum length in the batch.
            fp16 (bool): Whether to use mixed precision training.

        Returns:
            (DataCollatorForTokenClassification): The data collator object.
        """
        return DataCollatorForTokenClassification(
            self._subword_tokenizer,
            pad_to_multiple_of=8 if fp16 else None
        )

    def get_classification_transform(
            self,
            pad_to_max_length,
            max_seq_length,
            text_column_name,
            label_column_name,
            is_split_into_words
    ):

        text_transform = TextTransform(
            subword_tokenizer=self._subword_tokenizer,
            text_tokenizer=self._text_tokenizer,
            truncation=True,
            pad_to_max_length=pad_to_max_length,
            max_seq_length=max_seq_length,
            is_split_into_words=is_split_into_words,
        )

        label_transform = PromptLabelTransform(
            label_to_id=self._prompt_label_info.get_label_to_id(),
            mask_token_id=self._subword_tokenizer.mask_token_id
        )

        return PromptClassificationTransformation(
            text_transform=text_transform,
            label_transform=label_transform,
            text_column_name=text_column_name,
            label_column_name=label_column_name
        )

    def get_metrics(self):
        classification_metrics = ClassificationMetrics(
            post_process=self._post_process,
            label_list=self._prompt_label_list,
            evaluator=MultiLabelEvaluation(),
            optimal_f1_threshold=self._optimal_threshold
        )
        return classification_metrics.compute_multi_label_metrics
