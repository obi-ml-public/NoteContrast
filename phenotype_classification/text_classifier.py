import logging
from typing import Union, Optional, Sequence

from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast, PretrainedConfig, default_data_collator, DataCollatorWithPadding
)

from classification_datasets.text_tokenizers import SpacyTokenizer
from classification_datasets.text_transformations import (
    TextTransform,
    LabelTransform,
    ClassificationTransformation,
    MultiLabelTransform,
)
from metrics import ClassificationMetrics, MultiClassEvaluation, BinaryClassEvaluation
from model_outputs import BinaryProcess, MultiLabelProcess, MultiClassProcess
from classifier import Classifier

logger = logging.getLogger(__name__)
HF_DATASETS_CACHE = '/mnt/obi0/phi/ehr_projects/phenotype_classification/cache/huggingface/datasets/'


class TextClassifier(Classifier):
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
        # Check if multi label
        if self._problem_type == 'multi_label_classification':
            return False
        elif self._problem_type == 'single_label_classification':
            self._multi_label = False
            if len(self._label_list) > 2:
                return True
            elif len(self._label_list) == 2:
                if '0' in self._label_list or 0 in self._label_list:
                    return False
                else:
                    return True
            else:
                return False
        else:
            raise ValueError('Invalid problem type')

    def _check_multi_label(self):
        if self._problem_type == 'multi_label_classification':
            return True
        else:
            return False

    def get_post_process(self):
        if self._multi_label:
            return MultiLabelProcess(label_list=self._label_list, threshold=self._classification_threshold)
        else:
            if self._multi_class:
                return MultiClassProcess(label_list=self._label_list, threshold=self._classification_threshold)
            else:
                return BinaryProcess(label_list=self._label_list, threshold=self._classification_threshold)

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

        # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer,
        # so we change it if we already did the padding.
        if pad_to_max_length:
            return default_data_collator
        elif fp16:
            return DataCollatorWithPadding(self._subword_tokenizer, pad_to_multiple_of=8)
        else:
            return None

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
        if self._multi_label:
            label_transform = MultiLabelTransform(label_to_id=self.label_to_id)
        else:
            label_transform = LabelTransform(label_to_id=self.label_to_id)

        return ClassificationTransformation(
            text_transform=text_transform,
            label_transform=label_transform,
            text_column_name=text_column_name,
            label_column_name=label_column_name
        )

    def get_metrics(self):
        if self._multi_label:
            evaluator = None
        else:
            if self._multi_class:
                evaluator = MultiClassEvaluation()
            else:
                evaluator = BinaryClassEvaluation()

        classification_metrics = ClassificationMetrics(
            post_process=self._post_process,
            label_list=self._label_list,
            evaluator=evaluator,
            optimal_f1_threshold=self._optimal_threshold
        )
        if self._multi_label:
            return classification_metrics.compute_multi_label_metrics
        else:
            if self._multi_class:
                return classification_metrics.compute_multi_class_metrics
            else:
                return classification_metrics.compute_binary_metrics
