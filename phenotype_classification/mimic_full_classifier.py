import json
import logging
from typing import Union, Optional, Sequence

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PretrainedConfig
)

from classification_datasets.text_tokenizers import SpacyTokenizer
from classification_datasets.text_transformations import (
    TextTransform,
    DynamicPromptLabelTransform,
    DynamicPromptClassificationTransformation,
)
from metrics import MimicFullClassificationMetrics, MultiLabelEvaluation
from prompt_text_classifier import PromptTextClassifier

logger = logging.getLogger(__name__)
HF_DATASETS_CACHE = '/mnt/obi0/phi/ehr_projects/phenotype_classification/cache/huggingface/datasets/'


class MimicFullClassifier(PromptTextClassifier):
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
            validation_file,
            icd_codes,
            pad_to_max_length: bool = False,
            max_seq_length: Optional[int] = None,
            is_split_into_words: bool = False,
            text_column_name: str = 'text',
            label_column_name: str = 'labels',
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
        self._validation_file = validation_file
        self._icd_codes = icd_codes
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
            optimal_f1_threshold=optimal_f1_threshold,
            prompt_label_list=[]
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

        label_transform = DynamicPromptLabelTransform(
            mask_token_id=self._subword_tokenizer.mask_token_id
        )

        return DynamicPromptClassificationTransformation(
            text_transform=text_transform,
            label_transform=label_transform,
            text_column_name=text_column_name,
            label_column_name=label_column_name
        )

    def get_metrics(self):
        data_size = 0
        group_size = 6
        note_labels = list()
        prompt_labels = list()

        for line in open(self._validation_file):
            data_size += 1
            note = json.loads(line)
            prompt_labels.append(note['prompt_labels'])
            note_labels.append(note['note_labels'])

        prompt_labels = [
            ' '.join(prompt_labels[i:i + group_size]).split() for i in range(0, len(prompt_labels), group_size)
        ]
        note_labels = [note_labels[i].split() for i in range(0, len(note_labels), group_size)]

        for prompt_label in prompt_labels:
            if len(prompt_label) != 300:
                raise ValueError()

        if len(prompt_labels) != (data_size // group_size) or len(note_labels) != (data_size // group_size):
            raise ValueError()

        classification_metrics = MimicFullClassificationMetrics(
            post_process=self._post_process,
            label_list=self._icd_codes,
            evaluator=MultiLabelEvaluation(average_options=('micro', 'macro', 'weighted')),
            optimal_f1_threshold=self._optimal_threshold,
            prompt_labels=prompt_labels,
            note_labels=note_labels,
            group_size=group_size,
        )
        return classification_metrics.compute_multi_label_metrics
