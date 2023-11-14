# We need a custom data collator to handle ICD-10 sequences and position ID's
# This class uses the existing Huggingface data collator and adds support to handle
# the collation of custom position id's.

import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional

from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorMixin,
)


@dataclass
class ICDDataCollator:
    """
    Data collator
    Args:

    """
    tokenizer: PreTrainedTokenizerBase
    data_collator: Union[DataCollatorMixin, DataCollatorWithPadding]
    position_ids_name: str = 'position_ids'

    def __call__(self, features: List[Union[List[int], Any, Dict[str, Any]]]):

        # Collated position ids
        position_ids = [
            feature.pop(self.position_ids_name) for feature in features
        ] if self.position_ids_name in features[0].keys() else None

        # Process the text portion - Collated text/tokens
        batch = self.data_collator(features)

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        # Add the position ids to the batch
        if position_ids is not None:
            batch[self.position_ids_name] = self.get_collated_special_ids(
                special_ids=position_ids,
                padding_side=padding_side,
                sequence_length=sequence_length,
            )

        # Return collated batch
        return batch

    @staticmethod
    def get_collated_special_ids(special_ids, padding_side, sequence_length):
        if padding_side == "right":
            return torch.tensor([
                list(special_id) + [0] * (sequence_length - len(special_id)) for special_id in special_ids
            ])
        else:
            return torch.tensor([
                [0] * (sequence_length - len(special_id)) + list(special_id) for special_id in special_ids
            ])

@dataclass
class ICDDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all the same length. These subclasses the DataCollatorForLanguageModeling. Changes are made
    to the portion when mlm is set to False. Check if labels are passed along with the batch, and if they are use
    those labels instead of cloning the input ids. This is done to perform masked language model scoring during,
    validation step - refer this paper: Masked Language Model Scoring (https://arxiv.org/pdf/1910.14659.pdf).
    The mlm functions are used as in during training and there is now there are three other options.
    Use masked labels, use input ids as labels or pass the labels. The default values - 0.8 and 0.1 are now
    replaced as variables, hence the masking rate can be controlled.
    Args:
    <Tip>
    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.
    </Tip>"""
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True,
    mlm_probability: float = 0.20
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=self.mlm,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=self.pad_to_multiple_of,
            tf_experimental_compile=self.tf_experimental_compile,
            return_tensors=self.return_tensors
        )
        self.icd_data_collator = ICDDataCollator(
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.icd_data_collator(features)
