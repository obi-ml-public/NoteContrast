import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional

from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from transformers.data.data_collator import (
    torch_default_data_collator,
    tf_default_data_collator,
    numpy_default_data_collator,
    InputDataClass,
    DataCollatorForLanguageModeling,
    DataCollatorMixin,
)
from transformers.file_utils import PaddingStrategy


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

        # Collated labels
        position_ids = [
            feature.pop(self.position_ids_name) for feature in features
        ] if self.position_ids_name in features[0].keys() else None

        # Process the text portion - Collated text/tokens
        batch = self.data_collator(features)

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

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
            # This could also be a long tensor depending on the position ids
            # but since we are using ages we used float tensor here, but we should
            # have a better way of doing this and not hard coding float
            # same thing for all the operations below in tf and np
            return torch.tensor([
                list(special_id) + [0] * (sequence_length - len(special_id)) for special_id in special_ids
            ])
        else:
            return torch.tensor([
                [0] * (sequence_length - len(special_id)) + list(special_id) for special_id in special_ids
            ])


def m2m_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.

    # Extract nested features in the desired format
    nested_features = {key: [i[key] for i in features] for key in features[0]}

    if return_tensors == "pt":
        return {key: torch_default_data_collator(nested_feature) for key, nested_feature in nested_features.items()}
    elif return_tensors == "tf":
        return {key: tf_default_data_collator(nested_feature) for key, nested_feature in nested_features.items()}
    elif return_tensors == "np":
        return {key: numpy_default_data_collator(nested_feature) for key, nested_feature in nested_features.items()}


@dataclass
class M2MDataCollator:
    """
    Data collator
    Args:

    """
    text_data_collator: Union[DataCollatorMixin, DataCollatorWithPadding]
    label_data_collator: Union[
        DataCollatorMixin,
        DataCollatorWithPadding,
        tf_default_data_collator,
        numpy_default_data_collator,
        torch_default_data_collator
    ]
    label_tokenizer: PreTrainedTokenizerBase
    position_ids_name: str = 'position_ids'
    token_type_ids_name: str = 'token_type_ids'

    def __call__(self, features: List[Union[List[int], Any, Dict[str, Any]]]):

        # Extract nested features in the desired format
        nested_features = {key: [i[key] for i in features] for key in features[0]}
        # Store collated batch
        nested_batch = dict()
        # Process the text portion - Collated text/tokens
        batch = self.text_data_collator(nested_features['m2m_texts'])
        # If special token mask has been preprocessed, pop it from the dict.
        batch.pop("special_tokens_mask", None)
        # Copy of batch - not sure why, but, it was not working when we used the batch object directly
        nested_batch['m2m_texts'] = {key: value for key, value in batch.items()}

        # Collated labels
        position_ids = [
            feature.pop(self.position_ids_name) for feature in nested_features['m2m_labels']
        ] if self.position_ids_name in nested_features['m2m_labels'][0].keys() else None

        m2m_labels = self.label_data_collator(nested_features['m2m_labels'])
        sequence_length = m2m_labels["input_ids"].shape[1]
        padding_side = self.label_tokenizer.padding_side

        m2m_labels = self.label_data_collator(nested_features['m2m_labels'])

        if position_ids is not None:
            m2m_labels[self.position_ids_name] = self.get_collated_special_ids(
                special_ids=position_ids,
                padding_side=padding_side,
                sequence_length=sequence_length
            )

        # Copy of batch - not sure why, but, it was not working when we used the batch object directly
        nested_batch['m2m_labels'] = {key: value for key, value in m2m_labels.items()}

        # Return collated batch
        return nested_batch

    @staticmethod
    def get_collated_special_ids(special_ids, padding_side, sequence_length):
        if padding_side == "right":
            # This could also be a long tensor depending on the position ids
            # but since we are using ages we used float tensor here, but we should
            # have a better way of doing this and not hard coding float
            # same thing for all the operations below in tf and np
            return torch.tensor([
                list(special_id) + [0] * (sequence_length - len(special_id)) for special_id in special_ids
            ])
        else:
            return torch.tensor([
                [0] * (sequence_length - len(special_id)) + list(special_id) for special_id in special_ids
            ])




@dataclass
class M2MDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    text_tokenizer: PreTrainedTokenizerBase
    label_tokenizer: Optional[PreTrainedTokenizerBase] = None
    text_padding: Union[bool, str, PaddingStrategy] = True
    text_max_length: Optional[int] = None
    label_padding: Union[bool, str, PaddingStrategy] = True
    label_max_length: Optional[int] = None
    text_pad_to_multiple_of: Optional[int] = None
    label_pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        text_data_collator = DataCollatorWithPadding(
            tokenizer=self.text_tokenizer,
            padding=self.text_padding,
            max_length=self.text_max_length,
            pad_to_multiple_of=self.text_pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        if self.label_tokenizer is not None:
            label_data_collator = DataCollatorWithPadding(
                tokenizer=self.label_tokenizer,
                padding=self.label_padding,
                max_length=self.label_max_length,
                pad_to_multiple_of=self.label_pad_to_multiple_of,
                return_tensors=self.return_tensors
            )
        else:
            if self.return_tensors == "pt":
                label_data_collator = torch_default_data_collator
            elif self.return_tensors == 'tf':
                label_data_collator = tf_default_data_collator
            elif self.return_tensors == 'np':
                label_data_collator = numpy_default_data_collator
            else:
                raise ValueError('Invalid tensor type')
        self.m2m_data_collator = M2MDataCollator(
            text_data_collator=text_data_collator,
            label_data_collator=label_data_collator,
            label_tokenizer=self.label_tokenizer
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.m2m_data_collator(features)


@dataclass
class M2MDataCollatorForLanguageModeling:
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
    text_tokenizer: PreTrainedTokenizerBase
    label_tokenizer: Optional[PreTrainedTokenizerBase] = None
    text_mlm: bool = True,
    text_mlm_probability: float = 0.15
    text_pad_to_multiple_of: Optional[int] = None
    label_padding: Union[bool, str, PaddingStrategy] = True
    label_max_length: Optional[int] = None
    label_pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):

        text_data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.text_tokenizer,
            mlm=self.text_mlm,
            mlm_probability=self.text_mlm_probability,
            pad_to_multiple_of=self.text_pad_to_multiple_of,
            tf_experimental_compile=self.tf_experimental_compile,
            return_tensors=self.return_tensors
        )
        if self.label_tokenizer is not None:
            label_data_collator = DataCollatorWithPadding(
                tokenizer=self.label_tokenizer,
                pad_to_multiple_of=self.label_pad_to_multiple_of,
                return_tensors=self.return_tensors,
                padding=self.label_padding,
                max_length=self.label_max_length
            )
        else:
            if self.return_tensors == "pt":
                label_data_collator = torch_default_data_collator
            elif self.return_tensors == 'tf':
                label_data_collator = tf_default_data_collator
            elif self.return_tensors == 'np':
                label_data_collator = numpy_default_data_collator
            else:
                raise ValueError('Invalid tensor type')
        self.m2m_data_collator = M2MDataCollator(
            text_data_collator=text_data_collator,
            label_data_collator=label_data_collator,
            label_tokenizer=self.label_tokenizer
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.m2m_data_collator(features)
