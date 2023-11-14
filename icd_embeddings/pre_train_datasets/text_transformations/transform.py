import logging
from typing import Optional, Union, Sequence, List

from transformers import PreTrainedTokenizerFast


class Transform(object):
    """
    Class to handle dataset loading (from files and loading to dataset object) and dataset tokenization
    """

    def __init__(
            self,
            tokenizer,
            truncation: bool,
            pad_to_max_length: bool,
            max_seq_length: Optional[int],
            is_split_into_words: bool = False,
            special_token_position_ids: Optional[float] = None
    ):
        """
        Initialize the tokenizer and other tokenization attributes

        Args:
            tokenizer (PreTrainedTokenizerFast): Tokenizer object used to tokenize the input text
            truncation (bool, defaults to `True`): Truncate the text to the model max length
            pad_to_max_length (bool, defaults to `False`): Whether to pad all samples to `max_seq_length`.
            If False, will pad the samples dynamically when batching to the maximum length in the batch.
            max_seq_length (Optional[int], defaults to `514`): The maximum total input sequence length after
            tokenization. Sequences longer than this will be truncated.
            is_split_into_words (bool, defaults to `False`): Whether the input text has already been pre-tokenized
            special_token_position_ids (Optional[int], defaults to `None`): Support for adding custom special_token_position_ids

        """
        self._tokenizer = tokenizer
        self._truncation = truncation
        self._padding = "max_length" if pad_to_max_length else False
        self._pad_to_max_length = pad_to_max_length
        self._max_seq_length = self._get_max_seq_length(tokenizer=tokenizer, max_seq_length=max_seq_length)
        self._is_split_into_words = is_split_into_words
        self._special_token_position_ids = special_token_position_ids

    @staticmethod
    def pre_tokenize(
            texts: Union[List[List[str]], List[str]]
    ) -> Union[List[List[str]], List[str]]:
        """
        Apply any pre-tokenization or transformations to the input sequence before running
        subword tokenization.

        Args:
            texts (Union[List[List[str]], List[str]]): The batch of input text

        Returns:
            texts (Union[List[List[str]], List[str]]): The transformed batch of input text
        """
        return texts

    def __get_special_ids(self, special_ids):

        # Split the input string by whitespace
        # If it is already a split list, leave as is
        if type(special_ids) == str:
            special_ids = special_ids.split()

        # Convert the str to float.
        # Since we truncate our input text based on the max_seq_length, we also need to
        # truncate the position_ids. Our tokenizer handles the truncation of the input text,
        # but we need to manually handle the truncation of the position ids
        if self._tokenizer.truncation_side == 'left':
            # Truncate from left side.
            # The -2 is to account for the start (<s>) and end (</s>) of sequence tokens
            special_ids = [int(pos) for pos in special_ids][max(len(special_ids) - (self._max_seq_length - 2), 0):]
        else:
            special_ids = [int(pos) for pos in special_ids][:(self._max_seq_length - 2)]

        # The code block below adds a position ID for the start (<s>) and end (</s>) of sequence tokens
        if self._special_token_position_ids is None:
            special_ids.insert(0, 0)
            special_ids.append(0)
        else:
            raise NotImplementedError('Custom position ids for special tokens are not implemented yet')

        if self._pad_to_max_length:
            if self._tokenizer.padding_side == 'right':
                special_ids += [0] * (self._max_seq_length - len(special_ids))
            else:
                raise NotImplementedError('Left sided padding not implemented')

        return special_ids

    def get_special_ids(self, special_ids_list):

        for special_ids in special_ids_list:
            yield self.__get_special_ids(special_ids)

    # Local function to tokenize the inputs
    def tokenize_function(
            self,
            tokens_list: Union[List[List[str]], List[str], str],
            position_ids_list: Union[List[List[str]], List[str], str],
            token_type_ids_list: Union[List[List[str]], List[str], str]
    ):
        """
        Tokenize the input text

        Args:
            tokens_list (Union[List[List[str]], List[str], str]): The batch of input text
            position_ids_list (Union[List[List[str]], List[str], str]): The custom position id's
            token_type_ids_list (Union[List[List[str]], List[str], str]): The custom token type id's

        Returns:
            tokenized_data (): The tokenized data
        """
        tokenized_data = self._tokenizer(
            tokens_list,
            padding=self._padding,
            truncation=self._truncation,
            max_length=self._max_seq_length,
            is_split_into_words=self._is_split_into_words,
            return_special_tokens_mask=True,
            return_token_type_ids=False if token_type_ids_list is not None else True,
        )

        if position_ids_list is not None:
            if type(position_ids_list) == str:
                tokenized_data['position_ids'] = self.__get_special_ids(special_ids=position_ids_list)
            else:
                tokenized_data['position_ids'] = [
                    position_ids for position_ids in self.get_special_ids(special_ids_list=position_ids_list)
                ]

        if token_type_ids_list is not None:
            if type(token_type_ids_list) == str:
                tokenized_data['token_type_ids'] = self.__get_special_ids(special_ids=token_type_ids_list)
            else:
                tokenized_data['token_type_ids'] = [
                    token_type_ids for token_type_ids in self.get_special_ids(special_ids_list=token_type_ids_list)
                ]

        return tokenized_data

    @staticmethod
    def _get_max_seq_length(
            tokenizer: PreTrainedTokenizerFast,
            max_seq_length: Optional[int] = None
    ) -> int:
        """
        Get the max_seq_length based on the model_max_length specified by the tokenizer.
        Verify that the passed max_seq_length is valid and if not, return and valid max_seq_length
        that can be handled by the model/tokenizer.

        Args:
            tokenizer (PreTrainedTokenizerFast): The tokenizer object
            max_seq_length (Optional[int], defaults to `514`): The maximum total input sequence length after
            tokenization. Sequences longer than this will be truncated.

        Returns:
            max_seq_length (int): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated.

        """
        if max_seq_length is None:
            if tokenizer is None:
                return 0
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logging.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` "
                    f"({tokenizer.model_max_length}). Picking 1024 instead. "
                    f"You can change that default value by passing --max_seq_length xxx. "
                )
                max_seq_length = 1024
        else:
            if max_seq_length > tokenizer.model_max_length:
                logging.warning(
                    f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(max_seq_length, tokenizer.model_max_length)
        return max_seq_length
