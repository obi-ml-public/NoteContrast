import logging
from typing import Optional

from transformers import PreTrainedTokenizerFast


class Transform(object):
    """
    Class to handle dataset loading (from files and loading to dataset object) and dataset tokenization
    """

    def __init__(
            self,
            text_tokenizer,
            subword_tokenizer,
            truncation: bool,
            pad_to_max_length: bool,
            max_seq_length: Optional[int],
            is_split_into_words: bool
    ):
        self._text_tokenizer = text_tokenizer
        self._subword_tokenizer = subword_tokenizer
        self._truncation = truncation
        self._padding = "max_length" if pad_to_max_length else False
        self._pad_to_max_length = pad_to_max_length
        self._max_seq_length = self._get_max_seq_length(tokenizer=subword_tokenizer, max_seq_length=max_seq_length)
        self._is_split_into_words = is_split_into_words

    # Local function to tokenize the inputs
    def tokenize_function(self, tokens_list):
        raise NotImplementedError('This function needs to be implemented by the subclass')

    def pre_tokenize(self, texts):
        raise NotImplementedError('This function needs to be implemented by the subclass')

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
