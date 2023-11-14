from typing import Union, List
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from .subword_tokenizer import SubwordTokenizer


class HFSubwordTokenizer(SubwordTokenizer):
    """
    This class is a wrapper around the HuggingFace tokenizer object. There is a helper function
    that is used to get the subword token from a list of tokens.
    """

    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
    ):
        """
        Args:
            tokenizer (Union[PreTrainedTokenizerFast, PreTrainedTokenizer]): Huggingface tokenizer - this sub-word
            tokenizer is used so that the sequence length does not exceed max_tokens.
        """

        self._tokenizer = tokenizer

    def get_sub_words_from_tokens(
            self,
            tokenized_texts: Union[List[List[str]], List[str]]
    ):
        """
        Get the subwords from the list of tokens using the HuggingFace tokenizer.

        Args:
            tokenized_texts (Union[List[List[str]], List[str]]): Word based tokenized texts.

        Returns:
            (): Tokenized subwords.
        """

        return self._tokenizer.batch_encode_plus(
            tokenized_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
            max_length=None,
            is_split_into_words=True,
            return_token_type_ids=False,
            return_attention_mask=False
        )
