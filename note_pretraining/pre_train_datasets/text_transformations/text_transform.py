from typing import Optional

from .transform import Transform


class TextTransform(Transform):
    """
    Class to handle dataset loading (from files and loading to dataset object) and dataset tokenization
    """

    def __init__(
            self,
            text_tokenizer,
            subword_tokenizer,
            truncation: bool = True,
            pad_to_max_length: bool = False,
            max_seq_length: Optional[int] = 4096,
            is_split_into_words: bool = False,
            do_mlm=True,
    ):
        super().__init__(
            text_tokenizer,
            subword_tokenizer,
            truncation,
            pad_to_max_length,
            max_seq_length,
            is_split_into_words
        )
        self._do_mlm = do_mlm

    def pre_tokenize(self, texts):
        if self._text_tokenizer is None:
            return texts
        else:
            raise NotImplementedError()
            # return [
            #     self.get_token_text(
            #         tokens=self._text_tokenizer.get_tokens(text=text.replace('\n', ' '))
            #     ) for text in texts
            # ]

    # Local function to tokenize the inputs
    def tokenize_function(self, tokens_list, position_ids_list=None, token_type_ids_list=None):

        tokenized_data = self._subword_tokenizer(
            tokens_list,
            padding=self._padding,
            truncation=self._truncation,
            max_length=self._max_seq_length,
            is_split_into_words=self._is_split_into_words,
            return_special_tokens_mask=self._do_mlm,
        )

        return tokenized_data
