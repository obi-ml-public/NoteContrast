from typing import Optional

from .transform import Transform


class LabelTransform(Transform):
    """
    Class to handle dataset loading (from files and loading to dataset object) and dataset tokenization
    """

    def __init__(
            self,
            text_tokenizer,
            subword_tokenizer,
            truncation: bool = True,
            pad_to_max_length: bool = False,
            max_seq_length: Optional[int] = 512,
            is_split_into_words: bool = False,
            special_token_position_ids: Optional[float] = None
    ):
        super().__init__(
            text_tokenizer,
            subword_tokenizer,
            truncation,
            pad_to_max_length,
            max_seq_length,
            is_split_into_words
        )
        self._special_token_position_ids = special_token_position_ids

    def pre_tokenize(self, texts):
        if self._text_tokenizer is None:
            return texts
        else:
            raise NotImplementedError('TODO')

    def __get_special_ids(self, special_ids):

        # Split the input string by whitespace
        # If it is already a split list, leave as is
        if type(special_ids) == str:
            special_ids = special_ids.split()

        # Convert the str to float.
        # Since we truncate our input text based on the max_seq_length, we also need to
        # truncate the position_ids. Our tokenizer handles the truncation of the input text,
        # but we need to manually handle the truncation of the position ids
        if self._subword_tokenizer.truncation_side == 'left':
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
            if self._subword_tokenizer.padding_side == 'right':
                special_ids += [0] * (self._max_seq_length - len(special_ids))
            else:
                raise NotImplementedError('Left sided padding not implemented')

        return special_ids

    def get_special_ids(self, special_ids_list):

        for special_ids in special_ids_list:
            yield self.__get_special_ids(special_ids)


    # Local function to tokenize the inputs
    def tokenize_function(self, tokens_list, position_ids_list=None, token_type_ids_list=None):

        if self._subword_tokenizer is None:
            return tokens_list

        tokenized_data = self._subword_tokenizer(
            tokens_list,
            padding=self._padding,
            truncation=self._truncation,
            max_length=self._max_seq_length,
            is_split_into_words=self._is_split_into_words,
            return_special_tokens_mask=False,
            return_token_type_ids=False if token_type_ids_list is not None else True,
        )

        # The code fails when we don't do this step. I'm not sure why the code fails.
        tokenized_data = {k:v for k, v in tokenized_data.items()}

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
