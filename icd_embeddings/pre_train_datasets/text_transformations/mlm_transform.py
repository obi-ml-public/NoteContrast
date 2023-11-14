from typing import Optional


class MLMTransform(object):

    def __init__(
            self,
            transform,
            text_column_name: str,
            position_ids_column_name: Optional[str],
            token_type_ids_column_name: Optional[str]
    ):
        self._transform = transform
        self._text_column_name = text_column_name
        self._position_ids_column_name = position_ids_column_name
        self._token_type_ids_column_name = token_type_ids_column_name

    def encode_train(self, batch):
        icd_codes = self._transform.pre_tokenize(batch[self._text_column_name])
        position_ids_list = None if self._position_ids_column_name is None else batch[self._position_ids_column_name]
        token_type_ids_list = None if self._token_type_ids_column_name is None \
            else batch[self._token_type_ids_column_name]

        return self._transform.tokenize_function(
                tokens_list=icd_codes,
                position_ids_list=position_ids_list,
                token_type_ids_list=token_type_ids_list
        )

    def encode_validation(self, batch):
        return self.encode_train(batch)

    def encode_test(self, batch):
        return self.encode_train(batch)
