from typing import Optional


class M2MTransform(object):

    def __init__(
            self,
            text_transform,
            label_transform,
            text_column_name: str = 'text',
            label_column_name: str = 'label',
            position_ids_column_name: Optional[str] = None,
            token_type_ids_column_name: Optional[str] = None
    ):
        self._text_transform = text_transform
        self._label_transform = label_transform
        self._text_column_name = text_column_name
        self._label_column_name = label_column_name
        self._position_ids_column_name = position_ids_column_name
        self._token_type_ids_column_name = token_type_ids_column_name

    def encode_train(self, batch):
        texts = self._text_transform.pre_tokenize(batch[self._text_column_name])
        labels = self._label_transform.pre_tokenize(batch[self._label_column_name])
        position_ids_list = None if self._position_ids_column_name is None else batch[self._position_ids_column_name]
        token_type_ids_list = None if self._token_type_ids_column_name is None \
            else batch[self._token_type_ids_column_name]

        return {
            'm2m_texts': self._text_transform.tokenize_function(tokens_list=texts),
            'm2m_labels': self._label_transform.tokenize_function(
                tokens_list=labels,
                position_ids_list=position_ids_list,
                token_type_ids_list=token_type_ids_list
            )
        }

    def encode_validation(self, batch):
        texts = self._text_transform.pre_tokenize(batch[self._text_column_name])
        labels = self._label_transform.pre_tokenize(batch[self._label_column_name])
        position_ids_list = None if self._position_ids_column_name is None else batch[self._position_ids_column_name]
        token_type_ids_list = None if self._token_type_ids_column_name is None \
            else batch[self._token_type_ids_column_name]

        return {
            'm2m_texts': self._text_transform.tokenize_function(tokens_list=texts),
            'm2m_labels': self._label_transform.tokenize_function(
                tokens_list=labels,
                position_ids_list=position_ids_list,
                token_type_ids_list=token_type_ids_list
            )
        }

    def encode_test(self, batch):
        raise NotImplementedError('TODO')
