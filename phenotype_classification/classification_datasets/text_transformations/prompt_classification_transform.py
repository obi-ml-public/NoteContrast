from typing import Optional


class PromptClassificationTransformation(object):

    def __init__(
            self,
            text_transform,
            label_transform,
            text_column_name: str = 'text',
            label_column_name: str = 'label',
            model_type: Optional[str] = None
    ):
        self._text_transform = text_transform
        self._label_transform = label_transform
        self._text_column_name = text_column_name
        self._label_column_name = label_column_name
        self._model_type = model_type

    def encode_train(self, batch):
        texts = self._text_transform.pre_tokenize(batch[self._text_column_name])
        labels = self._label_transform.pre_tokenize(batch[self._label_column_name])
        tokenized_data = self._text_transform.tokenize_function(tokens_list=texts)
        tokenized_data['labels'] = self._label_transform.tokenize_function(
            tokens_list=labels,
            input_ids_list=tokenized_data['input_ids']
        )
        return tokenized_data

    def encode_validation(self, batch):
        return self.encode_train(batch)

    def encode_test(self, batch):
        raise NotImplementedError('TODO')
