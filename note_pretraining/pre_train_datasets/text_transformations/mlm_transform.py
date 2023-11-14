class MLMTransform(object):

    def __init__(
            self,
            text_transform,
            text_column_name: str = 'text'
    ):
        self._text_transform = text_transform
        self._text_column_name = text_column_name

    def encode_train(self, batch):
        texts = self._text_transform.pre_tokenize(batch[self._text_column_name])
        return self._text_transform.tokenize_function(tokens_list=texts)

    def encode_validation(self, batch):
        texts = self._text_transform.pre_tokenize(batch[self._text_column_name])
        return self._text_transform.tokenize_function(tokens_list=texts)

    def encode_test(self, batch):
        texts = self._text_transform.pre_tokenize(batch[self._text_column_name])
        return self._text_transform.tokenize_function(tokens_list=texts)

