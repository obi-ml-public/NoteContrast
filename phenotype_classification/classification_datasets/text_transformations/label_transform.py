class LabelTransform(object):
    """
    Class to handle dataset loading (from files and loading to dataset object) and dataset tokenization
    """

    def __init__(
            self,
            label_to_id
    ):
        self._label_to_id = label_to_id

    @staticmethod
    def pre_tokenize(texts):
        return texts

    # Local function to tokenize the inputs
    def tokenize_function(self, tokens_list):
        if isinstance(tokens_list, list):
            return [self._label_to_id[label] for label in tokens_list]
        else:
            return self._label_to_id[tokens_list]
