import numpy as np

class MultiLabelTransform(object):
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
        if type(texts) == str:
            return texts.split()
        else:
            return [text.split() for text in texts]

    def get_label_indexes(self, tokens_list):
        return [self._label_to_id[label] for label in tokens_list]

    def get_one_hot_labels(self, tokens_list):
        label_indexes = self.get_label_indexes(tokens_list=tokens_list)
        one_hot = np.zeros(len(self._label_to_id), dtype=int)
        one_hot[label_indexes] = 1
        return one_hot

    # Local function to tokenize the inputs
    def tokenize_function(self, tokens_list):
        if isinstance(tokens_list[0], list):
            return [self.get_one_hot_labels(labels).tolist() for labels in tokens_list]
        else:
            return self.get_one_hot_labels(tokens_list).tolist()
