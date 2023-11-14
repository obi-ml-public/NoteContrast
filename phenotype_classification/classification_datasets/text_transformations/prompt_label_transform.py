import numpy as np

class PromptLabelTransform(object):
    """
    Class to handle dataset loading (from files and loading to dataset object) and dataset tokenization
    """

    def __init__(
            self,
            label_to_id,
            mask_token_id,
    ):
        self._label_to_id = label_to_id
        self._mask_token_id = mask_token_id

    def pre_tokenize(self, texts):
        if type(texts) == str:
            return texts.split()
        else:
            return [text.split() for text in texts]

    def get_label_to_id(self, prompt_labels=None):
        return self._label_to_id

    def get_label_vector(self, labels, input_ids, prompt_labels=None):
        label_to_id = self.get_label_to_id(prompt_labels=prompt_labels)
        label_indexes = [label_to_id[label] for label in labels]
        label_vector = np.zeros(len(input_ids), dtype=int)
        label_vector += -100
        one_hot = np.zeros(len(label_to_id), dtype=int)
        one_hot[label_indexes] = 1
        mask = np.array(input_ids) == self._mask_token_id
        label_vector[mask] = one_hot
        return label_vector

    # Local function to tokenize the inputs
    def tokenize_function(self, tokens_list, input_ids_list):
        if isinstance(tokens_list[0], list):
            return [
                self.get_label_vector(
                    labels=labels,
                    input_ids=input_ids
                ).tolist() for labels, input_ids in zip(tokens_list, input_ids_list)
            ]
        else:
            return self.get_label_vector(labels=tokens_list, input_ids=input_ids_list).tolist()