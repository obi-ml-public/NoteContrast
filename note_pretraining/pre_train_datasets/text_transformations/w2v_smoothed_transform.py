import numpy as np
from scipy.special import softmax
from .transform import Transform


class W2VSmoothedTransform(Transform):
    """
    Class to handle dataset loading (from files and loading to dataset object) and dataset tokenization
    """

    def __init__(
            self,
            gensim_model,
            alpha,
            beta,
            distance_metric,
            reduction,
            text_tokenizer=None,
    ):
        super().__init__(
            text_tokenizer=text_tokenizer,
            subword_tokenizer=None,
            truncation=False,
            pad_to_max_length=False,
            max_seq_length=None,
            is_split_into_words=False
        )
        self._gensim_model = gensim_model
        self.alpha = alpha
        self.beta = beta
        self._distance_metric = distance_metric
        self._reduction = reduction

    def remove_invalid_keys(self, labels):
        return [label for label in labels if self._gensim_model.wv.key_to_index.get(label, None) is not None]

    def pre_tokenize(self, texts):
        if type(texts) == str:
            return self.remove_invalid_keys(texts.split('|'))
        else:
            return [self.remove_invalid_keys(text.split('|')) for text in texts]

    def get_label_indexes(self, labels):
        return [self._gensim_model.wv.key_to_index[label] for label in labels]

    def get_one_hot_labels(self, label_indexes):
        one_hot = np.zeros(len(self._gensim_model.wv.index_to_key), dtype=int)
        one_hot[label_indexes] = 1
        return one_hot

    def get_distances(self, labels):
        if self._distance_metric == 'cosine':
            return np.array([self._gensim_model.wv.distances(label) for label in labels])
        elif self._distance_metric == 'euclidean':
            return np.array(
                [
                    np.linalg.norm(self._gensim_model.wv.get_vector(label) - self._gensim_model.wv.vectors, axis=1)
                    for label in labels
                ]
            )
        else:
            raise ValueError('Invalid distance metric')


    def process_distances(self, distances):
        if self._reduction == 'min':
            distances = distances.min(axis=0)
        elif self._reduction == 'mean':
            distances = distances.mean(axis=0)
        else:
            raise ValueError('Invalid reduction parameter')

        return (distances - distances.mean()) / distances.std()

    def get_softmax(self, labels, one_hot, distances):
        softmax_distances = softmax(-1.0 * self.beta * distances)
        scale = softmax_distances[one_hot.astype(bool)].sum() / (len(one_hot) - len(labels))
        softmax_distances[one_hot.astype(bool)] = 0.0
        softmax_distances[~one_hot.astype(bool)] += scale
        return softmax_distances

    def smooth_labels(self, labels):
        label_indexes = self.get_label_indexes(labels=labels)
        one_hot = self.get_one_hot_labels(label_indexes=label_indexes)
        distances = self.process_distances(self.get_distances(labels=labels))
        softmax_distances = self.get_softmax(labels=labels, distances=distances, one_hot=one_hot)
        return ((1 - self.alpha) * one_hot) + (self.alpha * softmax_distances * len(labels))

    # Local function to tokenize the inputs
    def tokenize_function(self, tokens_list, position_ids_list=None, token_type_ids_list=None):
        if type(tokens_list[0]) == list:
            tokenized_data = {'embedding': [
                self.smooth_labels(labels).tolist() for labels in tokens_list
            ]}
        else:
            tokenized_data = {
                'embedding': self.smooth_labels(tokens_list).tolist()
            }
        return tokenized_data
