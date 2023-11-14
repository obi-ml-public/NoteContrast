from .transform import Transform


class W2VTransform(Transform):
    """
    Class to handle dataset loading (from files and loading to dataset object) and dataset tokenization
    """

    def __init__(
            self,
            gensim_model,
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

    def pre_tokenize(self, texts):
        if type(texts) == str:
            return texts.split('|')
        else:
            return [text.split('|') for text in texts]

    # Local function to tokenize the inputs
    def tokenize_function(self, tokens_list, position_ids_list=None, token_type_ids_list=None):
        if type(tokens_list[0]) == list:
            tokenized_data = {'embedding': [
                self._gensim_model.wv.get_mean_vector(labels, pre_normalize=False).tolist() for labels in tokens_list
            ]}
        else:
            tokenized_data = {
                'embedding': self._gensim_model.wv.get_mean_vector(tokens_list, pre_normalize=False).tolist()
            }
        return tokenized_data
