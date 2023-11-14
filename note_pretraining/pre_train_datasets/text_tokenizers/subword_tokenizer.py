
class SubwordTokenizer(object):
    """
    Parent sub-word tokenizer wrapper class that defines the functions that needs
    to be implemented by any of the other child sub-word tokenizer classes
    """

    def get_sub_words_from_tokens(self, tokenized_texts):
        raise NotImplementedError('Function needs to be implemented')
