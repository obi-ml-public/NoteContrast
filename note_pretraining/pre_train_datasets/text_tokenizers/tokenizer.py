
class Tokenizer(object):
    """
    Parent tokenizer wrapper class that defines the functions that needs
    to be implemented by any of the other child tokenizer classes
    """

    def get_sentence_tokens(self, text: str):
        """
        Sentencize the text and return the tokens in each sentence and return all sentences
        Nested list - Outer list represents the sentences, and the inner list represents
        the tokens in the sentences.

        Args:
            text (str): The text to be sentencized and tokenized.

        Returns:
            (List[List[Any]]): Nested list - Outer list represents the sentences, and the inner list
            represents the tokens in the sentences.
        """

        raise NotImplementedError('Function needs to be implemented')

    def get_tokens(self, text: str):
        """
        Tokenize the text and return the tokens in the text.

        Args:
            text (str): The text to be tokenized

        Returns:
            (List[Any]): The list of tokens in the text
        """

        raise NotImplementedError('Function needs to be implemented')
