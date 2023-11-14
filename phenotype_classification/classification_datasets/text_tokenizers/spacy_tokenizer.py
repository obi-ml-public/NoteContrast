from typing import NoReturn, List
from spacy import Language, tokens
from .tokenizer import Tokenizer


class SpacyTokenizer(Tokenizer):
    """
    This class is a wrapper around the SpaCy nlp object. It has two helper functions
    to tokenize the input text.
    """

    def __init__(
            self,
            nlp: Language,
    ) -> NoReturn:
        """
        Initialize the NLP object

        Args:
            nlp (Language): The spacy nlp object that is used to tokenize the text and split documents
            into sentences.
        """

        self._nlp = nlp

    def get_sentence_tokens(self, text: str) -> List[List[tokens.token.Token]]:
        """
        Sentencize the text and return the tokens in each sentence and return all sentences
        Nested list - Outer list represents the sentences, and the inner list represents
        the tokens in the sentences.

        Args:
            text (str): The text to be sentencized and tokenized.

        Returns:
            (List[List[tokens.token.Token]]): Nested list - Outer list represents the sentences, and the inner list
            represents the tokens in the sentences.
        """

        # Spacy DOC object
        document = self._nlp(text)
        return [
            [token for token in sentence if token.text.strip() not in ['\n', '\t', ' ', '']]
            for sentence in document.sents if sentence.text.strip() not in ['\n', '\t', ' ', '']
        ]

    def get_tokens(self, text) -> List[tokens.token.Token]:
        """
        Tokenize the text and return the tokens in the text.

        Args:
            text (str): The text to be tokenized

        Returns:
            (List[tokens.token.Token]): The list of tokens in the text
        """

        # Spacy DOC object
        document = self._nlp(text)
        return [
            token
            for sentence in document.sents if sentence.text.strip() not in ['\n', '\t', ' ', '']
            for token in sentence if token.text.strip() not in ['\n', '\t', ' ', '']
        ]
