from collections import Counter
from typing import List, Any


def align_token_sub_word_counts(
        sentence_token_object: List[Any],
        word_ids: List[int]
):
    """

    Args:
        sentence_token_object (List[Any]):
        word_ids (List[int]):

    Returns:
    """

    # Count the number of subwords for a given token
    sub_word_counts = Counter()
    for word_idx in word_ids:
        sub_word_counts[word_idx] += 1
    return [
        (token_object, sub_word_counts.get(index, None))
        for index, token_object in enumerate(sentence_token_object)
    ]
