from collections import Counter
from typing import Sequence, List

from .evaluation import Evaluation


class MultiClassEvaluation(Evaluation):
    """
    This class is used to evaluate token level precision, recall and F1 scores.
    Script to evaluate at a token level. Calculate precision, recall, and f1 metrics
    at the token level rather than the span level.
    """
    def __init__(self, average_options=('micro', 'macro', 'weighted')):
        super().__init__(average_options)

    def get_counts(self, sequence: Sequence[str], label_list: Sequence[str]) -> List[int]:
        """
        Use this function to get the counts for each label type.

        Args:
            sequence (Sequence[str]): Sequence of values that we want to calculate counts of unique values.
            label_list (Sequence[str]): A list of the label types.

        Returns:
            (List[int]): Position 0 contains the counts for the label type that corresponds to position 0
        """

        counts = Counter()
        counts.update(sequence)
        return [counts[label] for label in label_list]

    def get_optimal_threshold(self, model_predictions, model_labels):
        return None

    def classification_report(
            self,
            labels: Sequence[str],
            predictions: Sequence[str],
            prediction_scores,
            label_list: Sequence[str]
    ) -> dict:
        raise NotImplementedError()
