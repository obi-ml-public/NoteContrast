import numpy as np
from collections import Counter
from typing import Sequence, List, Mapping, Any
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve

from .evaluation import Evaluation


class BinaryClassEvaluation(Evaluation):
    """
    This class is used to evaluate token level precision, recall and F1 scores.
    Script to evaluate at a token level. Calculate precision, recall, and f1 metrics
    at the token level rather than the span level.
    """

    def __init__(self, average_options=('micro', 'macro', 'weighted', None)):
        super().__init__(average_options)

    def classification_report(
            self,
            labels: Sequence[str],
            predictions: Sequence[str],
            prediction_scores,
            label_list: Sequence[str],
    ) -> Mapping[str, Mapping[str, Any]]:

        precision_recall_f1_report = self.get_precision_recall_f1_report(
            labels=labels,
            predictions=predictions,
            label_list=label_list,
        )

        precision_recall_f1_report.pop(0)
        precision_recall_f1_report.pop('micro avg')
        precision_recall_f1_report.pop('macro avg')
        precision_recall_f1_report.pop('weighted avg')
        precision_recall_f1_report['pos'] = precision_recall_f1_report.pop(1)

        auc_metrics = self.get_auc_metrics(prediction_scores=prediction_scores, labels=labels)
        auprc_metrics = self.get_auprc_metrics(prediction_scores=prediction_scores, labels=labels)
        matthews_correlation_coefficient = self.get_matthews_correlation_coefficient(
            predictions=predictions, labels=labels
        )

        return {**precision_recall_f1_report, **auc_metrics, **auprc_metrics, **matthews_correlation_coefficient}


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

    @staticmethod
    def get_matthews_correlation_coefficient(predictions, labels):
        return {
            'matthews_correlation_coefficient': {
                "pos": matthews_corrcoef(y_true=labels, y_pred=predictions)
            }
        }

    @staticmethod
    def get_auc_metrics(prediction_scores, labels):
        return {
            'auc': {
                "pos": roc_auc_score(y_true=labels, y_score=prediction_scores),
            }
        }

    @staticmethod
    def get_auprc_metrics(prediction_scores, labels):
        return {
            'auprc': {
                "pos": average_precision_score(y_true=labels, y_score=prediction_scores),
            }
        }

    def get_optimal_threshold(self, model_predictions, model_labels):
        return self.get_optimal_f1_threshold(model_predictions, model_labels)

    @staticmethod
    def get_optimal_f1_threshold(model_predictions, model_labels):
        precision, recall, thresholds = precision_recall_curve(
            np.array(model_labels).reshape(-1),
            np.array(model_predictions).reshape(-1)
        )
        fscore = (2 * precision * recall) / (precision + recall)
        return thresholds[np.argmax(fscore)]

    @staticmethod
    def get_optimal_auc_threshold(model_predictions, model_labels):
        fpr, tpr, thresholds = roc_curve(
            np.array(model_labels).reshape(-1),
            np.array(model_predictions).reshape(-1)
        )

        # calculate the g-mean for each threshold
        geometric_means = np.sqrt(tpr * (1 - fpr))
        # locate the index of the largest g-mean
        return thresholds[np.argmax(geometric_means)]
