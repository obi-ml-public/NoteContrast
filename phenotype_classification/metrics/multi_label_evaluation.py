import numpy as np
from collections import Counter
from typing import Sequence, List
from sklearn.metrics import precision_recall_curve
from .evaluation import Evaluation
from sklearn.metrics import roc_auc_score


class MultiLabelEvaluation(Evaluation):
    """
    This class is used to evaluate token level precision, recall and F1 scores.
    Script to evaluate at a token level. Calculate precision, recall, and f1 metrics
    at the token level rather than the span level.
    """

    def __init__(self, average_options=('micro', 'macro', 'weighted', None)):
        super().__init__(average_options)
        self._label_mask = None

    @staticmethod
    def unpack_nested_list(nested_list: Sequence[Sequence[str]]) -> List[str]:
        """

        Args:
            nested_list (Sequence[Sequence[str]]): A nested list of predictions/labels

        Returns:
            (List[str]): Unpacked nested list of predictions/labels
        """

        return [inner for nested in nested_list for inner in nested]

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
        counts.update(self.unpack_nested_list(sequence))
        return [counts[label] for label in label_list]

    def _get_label_list(self, label_list):
        return np.arange(len(label_list))

    def _get_masked_label_list(self, label_list):
        if self._label_mask is None:
            return np.arange(len(label_list))
        else:
            return np.arange(len(label_list))
            # return np.arange(len(label_list))[self._label_mask]

    def get_auc_metrics(self, prediction_scores, labels):
        try:
            return {
                'auc': {
                    "macro_avg": roc_auc_score(y_true=labels, y_score=prediction_scores, average='macro'),
                    "micro_avg": roc_auc_score(y_true=labels, y_score=prediction_scores, average='micro')
                }
            }
        except ValueError as ve:
            if 'Only one class present in y_true. ROC AUC score is not defined in that case.' in ve.args:
                if self._label_mask is None:
                    return {
                        'auc': {
                            "macro_avg": 0.0,
                            "micro_avg": roc_auc_score(y_true=labels, y_score=prediction_scores, average='micro')
                        }
                    }
                else:
                    return {
                        'auc': {
                            "macro_avg": (
                                roc_auc_score(
                                    y_true=labels[:, self._label_mask],
                                    y_score=prediction_scores[:, self._label_mask],
                                    average='macro'
                                )
                            ),
                            "micro_avg": roc_auc_score(y_true=labels, y_score=prediction_scores, average='micro')
                        }
                    }

            else:
                raise ValueError()

    def get_precision_at_k(self, prediction_scores, labels, ks):
        precision_at_k = {k: self.precision_at_k(np.array(prediction_scores), np.array(labels), k=k) for k in ks}
        return {'precision_at': precision_at_k}

    def get_recall_at_k(self, prediction_scores, labels, ks):
        recall_at_k = {k: self.recall_at_k(np.array(prediction_scores), np.array(labels), k=k) for k in ks}
        return {'recall_at': recall_at_k}

    @staticmethod
    def precision_at_k(yhat_raw, y, k):
        # num true labels in top k predictions / k
        sortd = np.argsort(yhat_raw)[:, ::-1]
        top_k = sortd[:, :k]

        # get precision at k for each example
        vals = []
        for i, tk in enumerate(top_k):
            if len(tk) > 0:
                num_true_in_top_k = y[i, tk].sum()
                denominator = len(tk)
                vals.append(num_true_in_top_k / float(denominator))

        return np.mean(vals)

    @staticmethod
    def recall_at_k(yhat_raw, y, k):
        # num true labels in top k predictions / num true labels
        sortd = np.argsort(yhat_raw)[:, ::-1]
        topk = sortd[:, :k]

        # get recall at k for each example
        vals = []
        for i, tk in enumerate(topk):
            num_true_in_top_k = y[i, tk].sum()
            denom = y[i, :].sum()
            vals.append(num_true_in_top_k / float(denom))

        vals = np.array(vals)
        vals[np.isnan(vals)] = 0.

        return np.mean(vals)

    def get_optimal_threshold(self, model_predictions, model_labels):
        return self.get_optimal_f1_threshold(model_predictions, model_labels)

    @staticmethod
    def get_optimal_f1_threshold(model_predictions, model_labels):
        precision, recall, thresholds = precision_recall_curve(
            np.array(model_labels).reshape(-1),
            np.array(model_predictions).reshape(-1)
        )

        fscore = (2.0 * precision * recall) / (precision + recall)
        return thresholds[np.nanargmax(fscore)]

    def set_label_mask(self, label_mask):
        self._label_mask = label_mask

    def classification_report(
            self,
            labels: Sequence[str],
            predictions: Sequence[str],
            prediction_scores,
            label_list: Sequence[str],
    ) -> dict:

        precision_recall_f1_report = self.get_precision_recall_f1_report(
            labels=labels,
            predictions=predictions,
            label_list=label_list,
        )

        auc_metrics = self.get_auc_metrics(prediction_scores=prediction_scores, labels=labels)
        precision_at_k = self.get_precision_at_k(prediction_scores=prediction_scores, labels=labels, ks=[5, 8, 15])
        recall_at_k = self.get_recall_at_k(prediction_scores=prediction_scores, labels=labels, ks=[5, 8, 15])

        return {**precision_recall_f1_report, **auc_metrics, **precision_at_k, **recall_at_k}


