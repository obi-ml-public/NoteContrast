import numpy as np
from seqeval.reporters import DictReporter
from typing import Sequence, List, Tuple, Union, Optional, Mapping, Any
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix



class Evaluation(object):
    """
    This class is used to evaluate token level precision, recall and F1 scores.
    Script to evaluate at a token level. Calculate precision, recall, and f1 metrics
    at the token level rather than the span level.
    """

    def __init__(self, average_options=('micro', 'macro', 'weighted', None)):
        self._average_options = average_options

    def get_counts(self, sequence: Sequence[str], label_list: Sequence[str]) -> List[int]:
        """
        Use this function to get the counts for each label type.

        Args:
            sequence (Sequence[str]): Sequence of values that we want to calculate counts of unique values.
            label_list (Sequence[str]): A list of the label types.

        Returns:
            (List[int]): Position 0 contains the counts for the label type that corresponds to position 0
        """
        raise NotImplementedError('Implement in subclass')

    @staticmethod
    def get_precision_recall_fscore(
            labels: Sequence[str],
            predictions: Sequence[str],
            label_list: Sequence[str],
            average: Optional[str] = None
    ) -> Tuple[Union[float, List[float]], Union[float, List[float]], Union[float, List[float]]]:
        """
        Use this function to get the token level precision, recall and f-score. Internally we use the
        sklearn precision_score, recall_score and f1 score functions. Also return the count of each
        NER type.

        Args:
            labels (Sequence[str]): A list of the gold standard label labels.
            predictions (Sequence[str]): A list of the predicted label labels.
            label_list (Sequence[str]): A list of the label types.
            average (Optional[str], defaults to `None`): None for per label types scores, or pass an appropriate average
            value.

        Returns:
            eval_precision (Union[float, List[float]]): precision score (averaged or per label type)
            eval_precision (Union[float, List[float]]): recall score (averaged or per label type)
            eval_precision (Union[float, List[float]]): F1 score (averaged or per label type)
            counts (Union[int, List[int]]): Counts (total or per label type)
        """

        eval_precision = precision_score(y_true=labels, y_pred=predictions, labels=label_list, average=average)
        eval_recall = recall_score(y_true=labels, y_pred=predictions, labels=label_list, average=average)
        eval_f1 = f1_score(y_true=labels, y_pred=predictions, labels=label_list, average=average)
        return eval_precision, eval_recall, eval_f1

    def _get_label_list(self, label_list):
        return label_list

    # If evalauting on a subse tof labels - specially for macro averages
    def _get_masked_label_list(self, label_list):
        return label_list

    def precision_recall_fscore(
            self,
            labels: Sequence[str],
            predictions: Sequence[str],
            label_list: Sequence[str],
            average: Optional[str] = None
    ) -> Tuple[Union[float, List[float]], Union[float, List[float]], Union[float, List[float]], Union[int, List[int]]]:
        """
        Use this function to get the token level precision, recall and f-score. Internally we use the
        sklearn precision_score, recall_score and f1 score functions. Also return the count of each
        NER type.

        Args:
            labels (Sequence[str]): A list of the gold standard label labels.
            predictions (Sequence[str]): A list of the predicted label labels.
            label_list (Sequence[str]): A list of the label types.
            average (Optional[str], defaults to `None`): None for per label types scores, or pass an appropriate average
            value.

        Returns:
            eval_precision (Union[float, List[float]]): precision score (averaged or per label type)
            eval_precision (Union[float, List[float]]): recall score (averaged or per label type)
            eval_precision (Union[float, List[float]]): F1 score (averaged or per label type)
            counts (Union[int, List[int]]): Counts (total or per label type)
        """
        eval_precision, eval_recall, eval_f1 = self.get_precision_recall_fscore(
            labels=labels,
            predictions=predictions,
            label_list=label_list,
            average=average
        )
        counts = self.get_counts(sequence=labels, label_list=label_list)
        if average is None:
            eval_precision = list(eval_precision)
            eval_recall = list(eval_recall)
            eval_f1 = list(eval_f1)
        if average is not None:
            counts = sum(counts)
        return eval_precision, eval_recall, eval_f1, counts

    @staticmethod
    def get_confusion_matrix(labels: List[str], predictions: List[str], label_list: List[str]):
        """
        Use this function to get the token level precision, recall and f-score per label type
        and also the micro, macro and weighted averaged precision, recall and f scores.
        Essentially we return a classification report

        Args:
            labels (Sequence[str]): A list of the gold standard label labels
            predictions (Sequence[str]): A list of the predicted label labels
            label_list (List[str]): A list of the label types.

        Returns:
            (): Classification report
        """
        return confusion_matrix(y_true=labels, y_pred=predictions, labels=label_list)

    def get_precision_recall_f1_report(
            self,
            labels: Sequence[str],
            predictions: Sequence[str],
            label_list: Sequence[str]
    ) -> dict:
        """
        Use this function to get the token level precision, recall and f-score per NER type
        and also the micro, macro and weighted averaged precision, recall and f scores.
        Essentially we return a classification report which contains all this information

        Args:
            labels (Sequence[str]): A list of the gold standard NER labels for each note
            predictions (Sequence[str]): A list of the predicted NER labels for each note
            label_list (Sequence[str]): A list of the NER types.

        Returns:
            (dict): Classification report that contains the token level metric scores
        """
        # Store results in this and return this object
        reporter = DictReporter()
        if None in self._average_options:
            masked_label_list = self._get_masked_label_list(label_list)
            # Calculate precision, recall and f1 for each NER type
            eval_precision, eval_recall, eval_f1, counts = self.precision_recall_fscore(
                labels=labels,
                predictions=predictions,
                label_list=masked_label_list,
                average=None
            )

            # Store the results
            for row in zip(np.array(label_list)[masked_label_list], eval_precision, eval_recall, eval_f1, counts):
                reporter.write(*row)
            reporter.write_blank()

        # Calculate the overall precision, recall and f1 - based on the defined averages
        for average in self._average_options:
            if average is None:
                continue
            if average == 'micro':
                masked_label_list = self._get_label_list(label_list)
            else:
                masked_label_list = self._get_masked_label_list(label_list)
            eval_precision, eval_recall, eval_f1, counts = self.precision_recall_fscore(
                labels=labels,
                predictions=predictions,
                label_list=masked_label_list,
                average=average
            )

            # print('F1: ')
            # print(self.macro_f1(np.array(predictions), np.array(labels)))
            # Store the results
            reporter.write('{} avg'.format(average), eval_precision, eval_recall, eval_f1, counts)

        # Add a blank line
        reporter.write_blank()
        # Return the token level results
        return reporter.report()

    def get_optimal_threshold(self, model_predictions, model_labels):
        raise NotImplementedError()

    def classification_report(
            self,
            labels: Sequence[str],
            predictions: Sequence[str],
            prediction_scores,
            label_list: Sequence[str]
    ) -> Mapping[str, Mapping[str, Any]]:
        raise NotImplementedError()


    def macro_precision(self, yhat, y):
        num = self.intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
        return np.mean(num)

    def macro_recall(self, yhat, y):
        num = self.intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
        return np.mean(num)

    def macro_f1(self, yhat, y):
        prec = self.macro_precision(yhat, y)
        rec = self.macro_recall(yhat, y)
        if prec + rec == 0:
            f1 = 0.
        else:
            f1 = 2 * (prec * rec) / (prec + rec)
        return f1

    def intersect_size(self, yhat, y, axis):
        # axis=0 for label-level union (macro). axis=1 for instance-level
        return np.logical_and(yhat, y).sum(axis=axis).astype(float)
