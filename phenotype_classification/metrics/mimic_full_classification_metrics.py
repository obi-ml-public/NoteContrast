import itertools
import numpy as np
from .classification_metrics import ClassificationMetrics

class MimicFullClassificationMetrics(ClassificationMetrics):

    def __init__(
            self,
            post_process,
            label_list,
            evaluator,
            note_labels,
            prompt_labels,
            group_size,
            optimal_f1_threshold=None
    ):
        super().__init__(post_process, label_list, evaluator, optimal_f1_threshold)
        self._label_to_id = {label: index for index, label in enumerate(self._label_list)}
        self._group_size =  group_size
        self._length = len(note_labels)
        unique_labels = {label for note_label in note_labels for label in note_label}
        label_mask = self.get_one_hot_labels(unique_labels).astype(bool)
        self._evaluator.set_label_mask(label_mask)
        self._labels = np.array([self.get_one_hot_labels(note_label) for note_label in note_labels])
        self._indexes = np.array([self.get_indexes(prompt_label) for prompt_label in prompt_labels])

    def aggregate_list(self, input_list):
        return [
            list(itertools.chain.from_iterable(input_list[i:i + self._group_size]))
            for i in range(0, len(input_list), self._group_size)
        ]

    def get_one_hot_labels(self, labels):
        label_indexes = [self._label_to_id[label] for label in labels]
        one_hot = np.zeros(len(self._label_to_id), dtype=int)
        one_hot[label_indexes] = 1
        return one_hot

    def get_indexes(self, labels):
        return [self._label_to_id[label] for label in labels]

    def compute_multi_label_metrics(self, model_output):

        prediction_scores, predictions, labels = self._post_process.decode(
            model_predictions=model_output.predictions, model_labels=model_output.label_ids
        )

        prediction_scores = self.aggregate_list(prediction_scores)
        predictions = self.aggregate_list(predictions)
        minimum_prediction_score = -2 * np.abs(np.min(prediction_scores))

        prediction_scores_aggregated = (
                np.zeros((self._length, len(self._label_list)), dtype=float) + minimum_prediction_score
        )
        predictions_aggregated = np.zeros((self._length, len(self._label_list)), dtype=int)

        for i in range(self._length):
            prediction_scores_aggregated[i, self._indexes[i]] = prediction_scores[i]
            predictions_aggregated[i, self._indexes[i]] = predictions[i]

        if self._optimal_f1_threshold:
            optimal_f1_threshold = self._evaluator.get_optimal_threshold(
                prediction_scores_aggregated, self._labels
            )
            predictions_aggregated = (prediction_scores_aggregated > optimal_f1_threshold).astype(int)
        else:
            optimal_f1_threshold = -100

        metrics_report = self.get_metrics_report(
            prediction_scores=prediction_scores_aggregated,
            predictions=predictions_aggregated,
            labels=self._labels
        )

        results = self.get_metrics_dictionary(metrics_report=metrics_report)

        return self.format_results(
            results={**results, **{'optimal_threshold':optimal_f1_threshold}}
        )
